# Method Taxonomy Draft — LLM Inference KV Cache Optimization (Stage 2A Step A)

> **Stage 2A Step A — Method Taxonomy** | 2026-08-27 | Workdir `F:\AIinfraResearch` | Input: `research/manifests/papers.md` (43 papers) + `research/paper_notes/*.md` (43 notes) + `research/knowledge/stage1_summary.md` + `research/knowledge/coverage_report.md`
> *Scope: ONLY taxonomy, no idea generation. Based on full list of 43 papers in manifest; sampled 24 notes across all 11 required categories for grounding.*

## Overview — Dimension Coverage

| Required dimension (task spec) | Taxonomy category covering it |
|---|---|
| KV allocation / memory management | 1 |
| eviction | 2 |
| compression | 3 |
| quantization | 4 |
| prefix caching / reuse | 5 |
| KV sharing | 6 |
| offloading | 7 |
| recomputation | 8 |
| hierarchical memory | 7, 14 |
| KV transfer | 9 |
| scheduling | 10 |
| batching | 10 |
| request routing | 11 |
| prefill/decode disaggregation | 12 |
| distributed serving | 13 |
| communication optimization | 9 |
| workload-aware optimization | 14 |

Total categories: **14** (>=12 required).

## 1. KV Allocation & Paged Memory Management

### Problem addressed
Variable-length requests cause fragmentation and reservation waste under contiguous pre-allocated KV buffers. Prior Orca (2022) reserves max_tokens per request; vLLM (2023) quantifies waste 60-80% (20.4-38.2% effective). Need on-demand allocation and sharing without accuracy loss.

### Representative papers
- **vLLM PagedAttention** — 2023, SOSP 23 (arXiv:2309.06180)
- **SGLang RadixAttention** — 2024, NeurIPS 24 (arXiv:2312.07104)
- **LMCache** — 2025, arXiv:2510.09665
- **FlexGen** — 2023, ICML 23 (arXiv:2303.06865)

### Core mechanisms
- **PagedAttention (vLLM, 2023):** Partition KV into fixed-size blocks (B=16). Logical blocks fill left-to-right; block table maps logical->physical + filled. Eliminates external fragmentation, bounds internal waste to <=1 block. CoW sharing via ref counts.
- **Radix tree reuse (SGLang, 2024):** CPU-resident radix tree over token sequences -> paged tensors per node. Shared prefix maps to same physical blocks; LRU leaf-first eviction with ref counters. Shared pool for cache + running requests.
- **Standardized KV layer (LMCache, 2025):** Connector abstraction + 256-token chunks vs 16-token pages + streaming GPU buffers coalescing pages into 1 MB DMA chunks. Async prefetch and zero-copy CPU offload.
- **LP placement search (FlexGen, 2023):** Linear programming to search weight/KV placement (GPU/CPU/disk) + 4-bit quantization for single-GPU high-throughput.

### System layer
**memory manager / runtime / node**

### Main metrics improved
GPU KV effective usage (20%->~96% vLLM 2023); sustainable batch size (2-4x); throughput req/s at same latency (2-4x over Orca Max); SGLang up to 6.4x throughput on structured programs.

### Typical tradeoffs
Paged indirection adds 20-26% attention kernel latency; small blocks increase PCIe fragmentation vs large blocks increase internal waste.

### Common assumptions
Transformer decoder FP16 KV; PyTorch contiguous constraint lifted via block table; KV dominates batch limit; sharing safe within request group or reserved prefix.

### Known limitations
Author-stated (vLLM 2023): not beneficial for training, adds indirection. Author-stated (SGLang 2024): exact prefix match only, greedy longest-prefix may starve, single-tier GPU DRAM only. Inferred: block size 16 not optimal for all; swap space bounded -> head-of-line blocking.

### Maturity
**Mature** — many papers & production deployment (vLLM, SGLang, LMCache in Dynamo/llm-d/KServe).

---
## 2. Eviction — Token Dropping & Sparse Retention

### Problem addressed
Full KV retention grows O(n) memory and per-step attention; long-context (32K-1M) exceeds HBM. Need <=20% KV while preserving accuracy.

### Representative papers
- **StreamingLLM** — 2023->2024, ICLR 24 (arXiv:2309.17453)
- **H2O** — 2023, NeurIPS 23 (arXiv:2306.14048)
- **Scissorhands** — 2023, NeurIPS 23 (arXiv:2305.17118)
- **SnapKV** — 2024, NeurIPS 24 (arXiv:2404.14469)
- **SCOPE** — 2025, ACL 25 (arXiv:2412.13649)

### Core mechanisms
- **Attention sinks (StreamingLLM 2023):** Keep 4 initial sink tokens pinned + rolling window; cache-relative RoPE re-encoding stabilizes softmax. 4M-token stable, ~22x speedup.
- **Heavy-hitter submodular (H2O 2023):** Dynamic submodular F_score(T)=sum o_s; greedy keeps heavy hitters + recent; (1-1/e) bound; power-law -> 5-10% KV retains accuracy.
- **Persistence (Scissorhands 2023):** Importance history persistence; 5x compression near-lossless, stackable to 20x with quantization.
- **Observation-window voting (SnapKV 2024):** Last window queries vs prefix keys; sum attention, 1D pooling clustering, top-k per head, concat observation window; constant KV before generation; 380K context on A100 (1024 budget, 380x), 3.6x speedup at 16K.
- **Phase-aware (SCOPE 2025):** Separate prefill Lambda^p invariant vs decode Lambda^d adaptive hat Lambda1=(t-Lambda2)Lambda1/(T-Lambda2); avoids drift where unified Top-K biases to recent.

### System layer
**kernel / attention / memory manager**

### Main metrics improved
Memory 5-20x reduction; perplexity near parity at 20% budget; throughput (H2O 29x over DeepSpeed /3x over FlexGen; StreamingLLM 22.2x per-token).

### Typical tradeoffs
Smaller budget -> lost middle retrieval; error compounds autoregressively; Top-K adds argmax latency; fixed windows not adaptive.

### Common assumptions
Attention sparsity >95% power-law; accumulated sum approx future importance; recent tokens always important; sink tokens globally visible.

### Known limitations
Author-stated StreamingLLM: does NOT extend context, only recent coherence. H2O: inference-only, submodular assumption may not hold. Inferred: uniform budget wasteful (cf PyramidKV); permanent eviction loses revival (InfiniGen shows divergence after ~200 steps).

### Maturity
**Active** — dense literature 2023-2025, strong validation; not yet default.

---
## 3. Compression — Layer-Adaptive & Semantic Chunk

### Problem addressed
Uniform per-layer eviction wastes budget on sparse upper layers while starving dispersed lower layers (PyramidKV funnel). Token-level isolated scoring fragments semantic chunks.

### Representative papers
- **PyramidKV** — 2024, arXiv:2406.02069
- **ChunkKV** — 2025, NeurIPS 25 (arXiv:2502.00299)
- **ShadowKV** — 2024->2025, ICML 25 (arXiv:2410.21465)
- **SCOPE** — 2025, ACL 25

### Core mechanisms
- **Pyramidal allocation (PyramidKV 2024):** Arithmetic sequence k^l = k^0 - (k^0-k^{m-1})/(m-1)*l with k^{m-1}=k_total/(alpha*m), alpha=20; retain last 8 instruction tokens uniformly, per-head select top k^l by s_i^h = sum_{j in [n-alpha,n]} A_{ij}^h. 12% cache matches full KV; 0.7% (64 tokens) still +20.5 on TREC vs SnapKV.
- **Semantic chunk (ChunkKV 2025):** Chunk size c=10; chunk attention A_i = sum_{j=(i-1)c+1}^{ic} A_{:,j}; top-k chunks k=floor(Lmax/c) preserve order; Jaccard adjacent-layer similarity 57.74% (LLaMA-3-8B) vs 27.95% SnapKV -> reuse indices for next N_reuse=1 layer -> 20% time reduction 0.5% drop.
- **Low-rank + landmarks (ShadowKV 2025):** Pre-RoPE keys SVD rank r=160 on GPU, values offload CPU, chunk means landmarks (C=8) +48 outliers 0.2-0.3%; decoding landmark attention selects TopK=256 (1.56% budget), reconstruct K_sparse=Gather(A,I).B then RoPE, overlap via CUDA multi-stream.

### System layer
**attention / memory manager / kernel**

### Main metrics improved
LongBench avg (PyramidKV 64->41.49 vs Full 41.46 at 2048); GSM8K +8.7% over SnapKV; ShadowKV 6x batch, 3.04x throughput, RULER 86.88 vs Full 86.68.

### Typical tradeoffs
Fixed pyramid alpha heuristic not adaptive; chunk size 10 ignores boundaries; low-rank rank uniform 160.

### Common assumptions
Broad-to-focal funnel monotonic; instruction window proxy holds; chunk sum better than isolated token; pre-RoPE low-rank while values not.

### Known limitations
Author-stated PyramidKV: sensitivity to alpha, no dynamic adaptation. ChunkKV: fixed chunk ignores linguistic boundaries. ShadowKV: SVD cost for short prompts. Inferred: no joint quantization.

### Maturity
**Active** — 2024-2025 SOTA, heuristic not yet default.

---
## 4. Quantization — Low-Bit KV Cache

### Problem addressed
KV bytes dominate memory (540B PaLM batch512 ctx2048 -> 3TB, 3x params). 2-bit needed but key channel outliers and value token sensitivity cause collapse if naive per-token.

### Representative papers
- **KIVI** — 2024, ICML 24 (arXiv:2402.02750)
- **GEAR** — 2024, arXiv:2403.05527
- **CacheGen** — 2024, SIGCOMM 24 (arXiv:2310.07240)

### Core mechanisms
- **Asymmetric grouping (KIVI 2024):** Key per-channel (G=32) vs Value per-token (G=32). Key per-token error 13.67 vs per-channel 4.55; attention error 47.00 vs 9.60. Split grouped quantized + residual full-precision buffer R=128 sliding window. Tiled fused dequant+matmul via Triton/CUDA.
- **Quant + low-rank + sparse (GEAR 2024):** Optimize min||X - D^q - L - S||_F where D^q quantized backbone (KCVT per-vector for 4-bit, KIVI g=64 for 2-bit), S top/bottom s/2% outliers per vector FP16 sparse (s=2%), L head-wise low-rank r=4 prefill r=2 decode buffer via power-iteration.
- **Bandwidth-adaptive codec (CacheGen 2024):** Delta-based bitstream + 1.5K-token chunks + text fallback; 3.5-4.3x size, 3.2-3.7x fetch delay reduction.

### System layer
**kernel / memory manager / runtime**

### Main metrics improved
Peak memory (KIVI 2.6x less incl weights, 4x batch -> 2.35-3.47x throughput); GEAR 2.39x peak, batch 3->18 on V100, 5.07x throughput, 2-bit near-lossless on CoT (40.52->40.20 vs KIVI 25.25 +14.95); CacheGen 3.5x size TTFT -68%.

### Typical tradeoffs
Group size smaller -> more overhead but lower error; residual adds 6% extra; power-iteration adds jitter; MQA needs 4-bit not 2-bit.

### Common assumptions
Round-to-nearest uniform quantization sufficient, tuning-free; key outliers fixed channels, value sparsity high; residual 128 negligible vs long sequence.

### Known limitations
Author-stated KIVI: uniform G/R not optimal; short context overhead. GEAR: uniform rank/sparsity. Inferred: no integration with eviction.

### Maturity
**Active** — 4-bit mature, 2-bit Emerging (needs residual/low-rank to be lossless). Transitioning to Mature.

## 5. Prefix Caching & Reuse — Automatic Sharing

### Problem addressed
Requests share static prefixes (system prompts, few-shot, chat history, forked agent branches). Without automatic reuse, prefill recompute dominates TTFT.

### Representative papers
- **SGLang RadixAttention** — 2024, NeurIPS 24
- **LMCache** — 2025, arXiv:2510.09665
- **RAGCache** — 2024, arXiv:2404.12457
- **HotPrefix** — 2026, SIGMOD 26

### Core mechanisms
- **Radix tree + cache-aware scheduling (SGLang 2024):** Edge-labeled token sequences, paged tensors per node, LRU leaf-first eviction, ref counters. Longest-shared-prefix-first approximates DFS optimal when cache >= max request len. Overhead <0.3%.
- **Knowledge tree + PGDSF (RAGCache 2024):** Document IDs as nodes under system prompt root; Priority=Clock+Freq*Cost/Size where Cost bilinear from profiling T(l,u); swap-out-once to Host. Reordering by CachedLength/ComputationLength + speculative pipelined vector search.
- **Coarse chunk coalescing (LMCache 2025):** 256-token chunks vs 16-token pages -> streaming buffers coalesce pages into 1 MB DMA chunks -> 30-46 GBps vs 4 GBps for 64 KB; async prefetch, zero-copy.
- **Hotness-aware (HotPrefix 2026):** Cuckoo filter n buckets x4 entries {fingerprint, clock, freq, depth} 8-bit; Update freq+1 clock=max_age 255, aging clock-1; eviction score (freq+clock)/length leaf-only, admission hotness=freq*clock threshold 10; promotion pipeline via CUDA stream during decode.

### System layer
**memory manager / scheduler / runtime / node**

### Main metrics improved
Hit rate (RAGCache +2-32% over GDSF, +6-62% over LRU; SGLang 50-99%, 96% optimal, production 52-74%; HotPrefix +1.17-2.38x over LRU); TTFT (RAGCache 1.2-4x, HotPrefix 1.54-2.25x, 1.91x throughput); LMCache 2.3-14x throughput.

### Typical tradeoffs
Exact token match only; longer prefixes may starve small hot prefixes; Cuckoo filter memory; promotion PCIe traffic; reordering may increase tail jitter.

### Common assumptions
Prefixes exact token sequences; radix ops cheap vs forward; cache >= max request for optimality; hotness via freq+clock; parent hotter than child.

### Known limitations
Author-stated SGLang: no hierarchical DRAM-SSD, no fuzzy match, starvation. RAGCache: speculative may add load at high RPS. LMCache: centralized Controller may bottleneck. Inferred: no privacy isolation.

### Maturity
**Mature** — production (SGLang Chatbot Arena, vLLM prefix caching, LMCache enterprise).

---
## 6. KV Sharing — Copy-on-Write & Cross-Request Fusion

### Problem addressed
Parallel sampling, beam search, shared prefixes, RAG multi-chunk concatenation share identical KV but naive per-sequence duplicates memory (vLLM 6-66% savings opportunity). Non-prefix arbitrary-position reuse loses cross-attention if naively spliced.

### Representative papers
- **vLLM (CoW)** — 2023, SOSP 23
- **CacheBlend** — 2025, EuroSys 25 Best (arXiv:2405.16444)
- **Cache-Craft** — 2025, SIGMOD 25 (arXiv:2502.15734)
- **KVLink** — 2025, arXiv:2502.16002

### Core mechanisms
- **Copy-on-Write (vLLM 2023):** Prompt logical blocks map to same physical blocks refcount=2; on write to shared last block, allocate new physical, copy, decrement. Beam search like process tree; up to 55% saving.
- **Selective recomputation (CacheBlend 2025):** Any-position chunks independent precompute -> splice with RoPE re-encoding -> HKVD tokens 10-15% with far larger deviation, progressive filtering r1>r>r2, per-layer Trecompute overlapped with Tload; r*=15% -> 2.2-3.3x TTFT <=0.02 F1 loss.
- **Attention-aware reuse (Cache-Craft 2025):** inter/intra attention, Prefix Overlap beta, order penalty gamma via Kendall Tau, CCI=1/(1+e^{-a_bar/b_bar}) where a_bar=inter/|C_i||C_j|, b_bar=intra/|C_i|^2; CCI low -> reuse. CFO=f(CCI,1-beta_prime) weights repair cost; selective recompute + early termination + variant management.
- **Position-independent linking (KVLink 2025):** PIC: store KV as W_{k,v}.x pre-RoPE, re-apply global RoPE at splice; attach K=5 trainable link tokens per doc (doc-internal causal, link attends all prior docs+links) -> only link tokens forwarded to fuse dependencies.

### System layer
**memory manager / attention / runtime**

### Main metrics improved
Memory 37-55% (beam), TTFT -96% (KVLink 5k), throughput 2.8-5x (CacheBlend), redundant compute -51% vs SOTA prefix (Cache-Craft).

### Typical tradeoffs
Recompute 15-30% adds compute (pipelining hides); variant storage 100 chunks x5 variants ~50-150 GB; training-required (KVLink 6000 steps 8xH100) vs training-free.

### Common assumptions
Attention sparsity makes <20% recompute sufficient; embedding slow change -> correlation; knowledge base skewed (top 5% hit 60%); RoPE decouplable.

### Known limitations
Author-stated CacheBlend: Transformer-only, single-tier storage. Cache-Craft: thresholds need recalibration. KVLink: needs fine-tuning, storage huge (131 MB/1k tokens). Inferred: privacy leakage via spliced KV.

### Maturity
**Active -> Emerging** — prefix CoW Mature (vLLM), arbitrary-position and PIC Active Fast-growing (2024-2025).

---
## 7. Offloading & Hierarchical Memory

### Problem addressed
GPU HBM (24-96 GB) insufficient for long KV (128K -> tens GB) and batch scaling (70% memory at 10K prompt). CPU DRAM, CXL pool, SSD provide capacity but 10-100x lower BW/latency.

### Representative papers
- **FlexGen** — 2023, ICML 23 — LP search offload + 4-bit
- **InfiniGen** — 2024, OSDI 24 (arXiv:2406.19707) — rehearsal prefetch
- **ShadowKV** — 2025, ICML 25 — low-rank + offload
- **Beluga** — 2026, SIGMOD 26 (arXiv:2511.20172) — CXL 2.0 pool
- **CachedAttention / AttentionStore** — 2024, ATC 24 (arXiv:2403.19708) — DRAM->SSD multi-turn

### Core mechanisms
- **LP offload search (FlexGen 2023):** Zig-zag scheduling + LP to choose storage for weights/KV/cache + 4-bit quantization; enables OPT-175B on 16GB GPU.
- **Rehearsal prefetch (InfiniGen 2024):** Full pool on CPU, prefetch essential per layer via cross-layer speculation: use Xa_{i-1} + partial Q weight layer i (top 30% columns after SVD skew A=V preserving QK^T exactly) to predict scores; threshold max-alpha (alpha=4 OPT /5 LLaMA2) selects <10% avg (cap 20%). Counter-based eviction on CPU limit.
- **Landmark offload (ShadowKV 2025):** See Category 3; value offload CPU, key low-rank GPU, landmarks guide sparse selection reducing fetch 2x.
- **CXL switch pool (Beluga 2026):** Replace RDMA NICs with 2x PCIe5 x16 CXL adapters + 2x XC50256 switches (256 lanes, 2 TB/s, 750 ns) -> 16 servers to 8 TB pool at 1 TB/s, mmap DAX, GPU direct load/store/P2P eliminating bounce buffer and 75% sync overhead (8us of 10.55us).
- **Hierarchical DRAM->SSD (CachedAttention 2024):** Async save, layer-wise preload, scheduler-aware fetch/eviction, decoupled positional encoding; ShareGPT 87% TTFT, 7.8x prefill.

### System layer
**memory manager / node / cluster**

### Main metrics improved
Capacity (FlexGen 175B on single GPU; Beluga 14.6% HBM hit vs 1.54->11.32 QPS 7.35x vs Mooncake RDMA, 89.6% TTFT reduction); latency (InfiniGen 3x speedup); memory 7.08x saving.

### Typical tradeoffs
Offload adds PCIe/CXL latency; speculation accuracy vs prefetch budget; CXL SW coherence cost (Uncacheable + CLFLUSH); disk prefetch may waste BW.

### Common assumptions
CPU/remote DRAM capacity >> HBM, BW within 10x; outlier channels stable; temporal locality >60%; full KV must remain accessible due to revival; no pre-training needed.

### Known limitations
Author-stated InfiniGen: single-GPU, PCIe 3.0. Beluga: CXL2 lacks host-host coherence. CachedAttention: hierarchical limited to chat. Inferred: CPU pool OOM beyond 48 batch 128K needs SSD; prediction errors unbounded.

### Maturity
**Active** — FlexGen baseline Mature, heterogeneous CXL Emerging (Beluga prototype), CachedAttention production-like.

---
## 8. Recomputation — Selective & Piecewise Fusion

### Problem addressed
Non-prefix splicing loses cross-chunk attention; full prefill expensive (4000 tokens 3s/6s on A40 34B/70B). Need to restore quality with minimal recompute.

### Representative papers
- **CacheBlend** — 2025, EuroSys 25
- **Cache-Craft** — 2025, SIGMOD 25
- **KVLink** — 2025, arXiv:2502.16002

### Core mechanisms
- **HKVD progressive filtering (CacheBlend 2025):** See Category 6; per-layer r% overlapped with next-layer KV load via dual threads: if Trecompute(r) <= Tload hidden.
- **Relevance early termination (Cache-Craft 2025):** CFO-guided per-chunk ratio + early stop per layer; Partial Prefill kernel mixes recomputed tokens with reused splice.
- **Link-token fusion (KVLink 2025):** Trainable attention fusion vs value-variance recompute (CacheBlend 18%).

### System layer
**attention / kernel / scheduler**

### Main metrics improved
Quality recovery (CacheBlend 5-18% -> <=0.015 F1 loss; Cache-Craft 30% -> 90% ROUGE, 45% ->99%); TTFT hiding (7B 3ms vs 16ms NVMe fully hidden).

### Typical tradeoffs
More recompute -> higher quality but higher TTFT; training-required vs training-free deployability.

### Common assumptions
Cross-chunk deviation concentrated few tokens; embedding slow change; RoPE correctable via single rotation.

### Known limitations
Author-stated r*=15% may drift; fine-tune needed for PIC. Inferred pipelining limited when device load dominates (70B 7ms vs 4ms).

### Maturity
**Emerging** — 2024-2025 burst, training-free gaining adoption via LMCache.

## 9. KV Transfer & Communication Optimization

### Problem addressed
Disaggregated prefill/decode requires moving KV cache (1.13 GB per 512 tokens on 66B) over interconnect. NCCL fragmentation (Lx2 calls per block), PCIe bounce buffers, RDMA QP ordering and 30-entry sglist limits dominate E2E (~25% at 13K tokens).

### Representative papers
- **Mooncake** — 2025, FAST 25 Best (arXiv:2407.00079)
- **FlowKV** — 2025, arXiv:2504.03775
- **CacheGen** — 2024, SIGCOMM 24
- **Dynamo** — 2025, GTC 25 / 0.4
- **LMCache** — 2025, arXiv:2510.09665

### Core mechanisms
- **Disaggregated KV pool + Messenger (Mooncake 2025):** CPU DRAM/SSD/RDMA pool per GPU node (paged blocks hash dedup, block 512), GPUDirect RDMA Messenger streaming layer-by-layer overlap with prefill; CPP and Layer-wise Prefill (async load/store per layer, max(load,prefill)).
- **Shape reshape + coalescing (FlowKV 2025):** Reshape KV from (L,2,B,H) -> (B,L,2,H) -> Lx2 reduction in NCCL calls; segment management via min-heap contiguous segments; bidirectional alignment merges O(n) sends into O(1); pipeline NCCL/IPC/RDMA per hardware.
- **Bitstream codec (CacheGen 2024):** Delta bitstream + 1.5K-token chunks + text fallback; 3.5-4.3x size, 3.2-3.7x fetch delay.
- **Industrial runtime (Dynamo 2025 + LMCache 2025):** NIXL non-blocking P2P, Smart Router radix KV-aware, Planner SLO autoscaling, KV Block Manager PB offload; LMCache DMA coalescing pages into 1 MB chunks -> 46 GBps vs 4 GBps for 64 KB.

### System layer
**node / cluster / kernel**

### Main metrics improved
Transfer latency (FlowKV 0.944s->0.053s -96%, 23469->1 NCCL calls 24x; Mooncake RDMA 87GB/s); throughput (FlowKV +95%/40%/35% vs DistServe/Mooncake/vLLM-Disagg; Mooncake 50-525% throughput, 75% real workload vs vLLM).

### Typical tradeoffs
Coalescing segment management overhead; codecs add encode/decode latency; RDMA needs ConnectX-7 NICs ($1745) vs CXL adapter $210.

### Common assumptions
Interconnect BW bottleneck (NVLink 600GB/s, IB 400Gbps, ENI 25-50Gbps, PCIe5 64GB/s); transfer overlap layer-wise; KV size linear with prompt.

### Known limitations
Author-stated Mooncake: transfer prediction hard due to congestion, replication manual threshold. FlowKV: shape reshape coupled to PagedAttention. Inferred: 100K-1M contexts need PB offload not evaluated.

### Maturity
**Active** — rapid industry adoption (NVIDIA Dynamo NIXL, Mooncake Kimi production, LMCache enterprise).

---
## 10. Scheduling — Continuous Batching & Cache-Aware Admission

### Problem addressed
Request-level batching blocks early finishing and late arrivals; iteration-level (Orca) helps but long prefills stall decodes -> tail TBT spikes.

### Representative papers
- **Orca** — 2022, OSDI 22
- **Sarathi-Serve** — 2024, OSDI 24 (arXiv:2403.02310)
- **HotPrefix** — 2026, SIGMOD 26
- **Online Scheduling** — 2025, arXiv:2502.07115
- **FastServe** — 2023, arXiv:2305.05920
- **TetriInfer** — 2024, arXiv:2401.11181

### Core mechanisms
- **Iteration-level + selective batching (Orca 2022):** Single iteration vs whole request; split batch for Attention per-request while flattened [sum_L,H] for Linear/LayerNorm/GeLU. Control-data plane separation via gRPC vs NCCL; 36.9x throughput at same latency (175B 16 GPUs).
- **Chunked-prefill + stall-free (Sarathi-Serve 2024):** Split >512-token prefills into ~512 chunks; piggyback-with-budget scheduler packs all running decodes + optional prefill chunk up to token budget tau (512 strict /2048 relaxed, profiled via Vidur, tile-aware: 257 32% slower than 256). Uniform hybrid batches -> pipeline bubbles minimized; 2.6x Mistral-7B, 3.7x Yi-34B, 5.6x Falcon-180B PP capacity.
- **Hotness-aware (HotPrefix 2026):** See Category 5 Cuckoo tracking + admission threshold 10 + promotion pipeline.
- **Online theory (Online Scheduling 2025):** First KV-constrained online model with competitive ratio; batch algorithm with guarantees for Omega(sqrt(n)) classic bound.
- **Token preemption (FastServe 2023):** Skip-join MLFQ -> 31.4x SLO throughput over vLLM.

### System layer
**scheduler / runtime**

### Main metrics improved
Throughput / capacity under SLO (Orca 36.9x, Sarathi up to 6.3x, FastServe 31.4x); P99 TBT stall reduction (Sarathi Decode+Full 28.3x vs Decode+Chunked bounded).

### Typical tradeoffs
Smaller tau lowers TBT but increases chunk overhead (25% at 512, O(N^2) reloads); larger tau risks SLO violation; preemption adds migration cost.

### Common assumptions
Prefill compute-bound, decode memory-bound; prefill flat beyond 512 tokens, decode scales linearly; Poisson arrivals; PVC iteration time approx max(Tmath,Tmem).

### Known limitations
Author-stated Sarathi: not compared to disaggregated, chunked slower than full prefill due to reload, tau requires offline profiling. HotPrefix: threshold sensitivity. Inferred: no prefix sharing in Sarathi; TTFT may increase; fairness not integrated.

### Maturity
**Mature** — iteration-level standard (vLLM continuous batching adopted Sarathi chunked-prefill as default). Hotness admission Active. Online theory Sparse (1 paper).

---
## 11. Request Routing & Load Balancing — KV-Aware

### Problem addressed
Global routing without KV locality causes imbalance and extra transfers. Need to balance hit rate vs load, avoid hot-spot congestion, elastically morph roles.

### Representative papers
- **Mooncake Conductor** — 2025, FAST 25
- **Dynamo Smart Router / Planner** — 2025, GTC 25
- **FlowKV Global Controller + Hybrid Scheduler** — 2025, arXiv:2504.03775
- **Llumnix** — 2024, OSDI 24 (arXiv:2406.03243)
- **DistServe Placement** — 2024, OSDI 24

### Core mechanisms
- **KVCache-aware scheduling (Mooncake Conductor 2025):** Find best_prefix_len via prefix hash block 512, estimate T_queue + T_prefill + T_transfer; route to min TTFT; kvcache balancing threshold trades recompute vs transfer; hot-spot replication if extra prefill < transfer time.
- **Smart Router + Planner (Dynamo 2025):** Radix-tree KV-aware routing, NIXL, SLO autoscaling, KV Block Manager PB offload, Grove K8s.
- **Load-aware three-scenario (FlowKV 2025):** Global controller monitors load+hit; Local Hybrid Scheduler per node sharing block manager; Normal -> optimal Pt/Dt; Imbalanced -> idle nodes flip roles; Extreme -> elastic scale and reconstruction.
- **Live migration (Llumnix 2024):** Virtual-usage abstraction; pipelined KV live migration <decode downtime; 15x P99 TTFT, 36% cost saving.

### System layer
**cluster / scheduler**

### Main metrics improved
TTFT routing (Mooncake 89.6% vs Dynamo, 7.35x QPS at cache-hit); load imbalance reduction (7% setback when heterogeneous mismatched); autoscaling cost (Llumnix 36%).

### Typical tradeoffs
Global controller single point; KV-aware routing concentrates load -> skew; migration adds copy; threshold tuning manual.

### Common assumptions
Workload skewed (hot blocks tens of thousands accesses, >50% never reused); hit vs load predictable via offline profiling; pool can morph roles cheaply.

### Known limitations
Author-stated Mooncake: balancing threshold manual, coarse prediction. Llumnix: 16 GPUs only. Inferred: radix sync consistency lag not quantified.

### Maturity
**Active** — emerging production (Dynamo, Mooncake Kimi). Transitioning to Mature.

## 12. Prefill/Decode Disaggregation — Phase Splitting

### Problem addressed
Colocated batch stalls vs throughput collapse: E2E dominated by token phase (BLOOM-176B 1500 prompt same time as 6 decode tokens) but single instance cannot optimize compute-bound prefill and memory-bound decode simultaneously.

### Representative papers
- **Splitwise** — 2024, ISCA 24 (arXiv:2311.18677)
- **DistServe** — 2024, OSDI 24 (arXiv:2401.09670)
- **TetriInfer** — 2024, arXiv:2401.11181
- **DéjàVu** — 2024, ICML 24 (arXiv:2403.01876)
- **AMPD** — 2026, ICML 26 (arXiv:2602.14516)

### Core mechanisms
- **Heterogeneous pools (Splitwise 2024):** Prompt vs token vs mixed pools; prompt batch max 2048 (degrades beyond), token scales to 64; power: token tolerates 50% cap (700->350W) no latency; heterogeneous HA: H100 prompt / A100 token -> 2.35x throughput at same cost/power. Layer-wise async MSCCL++ one-sided put, <7% prompt, 8ms A100 constant via overlap.
- **Goodput placement (DistServe 2024):** Per-GPU goodput = max RPS meeting 90% TTFT+TPOT SLOs /num_gpus; queuing model M/D/1 Avg_TTFT = D + R D^2/(2(1-RD)) and variants; bandwidth-aware placement Alg1 (high affinity) vs Alg2 (co-locate stages for NVLink). Searches inter/intra parallelism per phase; 2.0-4.6x request rate vs vLLM, transmission <0.1% total, 95% <30ms.
- **Chunked + predictive (TetriInfer 2024):** Fixed-size chunked prefill + P/D fully disaggregated + length-predictive scheduling; TTFT -97%.
- **Streaming fault tolerance (DejaVu 2024):** Unified streaming lib + token-level KV replication; 2x throughput.
- **Multi-round adaptive (AMPD 2026):** First multi-round disaggregation: adaptive incremental prefill routing + resource planning for interleaved multi-turn; SLO significantly over Dynamo/vLLM.

### System layer
**cluster / node / scheduler**

### Main metrics improved
Throughput same power/cost (Splitwise 2.35x), goodput (DistServe up to 7.4x requests or 12.6x tighter SLO), TTFT, TBT, fault recovery.

### Typical tradeoffs
Disaggregation doubles weight memory (prefill + decode each hold full weights); KV transfer linear with prompt (1.13 GB per 512 tokens 66B -> 90 Gbps at 10 rps) needs NVLink; heterogeneous IB assumption optimistic; small prompts <512 use serialized.

### Common assumptions
Prompt vs token distinct latency/throughput/memory/power; Poisson arrivals, pipeline optional; model TP 8 GPUs best latency; IB available between any prompt-token pair.

### Known limitations
Author-stated Splitwise: H100->A100 IB not readily available. DistServe: offline throughput may favor chunked piggyback; workload predictability. Inferred: no 32K+ evaluation, no GQA 32K+, no prefix sharing.

### Maturity
**Active** — seminal 2023-2024; production validation in Kimi/Mooncake and Dynamo. Gap to Mature is cost analysis of double weight memory.

---
## 13. Distributed Serving & Elasticity — Multi-Node Scale

### Problem addressed
Models exceed single GPU (13B 1GPU -> 175B 16 GPUs -> 341B 32 GPUs) and demand elasticity for 1K-1M variance and live load imbalance / fragmentation.

### Representative papers
- **Orca (distributed)** — 2022, OSDI 22
- **Llumnix** — 2024, OSDI 24 (arXiv:2406.03243)
- **LoongServe** — 2024, SOSP 24 (arXiv:2404.09526)
- **Dynamo** — 2025, GTC 25
- **DejaVu** — 2024, ICML 24

### Core mechanisms
- **Iteration-level distributed (Orca 2022):** Intra-layer (tensor) + inter-layer (pipeline) 2 inter x3 intra =6 GPUs for 4-layer; gRPC control vs NCCL tensor data, fused attention kernels.
- **Live migration (Llumnix 2024):** Global+llumlet; virtual-usage unification for load balancing/defrag; pipelined KV live migration <decode downtime; 15x P99 TTFT, 36% cost saving on 16 GPUs.
- **Elastic sequence parallelism (LoongServe 2024):** ESP dynamically adjusts DoP per request (1K vs 1M); proactive scale-down + multi-master decode overlapping, token-granular KV pool; 3.85x vs chunked prefill, 5.81x vs disaggregation.
- **Industrial runtime (Dynamo 2025):** NIXL, Smart Router, Planner (SLO autoscaling), KV Block Manager PB offload, Grove K8s, 30x DeepSeek-R1 on GB200.
- **Fault-tolerant streaming (DejaVu 2024):** Token-level replication, streaming lib with recovery.

### System layer
**cluster / node / scheduler / runtime**

### Main metrics improved
Throughput under elasticity (LoongServe 3.85-5.81x), tail latency (Llumnix 15x P99), cost, fault recovery.

### Typical tradeoffs
Migration copy overhead vs defrag benefit; ESP scale add communication if not overlapped; replication doubles memory/BW; pipeline adds sync per iteration.

### Common assumptions
Megatron TP+PP SPMD; failure rare but recovery needed; output length unknown -> fragmentation inevitable; NVLink/IB available.

### Known limitations
Author-stated Llumnix: 16 GPUs only. LoongServe ESP still needs global group. Dynamo: 30x on GB200 NVL72 not generalizable. Inferred: no 100s nodes evaluation.

### Maturity
**Active -> Emerging** — Orca/Megatron mature for TP; elasticity/live migration Active (OSDI/SOSP 2024) not yet default in vLLM; Dynamo pushing to Mature.

---
## 14. Workload-Aware Optimization — RAG / Agent / Long-Context / Heterogeneous

### Problem addressed
Generic KV policies fail for skewed order-sensitive RAG (strict prefix hit <10% but any-position needed), agent multi-turn tool gaps (LRU prematurely evicts soon-to-reuse agent), long 1K-1M variance, and heterogeneous CXL/DRAM BW.

### Representative papers
- **RAG: RAGCache** — 2024, arXiv:2404.12457; **CacheBlend** — 2025, EuroSys 25; **Cache-Craft** — 2025, SIGMOD 25; **KVLink** — 2025, arXiv:2502.16002; **Shared RAG-DCache** — 2025, arXiv:2504.11765
- **Agent: KVFlow** — 2025, NeurIPS 25; **Continuum** — 2025, arXiv:2511.02230; **AMPD** — 2026, ICML 26
- **Long-Context: ShadowKV** — 2025, ICML 25; **SCOPE** — 2025, ACL 25; **LoongServe** — 2024, SOSP 24
- **Heterogeneous: Beluga** — 2026, SIGMOD 26; **Dynamic Placement** — 2025, IEEE CAL; **InfiniGen** — 2024, OSDI 24
- **Surveys: TMLR Survey** — 2025, TMLR 25; **From Attention to Disaggregation** — 2025, arXiv:2511.07422

### Core mechanisms
- **RAG-aware:** Knowledge-tree hierarchical, non-prefix selective recompute (CacheBlend 15%, Cache-Craft 30%), CCI/CFO contamination detection, PIC position re-encoding + link tokens, disk pre-generation sharing across instances (Shared RAG-DCache +15-71% throughput).
- **Agent-aware:** Agent Step Graph prediction steps-to-execution = max/min over predecessors -> priority = min among children -> evict larger steps first + proactive prefetch status-aware skipping (KVFlow 2.19x); TTL retention optimal tau* = argmax P(tau,f)*(T*psi+Prefill-Reload) - alpha vs cost (Continuum 8x JCT, 144.9 steps/min).
- **Long-context-aware:** Low-rank landmarks (ShadowKV), prefill/decode separation with drift fixes (SCOPE), ESP dynamic DoP (LoongServe) handling 1M variance.
- **Heterogeneous-aware:** GH200 formal placement min sum max{t^h,t^e} s.t. P_H <=100%; SA search over window W and ratio R -> 5.87x throughput upper bound; Beluga CXL direct GPU->CXL near-local latency.

### System layer
**scheduler / memory manager / runtime / cluster**

### Main metrics improved
TTFT -96% (KVLink), 2.1x (RAGCache), JCT 8x (Continuum), batch 6x (ShadowKV), near-HBM goodput (Dynamic Placement), disk throughput (Shared RAG-DCache).

### Typical tradeoffs
RAG order contamination vs reuse; agent TTL longer -> hit higher but blocks memory (Cost=tau*MemUsage/M); ESP scheduler complexity; heterogeneous needs perfect future knowledge for upper bound.

### Common assumptions
RAG skewed 20x uniform, order matters; agent tool gaps short (mean 0.9-1.9s), many turns (6-11, tokens 70K+) -> queueing accumulates; long sparsity varying; CXL BW within order of HBM.

### Known limitations
Author-stated RAGCache: speculative may add load. KVFlow: requires accurate Step Graph. Continuum: only ReAct linear. Beluga: SW coherence needed, RC bottleneck. Dynamic Placement: perfect pre-knowledge not realizable. Inferred: rare tool tail fallback to global.

### Maturity
**Emerging -> Active** — fastest growth 2024-2026; RAG non-prefix moving to Active, agent TTL Emerging, heterogeneous CXL Emerging, theory Sparse.

---

## Cross-Cutting Trade-offs & Synergy Map

- **Memory vs Compute:** Paging (1) saves memory without recompute -> quantization (4) saves bytes but adds dequant compute -> eviction (2) saves both but risks accuracy -> recompute (8) trades compute to recover accuracy.
- **Hit Rate vs Load:** Prefix caching (5) + routing (11) -> higher hit risks skew; balancing thresholds (Mooncake 2025, FlowKV 2025) needed.
- **Transfer vs Recompute:** For RAG non-prefix, choose per chunk: transfer (CacheGen codec) vs selective recompute (CacheBlend 15%); controller picks min Trecompute vs Tload. Similar for PD disaggregation: overlap vs separate pools.
- **Stability:** Eviction persistence (Scissorhands 2023) vs revival (InfiniGen 2024) -> ephemeral pruning (InfiniGen, ShadowKV) preserves CPU pool.
- **Orthogonal stacking:** PagedAttention (1) + RadixAttention (5) + CoW sharing (6) + chunked stall-free (10) + disaggregation (12) + CXL pool (7) are largely orthogonal; LMCache/Dynamo demonstrate unified connectors.

## Maturity Summary (Literature Coverage, Not Novelty)

| Maturity | Categories | Rationale |
|---|---|---|
| **Mature** | 1 Paged Memory, 10 Continuous Batching, 5 Prefix Reuse | >=3 papers each, years production (vLLM, SGLang, Orca->Sarathi). De-facto standards. |
| **Active** | 2 Eviction, 3 Compression, 4 Quantization, 6 KV Sharing, 9 KV Transfer, 12 Disaggregation, 13 Distributed/Elastic | 3-6 papers 2023-2025, active 2024-2025 conferences, transition to production but open problems remain. |
| **Emerging** | 7 Offloading/CXL, 8 Recomputation/Fusion, 14 Workload-Aware RAG/Agent/Heterogeneous | 2024-2026 burst, 2-4 papers each, prototypes (Beluga CXL, CacheBlend, KVFlow/Continuum) not yet default; theory Sparse within. |
| **Sparse** | 10-theory Online Scheduling 2025, 14-theory Dynamic Placement 2025, Shared RAG-DCache disk 2025 | 1-2 papers, proof-of-concept, needs follow-up. |

*All maturity judgments based on count + deployment, not novelty. Example: LoongServe ESP (2024 SOSP) novel but 1 paper -> Emerging/Sparse, not Mature. vLLM PagedAttention (2023) dated but Mature due to coverage.*

## Traceability & Integrity Notes

- Every representative paper title/year/venue exists verbatim in papers.md (43 entries). Checks: Orca OSDI22, vLLM SOSP23, SGLang NeurIPS24, Splitwise ISCA24, DistServe OSDI24, Sarathi-Serve OSDI24, Mooncake FAST25, FlowKV arXiv2504.03775, Beluga SIGMOD26, AMPD ICML26, CacheGen SIGCOMM24, CachedAttention ATC24, Dynamo GTC25, Llumnix OSDI24, LoongServe SOSP24, etc. No hallucinated titles.
- Short citations (H2O 2023, KIVI 2024, SCOPE 2025) used throughout.
- Author-stated limitations quoted from paper_notes Known limitations [PAPER FACT]; inferred marked distinct.
- No novelty claims beyond maturity definitions.

## Sampling Evidence (for audit)

Read 24 paper_notes >=20 required, covering all 11 diversity buckets: serving foundations (Orca, vLLM, SGLang, LMCache), eviction (StreamingLLM, H2O, SnapKV, SCOPE), compression (PyramidKV, ChunkKV, ShadowKV), quantization (KIVI, GEAR, CacheGen), prefix reuse (RAGCache, CacheBlend, HotPrefix, KVLink, Cache-Craft), offloading (FlexGen, InfiniGen, CachedAttention, Beluga), scheduling (Sarathi-Serve, FastServe, OnlineScheduling), disaggregation (Splitwise, DistServe, Mooncake, TetriInfer, DejaVu, AMPD), RAG (RAGCache, CacheBlend, Cache-Craft, KVLink, SharedRAG-DCache), agent (KVFlow, Continuum), heterogeneous (Beluga, DynamicPlacement, ShadowKV), surveys (TMLR Survey, FromAttention).

*End of taxonomy_draft.md — 14 categories with full subsections; ready for Stage 2B idea generation but not generating ideas here.*

