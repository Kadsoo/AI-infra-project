# Bottleneck Map — LLM Inference KV Cache Optimization (Stage 2A Step B)

> **Workdir:** `F:\AIinfraResearch` | **Input:** `research/paper_notes/*.md` (43 notes) + `research/manifests/papers.md` (43 papers) + `research/knowledge/taxonomy_draft.md` | **Date:** 2026-08-27
> **Method:** Read 24+ diverse notes + papers.md; categorized by BOTTLENECK (not by paper); traceability via short citations; no new solutions proposed.
> **Coverage audit:** All 12 mandatory bottlenecks covered + 2 additional (Hierarchical Tiering Complexity, Compression-vs-Quality Degradation). Total 14 bottlenecks.

## Methodology & Traceability

- **Notes read (24+):** vLLM, SGLang, Mooncake, Sarathi-Serve, LMCache, Orca, FlexGen, Splitwise, DistServe, RAGCache, CacheBlend, InfiniGen, KVSurvey (TMLR Survey), H2O, StreamingLLM, KIVI, Beluga, FlowKV, SnapKV, ShadowKV, PyramidKV, Cache-Craft, KVLink, KVFlow, plus manifests/papers.md and taxonomy_draft.md sampling 24 categories.
- **Citation style:** Short name + year, e.g., `vLLM (2023)`, corresponds to entries in `research/manifests/papers.md` and `research/paper_notes/*.md`. Year = arXiv first / venue year per manifest.
- **Limitation attribution:** `Author-stated:` quoted from paper_notes sections 12; `Inferred (repeated):` marked when inferred limitation recurs across >=2 papers.

---

## Bottleneck 1 — GPU HBM Capacity Exhaustion (KV Cache Dominates Memory)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **GPU HBM** (device DRAM, e.g., A100 40/80 GB, H20 96 GB). Secondary: **Scheduler / Cluster** (admission control when HBM full).

### Trigger condition (when does it appear, workload/hardware condition)
- Large batch x long prompt: OPT-175B batch 512 x 2048 ctx -> 1.2-3 TB KV (3-3.8x weights) [FlexGen 2023, KIVI 2024]. PaLM 540B batch 512 x2048 -> 3 TB [KIVI 2024].
- Long-context serving: LLaMA-2-7B 100K tokens -> >50 GB KV vs <1 GB at 2K [PyramidKV 2024]; Llama-3-8B 1M context OOM at batch 2 even on 80 GB [ShadowKV 2025].
- Decoding phase growth: each token adds ~800 KB (OPT-13B) or 0.1-0.3 MB/token (LLaMA-3) per layer x heads, linear with length [vLLM 2023, Cache-Craft 2025].
- Single-GPU or tensor-parallel shard with limited HBM vs CPU DRAM 256 GB+ available but 10-100x slower.

### Why it hurts (mechanism)
KV cache per token per layer per head must be resident for attention `Q·K^T` and `A·V`. When sum of weight (26-346 GB for 13B-175B) + KV exceeds HBM, system cannot admit new requests: either OOM, or forced eviction/swap, drastically capping batch size and throughput. Compute is underutilized while capacity-bound. vLLM profiling: 65% weight + 30% KV + remainder activations on A100-40GB, only tens of requests fit [vLLM 2023].

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Effective batch size** / max concurrent requests per iteration (Fig.13 vLLM 2023).
- **Peak memory usage** GB, KV memory % vs weight, effective KV usage % (20.4-38.2% baselines vs >96% vLLM [vLLM 2023]).
- **Throughput req/s or tokens/s at same latency** (2-4x gain when capacity freed [vLLM 2023]).
- **OOM threshold** (tokens until OOM: LWM baseline 33K vs SnapKV 380K [SnapKV 2024]).
- **Memory saving factor** (KIVI 2.6x less incl. weights; ShadowKV 7.08x per formula [KIVI 2024, ShadowKV 2025]).

### Papers addressing it (2-6, with year)
- vLLM (2023) — PagedAttention 2-4x throughput
- FlexGen (2023) — LP offload search single-GPU 175B
- H2O (2023) — 5x reduction via heavy-hitter retention
- KIVI (2024) — 2-bit asymmetric quantization
- ShadowKV (2025) — low-rank keys + value offload 6x batch
- LMCache (2025) — standardized KV layer + CPU offload

### Existing solution families (summarize approaches)
1. **Paged / Non-contiguous allocation:** vLLM PagedAttention partitions into 16-token blocks bounding waste to <=1 block; SGLang/LMCache similar.
2. **Eviction / Sparse retention:** H2O heavy-hitter + recent (dynamic submodular), StreamingLLM sinks+window, SnapKV observation-window voting — keep 5-20% KV.
3. **Quantization / Compression:** KIVI per-channel key + per-token value 2-bit (group 32, residual 128); GEAR quant+low-rank+sparse; CacheGen delta codec 3.5-4.3x.
4. **Offloading / Hierarchical:** FlexGen LP placement GPU/CPU/disk + 4-bit; InfiniGen rehearsal prefetch <10% fetch; ShadowKV low-rank pre-RoPE (rank160) + CPU value; Beluga CXL pool 8 TB.
5. **Standardized layer:** LMCache chunk (256) + streaming coalesce 1 MB DMA 30-46 GBps.

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (vLLM 2023):** Paging not beneficial for training, adds 20-26% attention kernel indirection overhead; swap space bounded by GPU KV -> head-of-line blocking.
- **Author-stated (KIVI 2024):** Uniform G=32/R=128 not optimal; short contexts overhead of residual; MQA (Falcon) needs 4-bit not 2-bit.
- **Author-stated (ShadowKV 2025):** SVD cost for short prompts; uniform rank 160 not adaptive; depends on PCIe 31.5 GB/s and CPU DRAM capacity.
- **Author-stated (InfiniGen 2024):** Single-GPU PCIe 3.0 only; extra GPU memory for partial weights/keys (2.5%/15%).
- **Inferred (repeated across FlexGen, InfiniGen, LMCache):** CPU pool OOM beyond ~48 batch x 128K still requires SSD tier not evaluated; prediction errors unbounded; no joint token x bit optimization.

---
## Bottleneck 2 — Memory Fragmentation & Reservation Waste

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **GPU HBM allocator / Memory manager**. Secondary: **PCIe** (small fragmented transfers) and **Scheduler** (blocking on allocation).

### Trigger condition (when does it appear, workload/hardware condition)
- Variable-length requests with unknown output length: Orca reserves `max_tokens` (2048) per request [Orca 2022]; contiguous pre-allocation for max length leaves 60-80% waste (20.4% Max, 38.2% Pow2 effective [vLLM 2023]).
- Paged blocks too large (>32) -> internal fragmentation; too small (e.g., 16 KB pages 62.5 KB per 16 tokens) -> external fragmentation + many small PCIe/NCCL calls.
- Buddy allocator with differing pre-allocated sizes leaves unusable holes [vLLM 2023].
- Small-page LMCache (16 tokens) coalesced poorly -> 4 GBps at 64 KB vs 30 GBps at 1 MB [LMCache 2025]; FlowKV 23,469 NCCL calls per request due to discrete blocks [FlowKV 2025].

### Why it hurts (mechanism)
Contiguous PyTorch requirement forces pre-reserve max; actual length << max, unused slots unavailable to others (internal) while holes from differing sizes unusable (external). Future-token reservation occupies memory entire lifetime before generation. Paged indirection fixes fragmentation but adds block-table lookup branches and non-coalesced concerns. PCIe/NCCL needs contiguous 1-2 MB to achieve 75-80% PCIe 5.0 BW; fragmented 64KB chunks achieve only 4 GBps vs 46 GBps with coalescing [LMCache 2025].

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Effective KV usage % / waste %** (Fig.2 vLLM 2023: 20.4-38.2% vs ~96% vLLM).
- **Internal waste bound** <=1 block per sequence (vLLM 16).
- **PCIe/RDMA throughput** per transfer size (Table1 LMCache 2025).
- **Kernel latency overhead** 20-26% vs contiguous FT kernel [vLLM 2023].
- **NCCL calls per request** (FlowKV 23469->1 after coalescing).
- **Batch size** improvement 2-4x when defragmented.

### Papers addressing it (2-6, with year)
- vLLM (2023) — PagedAttention 16-token blocks
- SGLang (2024) — RadixAttention paged trie
- LMCache (2025) — 256-token chunk + DMA coalesce
- FlowKV (2025) — segment management + shape reshape
- Beluga (2025) — CXL DAX mmap vs RDMA bounce buffer
- Orca (2022) — baseline showing reservation waste

### Existing solution families (summarize approaches)
1. **Paged allocation:** Fixed-size KV blocks (B=16) mapping logical->physical via block table + ref-count CoW [vLLM].
2. **Trie / Chunk pooling:** Radix tree paged tensors per node + LRU leaf-first [SGLang]; LMCache 256-token chunks + streaming buffers coalescing 16-token pages into 1 MB DMA.
3. **Segment compaction:** OS-segment idea min-heap contiguous segments, bidirectional alignment merging O(n)->O(1) sends [FlowKV].
4. **Zero-copy direct access:** GPU direct CXL load/store via adapters + DAX mmap eliminating bounce buffer and 75% sync overhead [Beluga].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (vLLM 2023):** Requires LLM-specific adaptations (all-or-nothing eviction, fused kernels); not for training; 20-26% kernel penalty.
- **Author-stated (SGLang 2024):** Need tuning block size 16-128 per workload (ShareGPT likes 64 vs Alpaca 16); exact prefix only; production hit 52-74% < benchmark 50-99%.
- **Author-stated (LMCache 2025):** Centralized Controller may bottleneck batched sync at 1000+ instances; 500 GB CPU DRAM still insufficient for many models—needs hierarchical DRAM-SSD-S3 auto-tiering not formalized.
- **Inferred (repeated vLLM, FlowKV, Beluga):** No auto-tuning of block size; fragmentation vs kernel overhead tradeoff still manual; security/isolation of CoW sharing across tenants not addressed.

---

## Bottleneck 3 — KV Cache Miss / Low Hit Rate (Prefix Sharing Failure)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Memory manager (cache) + Scheduler**. Secondary: **GPU HBM** (when miss forces recompute) and **Network** (if remote fetch).

### Trigger condition (when does it appear, workload/hardware condition)
- Non-prefix reuse: RAG multi-chunk arbitrary position -> strict prefix hit <10% [Cache-Craft 2025]; production exact prefix caching 8% requests / 18% tokens [Cache-Craft]; Mooncake max reuse plateau 30->50% even at infinite storage [Mooncake 2025].
- Order sensitivity: [D1,D2] vs [D2,D1] KV not interchangeable due to positional/causal masking [RAGCache 2024].
- Small cache vs working set: 1K vs 50K blocks 30%->50% hit, hot skew (50% blocks never reused, hot blocks 10k+ accesses) [Mooncake 2025]; CPU offload eviction when HBM 14.6% peak hit [Beluga 2025].
- Multi-turn/agent polluted history: >50% of 5-chunk requests cross 3+ historical requests [Cache-Craft].

### Why it hurts (mechanism)
Reuse requires exact token prefix match plus same positional embeddings. Non-prefix chunks lose cross-attention if naively spliced; order change invalidates RoPE. Miss forces fall back to full prefill (3-6 s for 4000 tokens on A40 70B [CacheBlend 2025]) inflating TTFT and GPU-hours. LRU thrashes when small hot prefix starves large cold documents.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Hit rate** = CachedLength / ComputationLength (SGLang) or hit docs / total docs (RAGCache).
- **Gain:** SGLang 50-99% (96% optimal), production 52-74% [SGLang 2024]; RAGCache PGDSF +2-32% over GDSF, +6-62% over LRU [RAGCache 2024]; HotPrefix +1.17-2.38x over LRU [HotPrefix 2026].
- **TTFT reduction** 1.2-4x (RAGCache), 1.54-2.25x (HotPrefix), 1.9x throughput (RAGCache).
- **Cache hit ratio vs capacity** curve plateau at 50-51% [Mooncake 2025].

### Papers addressing it (2-6, with year)
- SGLang (2024) — RadixAttention LRU leaf-first
- RAGCache (2024) — Knowledge-tree + PGDSF replacement
- HotPrefix (2026) — Cuckoo-filter hotness-aware admission
- CacheBlend (2025) — selective recompute for any-position reuse
- KVLink (2025) — position re-encoding + link tokens
- Mooncake (2025) — distributed KV pool + prefix-hash dedup

### Existing solution families (summarize approaches)
1. **Trie/Tree exact match:** RadixAttention edges token sequences, LRU leaf-first, ref-count [SGLang]; Knowledge-tree doc IDs under system prompt + PGDSF Priority=Clock+Freq·Cost/Size [RAGCache].
2. **Hotness-aware:** Cuckoo filter {fingerprint,clock,freq,depth} tracking, eviction score (freq+clock)/len, admission threshold 10 [HotPrefix].
3. **Any-position fusion:** HKVD 15% recompute progressive filtering [CacheBlend]; CCI/CFO contamination detection + partial prefill [Cache-Craft]; pre-RoPE storage + re-apply global RoPE + 5 trainable link tokens [KVLink].
4. **Distributed dedup:** Prefix-hash block 512 deduplication + hash-concatenation [Mooncake].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (SGLang 2024):** Exact prefix only, no fuzzy/semantic match; greedy longest-prefix may starve small hot prefixes; single-tier GPU DRAM only.
- **Author-stated (RAGCache 2024):** Speculative pipelining may add load at high RPS; host memory assumption 1-2x GPU.
- **Author-stated (HotPrefix 2026):** Threshold sensitivity; no distributed CXL integration.
- **Author-stated (KVLink 2025):** Needs 6000-step fine-tune (8xH100), 131 MB/1K tokens storage huge; training-required vs training-free tradeoff.
- **Inferred (repeated CacheBlend, Cache-Craft, KVLink):** No privacy isolation; variant storage explosion (100 chunks x5 variants approx 50-150 GB) not solved; 50% reuse ceiling even with infinite cache suggests workload-inherent limit.

---
## Bottleneck 4 — Redundant Recomputation (Full Prefill on Repeated Context)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **GPU compute (prefill kernels) / Attention**. Secondary: **Scheduler** (when recomputation blocks admission).

### Trigger condition (when does it appear, workload/hardware condition)
- Same chunks/documents appear across 60%+ RAG requests (top 5% docs ->60% hits [RAGCache, Cache-Craft]) but exact prefix cache covers only 8% requests; each query would recompute 60-98% prefill tokens [Cache-Craft 2025] that are otherwise reusable.
- Production Sys-X: 75% retrieved chunks reprocessed in one month =12B tokens approx 9600 GPU-hours ~$50K [Cache-Craft].
- Multi-agent workflows: shared system prompts repeated per agent invocation [KVFlow 2025].
- No intermediate cache -> 4000-token prefill 3 s (34B)/6 s (70B) per request [CacheBlend 2025].

### Why it hurts (mechanism)
Prefill is quadratic O(n^2) attention; recomputing identical KV wastes FLOPs and extends TTFT, reduces MFU, and forces over-provisioning. Cross-attention between chunks discarded when independently precomputed -> quality drop up to 35% if naively reused without repair [KVLink 2025] or 50% F1 drop when mixing 5 blocks from different histories [Cache-Craft].

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Prefill latency** ms vs prompt length (3 s/6 s for 4000 tokens).
- **TTFT reduction** 2.2-3.3x (CacheBlend), 96% TTFT cut (KVLink 5K), 2-2.25x (HotPrefix).
- **Throughput** 2.8-5x vs full recompute (CacheBlend), 2.19x over SGLang (KVFlow).
- **Redundant tokens %** / recomputation ratio r* 15% typical; redundant compute -51% vs SOTA prefix (Cache-Craft).
- **Quality delta** (F1/ROUGE drop <=0.02 with repair vs 0.1-0.35 without).

### Papers addressing it (2-6, with year)
- CacheBlend (2025) — HKVD 15% selective recompute
- Cache-Craft (2025) — CCI/CFO + early termination
- KVLink (2025) — link-token fusion (trainable)
- RAGCache (2024) — hierarchical reuse avoiding recompute
- SGLang (2024) — Radix reuse avoiding recompute for prefix

### Existing solution families (summarize approaches)
1. **Selective recomputation:** HKVD tokens with larger deviation progressive filtering r1->r2, per-layer Trecompute overlapped with Tload [CacheBlend].
2. **Attention-aware reuse scoring:** inter/intra + beta/gamma + CCI sigmoid 1/(1+e^{-a_bar/b_bar}) + CFO to decide per-chunk ratio + early stop [Cache-Craft].
3. **Link-token fusion:** Pre-RoPE storage, global RoPE re-encoding, K=5 trainable tokens per doc attending all prior docs/links, only link tokens forwarded [KVLink].
4. **Pipeline hiding:** Dual threads overlap recompute layer N with KV load layer N+1; if Trecompute <= Tload hidden [CacheBlend].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (CacheBlend 2025):** Transformer-only, single-tier storage; r*=15% may drift with workload; pipelining limited when device load dominates (70B 7 ms vs 4 ms NVMe).
- **Author-stated (Cache-Craft 2025):** Thresholds CCI/CFO need recalibration per distribution drift; limited to LLaMA-3 evaluation.
- **Author-stated (KVLink 2025):** Requires fine-tuning; storage huge; not evaluated at scale heterogeneous storage.
- **Inferred (repeated):** No integration with quantization/eviction; privacy leakage via spliced KV; training-required vs training-free deployability tradeoff unresolved.

---

## Bottleneck 5 — PCIe Bandwidth / CPU-GPU Transfer Saturation

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **PCIe** (CPU<->GPU, Gen1 2 GB/s on A10G, Gen5 64 GB/s on H100, Gen3 31.5 GB/s per A6000). Secondary: **GPU HBM** stalls waiting for load.

### Trigger condition (when does it appear, workload/hardware condition)
- Full KV offload to CPU DRAM (InfiniGen, LMCache): fetching selected KV per layer per token over PCIe dominates. Even with prefetch, hundreds of GB over PCIe stalls Transformer block [InfiniGen 2024, Fig.3].
- Small paged pages 62.5 KB (16 tokens) -> 4 GBps vs 1 MB coalesced 30-46 GBps [LMCache 2025].
- Long-context decode: each step accesses KV; HBM 2 TB/s needed, offloaded shifts to PCIe 31.5 GB/s = 63x slower [ShadowKV 2025].
- Concurrent workflows arrive >50 QPS -> PCIe queuing [KVFlow 2025].

### Why it hurts (mechanism)
PCIe is 10-100x lower BW than HBM and serialized small transfers add launch + sync overhead (H20 16 KB transfer total 10.55 us, data 2.68 us, 75% sync overhead [Beluga 2025]). Fragmented blocks cause many kernel launches competing with GEMM for SMs. Reactive load on miss blocks generation.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **PCIe throughput** per transfer size (LMCache Table1).
- **Transfer latency** ms per layer (InfiniGen Fig.18 96.9% transfer vs Ideal 1.52x slower).
- **Throughput tokens/s** vs batch: InfiniGen 27->42 tokens/s (batch4->20) vs INT4 12->14 [InfiniGen 2024]; LMCache 400 GBps vs native 88 GBps CPU offload [LMCache 2025].
- **Speedup** 3.00x over prior KV management (InfiniGen), 6x batch with ShadowKV.
- **Prefill vs decode PCIe overlap** %.

### Papers addressing it (2-6, with year)
- InfiniGen (2024) — speculative prefetch <10% fetch
- FlexGen (2023) — zig-zag block + overlap 6-way I/O
- ShadowKV (2025) — landmarks reduce fetch 2x + CUDA multi-stream overlap
- Beluga (2025) — CXL vs PCIe bounce-buffer analysis
- LMCache (2025) — chunk coalesce 1 MB DMA
- KVFlow (2025) — proactive prefetch full-duplex overlap

### Existing solution families (summarize approaches)
1. **Sparse prefetch via speculation:** Rehearsal using Xa_{i-1}+partial Q (30% columns via SVD skew V) predicting scores, threshold max-alpha (4/5) -> <10% avg fetch, cap 20% [InfiniGen].
2. **Coalescing & overlap:** Zig-zag block schedule + 6-way overlap (weights/KV/activations) [FlexGen]; 1 MB coalesced DMA via streaming buffers [LMCache].
3. **Landmark guidance:** Chunk mean (C=8) landmarks +48 outliers; TopK 256 (1.56%) selection, reconstruct K_sparse=Gather(A,I)·B then RoPE, overlap via CUDA multi-stream [ShadowKV].
4. **CXL replacement / proactive prefetch:** CXL adapter 750 ns 64B vs PCIe bounce; proactive load of next agent during current agent compute using full-duplex [Beluga, KVFlow].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (InfiniGen 2024):** Single-GPU PCIe 3.0; partial cache overhead 15% KV; prediction errors unbounded; not for pipeline parallel.
- **Author-stated (Beluga 2025):** CXL2 lacks host-host coherence (needs ntstore+CLFLUSH/UC+CLFLUSH, 281 ms if UC); Root Complex bottleneck (33 vs 46 GB/s); needs direct GPU-to-CXL fabric (future).
- **Author-stated (LMCache 2025):** Low BW remote S3 1 GBps only wins for >256K contexts at 32 Gbps otherwise prefill still wins; Python->Rust rewrite pending.
- **Inferred (repeated):** CPU DRAM capacity still finite; PCIe full-duplex hidden but still limited at high concurrency; not evaluated under PCIe 5.0 vs NVLink heterogeneity.

---

## Bottleneck 6 — Network Transfer Overhead (Disaggregated Prefill/Decode & Distributed KV)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Network (RDMA 400 Gbps HDR IB, RoCE, NCCL, IPC)** between nodes. Secondary: **Scheduler** (placement to avoid network) and **GPU HBM** (holding KV for transfer).

### Trigger condition (when does it appear, workload/hardware condition)
- Disaggregated P/D: KV size 1.13 GB per 512 tokens on 66B [DistServe 2024]; 10 rps ->90 Gbps needed; 13K tokens NCCL transfer occupies ~25% E2E [FlowKV 2025].
- Block-wise PagedAttention (L=32 layers) -> Lx2 NCCL calls per block (23,469 calls/request) [FlowKV].
- Inter-node 25-50 Gbps vs intra-node NVLink 600 GB/s; cross-node 25 Gbps limited testbed [DistServe].
- Multi-node prefill pipeline needs 2 expensive RDMA all-reduces per layer reducing MFU [Mooncake 2025].

### Why it hurts (mechanism)
Requires moving KV over interconnect. NCCL only supports contiguous addresses -> fragmented blocks -> many small kernels contending with GEMM for SMs, blocking compute. RDMA requires extra copies via host bounce buffers, QP ordering, sglist limit 30 entries vs 128 non-contiguous chunks per Qwen-32B block -> splitting into multiple requests [Beluga 2025]. 75% sync overhead (8 us of 10.55 us) [Beluga].

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Transfer latency** 0.944 s ->0.053 s after FlowKV (-96%), 2.02 s ->0.044 s at 8K (Mooncake 87 GB/s RDMA).
- **NCCL/SGList calls per request** 23469->1 (24x reduction).
- **% of E2E** 25% at 13K (FlowKV Fig.1), <0.1% after co-location constraint [DistServe Fig.10].
- **Throughput** +95%/40%/35% vs DistServe/Mooncake/vLLM-Disagg [FlowKV]; Mooncake 50-525% throughput, 75% real workload vs vLLM.
- **TPOT/P99** spikes when transfer overlaps.

### Papers addressing it (2-6, with year)
- FlowKV (2025) — shape reshape + coalescing
- Mooncake (2025) — KV pool + GPUDirect RDMA Messenger + CPP
- DistServe (2024) — bandwidth-aware placement Alg1/2
- Splitwise (2024) — layer-wise async MSCCL++ one-sided put
- CacheGen (2024) — delta bitstream codec 3.5-4.3x
- Dynamo (2025) — NIXL non-blocking P2P (industrial)
- Beluga (2025) — CXL as alternative to RDMA

### Existing solution families (summarize approaches)
1. **Coalescing & shape optimization:** Reshape (L,2,B,H)->(B,L,2,H) reduces Lx2 calls; segment min-heap + bidirectional alignment merging O(n)->O(1) [FlowKV].
2. **Overlap & chunked pipeline:** Layer-by-layer async load/store overlapping with compute `max(load,prefill)` [Mooncake]; layer-wise pipelined transfer per layer hiding constant 8 ms A100/5 ms H100 <7% prompt [Splitwise].
3. **Placement-aware:** Enumerate inter/intra parallelism per phase maximizing per-GPU goodput via M/D/1 queue model; constrain same-index P/D stages co-locate on same node for NVLink [DistServe].
4. **Codec & runtime:** Delta bitstream +1.5K chunks + text fallback 3.5x [CacheGen]; NIXL + DMA coalescing 1 MB [Dynamo/LMCache]; CXL P2P direct load/store [Beluga].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Mooncake 2025):** Transfer prediction hard due to congestion, replication threshold manual; transfer depends on network status not just size.
- **Author-stated (FlowKV 2025):** Shape reshape coupled to PagedAttention; not generalized to other allocators.
- **Author-stated (DistServe 2024):** Requires history to fit workload distribution predictable over hours/days; offline profiling via simulator C1-5 constants.
- **Author-stated (Beluga 2025):** CXL switch becomes single point of failure; cost economics not stable ($5,800 sample).
- **Inferred (repeated):** 100K-1M contexts need PB offload not evaluated; double weight memory overhead for P/D pools not quantified.

---
## Bottleneck 7 — Prefill/Decode Interference (Colocated Phase Contention)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **GPU compute / Scheduler (iteration-level batching)**. Secondary: **Cluster** (when coupled resources limit scaling).

### Trigger condition (when does it appear, workload/hardware condition)
- Colocated batching mixes compute-bound prefill (quadratic, saturates at 512 tokens, 82% GPU util) with memory-bound decode (13% util, linear scaling) [FlexGen, Sarathi-Serve 2024].
- Long prompts (median 1730 ShareGPT, 7059 ArXiv) vs Poisson arrival late joiners: prefill-prioritizing (Orca/vLLM hybrid) eagerly schedules prefills -> stalls decodes seconds: Yi-34B 128 requests stalls several seconds (Fig1a Sarathi-Serve); TBT tail spikes.
- Token distribution most time <=20 active tokens (60-70% time <=20 for conversation [Splitwise]) but batch must include long prefill.
- Variable prompt length 32-512 vs gen 1-128 random -> batch mismatch.

### Why it hurts (mechanism)
Single iteration length = max(prefill chunk + decode tokens). Long prefill adds seconds to iteration latency, causing generation stalls, large TBT/P99, pipeline bubbles PB1-3 (varying prefill tokens, P vs D mismatch, varying KV length) -> ~950 ms bubble for Falcon-180B 4K prompt vs 200 ms decode-only batch32 [Sarathi-Serve]. Separated phases could optimize independently but colocated forces same parallelism and batch config.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **TBT (Time Between Tokens) P50/P99** and **TTFT** tail (vLLM stalls seconds).
- **Decode+Full Prefill vs Decode+Chunked** 28.3x increase TBT [Sarathi-Serve Fig9].
- **Pipeline bubble time** ms.
- **Capacity QPS meeting SLO** (Sarathi 2.6x Mistral-7B, 3.7x Yi-34B, 4.3-6.3x LLaMA2-70B vs vLLM/Orca).
- **Prefill throughput flat** beyond batch 1 at 512 tokens while decode scales to 64 [Splitwise Fig6].

### Papers addressing it (2-6, with year)
- Orca (2022) — iteration-level + selective batching baseline
- Sarathi-Serve (2024) — chunked-prefill + stall-free budgeting
- Splitwise (2024) — physical P/D pool splitting
- DistServe (2024) — goodput-optimized disaggregation
- TetriInfer (2024) — fixed chunk + predictive scheduling
- Mooncake (2025) — KV-centric disaggregation + CPP

### Existing solution families (summarize approaches)
1. **Iteration-level & selective batching:** Single iteration per schedule, split before Attention, flattened [sum_L,H] for linear ops [Orca].
2. **Chunked stall-free:** Split >512-token prefills into 512 chunks, piggyback-with-budget tau=512/2048 via Vidur profiling, uniform hybrid batches -> balanced micro-batches, PP bubbles minimized [Sarathi-Serve]. Overhead ~25% at 512 negligible at 2048.
3. **Physical disaggregation:** Prompt vs token vs mixed elastic pools [Splitwise]; per-GPU goodput search Alg1/2 [DistServe]; Conductor KV-aware + chunked pipeline parallelism (CPP) grouping X nodes [Mooncake].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Sarathi-Serve 2024):** Chunked slower than full prefill due to O(N^2) KV reloads (N chunks -> N-1 reloads); tau requires offline Vidur profiling; not compared quantitatively to disaggregated; no prefix sharing.
- **Author-stated (Splitwise 2024):** Heterogeneous H100->A100 IB not readily available; no KV compression; CLS scalability bottleneck for large clusters.
- **Author-stated (DistServe 2024):** Throughput-optimized offline scenarios may favor chunked piggyback; single/re few GPUs limited search space; no 1M evaluation.
- **Inferred (repeated):** Double weight memory for P/D pools; static partitions cause temporary imbalance Fig11 Mooncake (2P+2D worse TTFT than 3P+1D); TTFT may increase vs hybrid-only (0.76s vs 0.53s openchat [Sarathi Table4]).

---

## Bottleneck 8 — Scheduler Inefficiency (Queueing, Head-of-Line Blocking, FCFS)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Scheduler (centralized queue, admission)**. Secondary: **Cluster autoscaler**.

### Trigger condition (when does it appear, workload/hardware condition)
- Poisson or bursty arrivals with variable prompt lengths -> FCFS or request-level batching waits entire batch to finish [Orca 2022 Fig.2]. Orca Max reserves 2048 -> only 0.49 req/s vs Orca+hitch 2+ req/s [Orca].
- Iteration-level still FCFS by arrival: `max_bs` tuned manually, no SLO awareness [Orca].
- HotPrefix threshold sensitivity; OnlineScheduling competitive ratio Omega(sqrt(n)) classic bound shows theoretical hardness [OnlineScheduling 2025].
- High load QPS sweep vs relaxed SLO: vLLM capacity capped identical for max batch 32/64/128 [Sarathi Fig12] -> cannot leverage larger batch.

### Why it hurts (mechanism)
Late arrivals blocked by early long requests; early-finishing inactive requests waste compute within batch. Scheduler cannot preempt long prefills, leading to queuing delay up to whole batch time (seconds). No preemption or priority -> missed SLOs, poor goodput. Mis-predicted queue time + prefill time + transfer time leads to suboptimal routing.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Capacity QPS under SLO** (Sarathi Fig10-11) and **P99 TBT** vs QPS.
- **Median scheduling delay** <2 s limit (Sarathi).
- **Queue time** T_queue sum queued prefills (Mooncake Alg1).
- **Goodput** = RPS meeting 90% TTFT+TPOT /num_gpus (DistServe).
- **Competitive ratio** vs optimal (OnlineScheduling).

### Papers addressing it (2-6, with year)
- Orca (2022) — iteration-level + selective batching FCFS
- Sarathi-Serve (2024) — stall-free token-budget scheduler
- FastServe (2023) — skip-join MLFQ token-level preemption
- Llumnix (2024) — live migration + virtual usage balancing
- OnlineScheduling (2025) — KV-constrained online theory
- HotPrefix (2026) — hotness-aware admission + Cuckoo promotion

### Existing solution families (summarize approaches)
1. **Continuous / Iteration-level:** Per-iteration admission/decisions reducing queue to <=1 iteration [Orca].
2. **Budgeted hybrid:** `token_budget tau` piggyback-with-budget packs all decodes + optional prefill chunk up to tau [Sarathi].
3. **Preemptive MLFQ:** Skip-join Multi-Level Feedback Queue token granularity [FastServe].
4. **Migration / Theory:** Virtual-usage live migration pipelined KV <decode downtime 15x P99 (Llumnix); online batch algorithm with competitive ratio + profiling predictor (OnlineScheduling); hotness tracking/admission threshold 10 (HotPrefix).

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Orca 2022):** Tight coupling sacrifices layered abstraction; `max_bs` manual; no prefix sharing.
- **Author-stated (Sarathi 2024):** tau static per model/HW (512 strict vs 2048 relaxed) not per-request adaptive; TTFT may increase; no fairness preemption integrated.
- **Author-stated (HotPrefix 2026):** Threshold sensitivity; single-tier; not evaluated beyond 16 GPUs.
- **Author-stated (OnlineScheduling 2025):** Sparse single paper, proof-of-concept; requires competitive ratio proof only for specific model.
- **Inferred (repeated):** Still FCFS per priority class -> fairness/SLO balancing not solved; prediction errors reintroduce fluctuation (Mooncake Fig9-10 anti-phase).

---

## Bottleneck 9 — Poor Reuse Prediction / Hotness Misclassification

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Scheduler / Memory manager (eviction policy)**. Secondary: **Cluster** (routing based on predicted hit).

### Trigger condition (when does it appear, workload/hardware condition)
- Skewed popularity: top 3% docs 60% hits (20x uniform) [RAGCache]; but PGDSF must estimate Cost via bilinear T(l,u) profiling, sensitive to error.
- Agent workflows: LRU evicts soon-to-reuse Expresser while retaining unlikely suffixes -> frequent misses [KVFlow 2025].
- RAG contamination: CCI low = reusable but external prefix overlap beta_prime low -> need CFO tradeoff [Cache-Craft]; mis-estimating leads to 50% F1 drop.
- HotPrefix filter aging clock drift at high RPS may misclassify.

### Why it hurts (mechanism)
If predictor underestimates future reuse, hot block evicted -> miss penalty includes recomputation or PCIe/RDMA fetch (3-6 s). Overestimates -> cache pollution holding dead entries, starving truly hot entries, increasing tail. For agents, prediction error wastes prefetch BW and pollutes cache.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Hit rate delta** vs oracle: RAGCache speculative may add load at high RPS; KVFlow 2.19x over SGLang when prediction correct.
- **Reuse prediction accuracy** (steps-to-execution vs actual).
- **TTL cost model** P(tau,f)·(T·psi+Prefill-Reload) - alpha·Cost [Continuum].
- **JCT (Job Completion Time)** 8x improvement when TTL correct (Continuum).

### Papers addressing it (2-6, with year)
- RAGCache (2024) — PGDSF Priority = Clock+Freq·Cost/Size
- HotPrefix (2026) — Cuckoo 4x4 filter + (freq+clock)/len leaf-only
- KVFlow (2025) — Agent Step Graph steps-to-execution max/min
- Continuum (2025) — TTL retention optimal tau*
- Cache-Craft (2025) — CCI/CFO contamination score
- KVLink (2025) — trained link-token attention fusion vs variance recompute

### Existing solution families (summarize approaches)
1. **Frequency+Cost aware:** PGDSF clock + cost/size where cost via T(l,u) bilinear interpolation [RAGCache].
2. **Hotness + Bloom-like:** Cuckoo filter n buckets x4 entries {fingerprint,clock(8b),freq,depth}, aging clock-1, admission hotness=freq·clock >=10 [HotPrefix].
3. **Workflow graph:** Steps-to-execution = max/min over predecessors, propagate min among radix children, evict larger steps first + proactive prefetch status-aware [KVFlow].
4. **TTL optimization:** tau* = argmax P(tau,f)·gain - alpha·Cost via offline trace fitting [Continuum].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (RAGCache 2024):** Coarse cost profiling; host memory 192 GB assumption; not evaluated for heterogeneous CXL.
- **Author-stated (HotPrefix 2026):** Requires accurate Step Graph; sensitivity to threshold 10.
- **Author-stated (Continuum 2025):** Only ReAct linear traces; rare tool tail fallback global.
- **Author-stated (KVFlow 2025):** Requires accurate Step Graph; branching wastes BW if prediction wrong.
- **Inferred (repeated):** Perfect future knowledge assumed for upper bounds (DynamicPlacement GH200 5.87x); rare tail switch to global heuristic; no privacy isolation.

---
## Bottleneck 10 — Load Imbalance & Autoscaling Inefficiency

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Cluster (global router / Conductor / Planner)**. Secondary: **Network** (hotspot replication) and **Scheduler** (role assignment).

### Trigger condition (when does it appear, workload/hardware condition)
- Prefix skew: hot blocks tens of thousands accesses vs >50% never reused [Mooncake]; KV-aware routing concentrates load -> hotspot congestion Conductor must replicate.
- P/D ratio mismatch: Mooncake 2P+2D worse TTFT than 3P+1D despite more decode nodes (Fig11) due to load imbalance; conversation vs coding traces need different ratios (35P+5T coding vs 25P+15T conversation on HH [Splitwise Fig20]).
- Poisson burst 0.3->9.0 QPS load sweep shows QPS 1.54 vs 11.32 with same capacity when imbalanced [Beluga Fig11].
- Multi-tenant heterogeneous HBM (24-96 GB) vs CXL pool TBs.

### Why it hurts (mechanism)
Imbalanced P/D or hot cache skew leaves some GPUs idle while others queue (e.g., prompt machines idle most time but run larger batches when active vs token machines better batched [Splitwise Fig17]). Hot replica incurs extra prefill vs transfer tradeoff not optimized; poor autoscaling increases TBT/SLO violations and wasted power.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Load ratio** T_queue+T_prefill+T_transfer per instance vs thresholds `kvcache_balancing_threshold` manual [Mooncake].
- **Throughput QPS at same power** (Splitwise-AA 2.15x vs Baseline-A100 iso-power; 1.4x vs H100 iso-cost).
- **Hotspot replication count** vs transfer time estimation.
- **Autoscaling cost** 36% saving via Llumnix virtual-usage, Elise? vs static.
- **P/D utilization** % per pool.

### Papers addressing it (2-6, with year)
- Mooncake (2025) — Conductor KV-aware balancing + hot-spot replication
- Llumnix (2024) — live migration virtual-usage unified LB/defrag 15x P99
- FlowKV (2025) — Global Controller load-aware 3-scenario (Normal/Imbalanced/Extreme) + hybrid scheduler role flip
- Dynamo (2025) — Smart Router radix-aware + Planner SLO autoscaling + KV Block Manager
- DynamicPlacement (2025) — GH200 formal placement min sum max(t^h,t^e) + SA search over window W,ratio R 5.87x bound
- LoongServe (2024) — Elastic Sequence Parallelism dynamic DoP 3.85x vs chunked

### Existing solution families (summarize approaches)
1. **KV-aware routing + balancing:** Find best_prefix_len via prefix hash block512, estimate TTFT = T_queue+T_prefill+T_transfer, route min, threshold trades recompute vs transfer, replication if extra prefill < transfer [Mooncake].
2. **Live migration:** Pipelined KV migration <decode downtime + virtual-usage traction [Llumnix].
3. **Global elastic:** Load-aware controller monitors hit+load, local hybrid scheduler shares block manager, idle nodes flip roles [FlowKV]; Grove K8s + PB offload [Dynamo].
4. **Formal placement search:** Enumerate window W and ratio R via SA search vs bandwidth-aware placement Alg1 (high affinity) vs Alg2 (co-locate) [DynamicPlacement, DistServe].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Mooncake 2025):** Balancing threshold manually tuned; prediction hard due to congestion, coarse.
- **Author-stated (Llumnix 2024):** Evaluated 16 GPUs only; not shown for 100s nodes.
- **Author-stated (FlowKV 2025):** Global controller single point + reconstruction cost not quantified for extreme scale.
- **Author-stated (DynamicPlacement 2025):** Perfect pre-knowledge not realizable; offline SA not online; PCIe5/CXL3 not evaluated beyond GH200.
- **Inferred (repeated):** Radix sync consistency lag not quantified; heterogeneous RDMA vs CXL mix not evaluated for autoscaling.

---

## Bottleneck 11 — Tail Latency (P95/P99 TBT, TTFT SLO Violations)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Scheduler / Runtime**. Secondary: **Network (transfer jitter)** and **GPU kernels (tile quantization)**.

### Trigger condition (when does it appear, workload/hardware condition)
- High QPS near capacity: vLLM shows generation stalls seconds Yi-34B 128 requests [Sarathi Fig1], tail spikes with load [Sarathi Fig1b].
- Hybrid batches without budget: Decode+Full Prefill raises TBT 28.3x vs decode-only [Sarathi Fig9]; production offline profile 5x decode iteration as SLO tight [Sarathi Table3].
- Long prompts 4K-32K vs batch32 interleaving -> per-iteration variance, PP bubble ~950 ms [Sarathi].
- Congested hotspots or PCIe 25 Gbps shared -> P99 TTFT 44.6 vs 5.02 [Beluga Table5]; vLLM only 57% meet TBT vs Mooncake ~100% [Mooncake Fig13].

### Why it hurts (mechanism)
SLOs defined as P90 TTFT (10x isolated baseline) and TBT (5x) [Mooncake] or strict 0.1s/1s [Sarathi]. Tail spikes cause goodput collapse (only completed requests count -> wasted prefill). Scheduler reordering may increase jitter; kernel tile quantization 257->32% slowdown vs 256 [Sarathi] adds variance. Transfer queuing amplifies.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **P50/P90/P99 TTFT/TBT** ms, **P99 TBT SLO attainment** % (Sarathi strict 0.1s vs relaxed 0.5s).
- **Capacity QPS** meeting 90%/99% attainment (DistServe 2.0-4.6x vs vLLM at 90%, 3-8x at 99%).
- **Pipeline bubble** ms and **tail jitter**.
- **Goodput** RPS good vs total.

### Papers addressing it (2-6, with year)
- Sarathi-Serve (2024) — stall-free chunked 28.3x bound
- Orca (2022) — iteration-level reduces queuing to 1 iteration 36.9x throughput at 190 ms/token
- FastServe (2023) — token preemption + skip-join MLFQ 31.4x SLO throughput
- DistServe (2024) — goodput placement + M/D/1 Avg_TTFT model
- Mooncake (2025) — Conductor predicts T_queue+T_prefill+T_transfer minimizes P99 44.6->5.02
- Beluga (2025) — near-local CXL 89.6% TTFT P99 reduction

### Existing solution families (summarize approaches)
1. **Stall-free budgeting:** tau=512 strict/2048 relaxed via Vidur profiling -> uniform hybrid batches bound TBT [Sarathi].
2. **Iteration & preemptive:** Iteration-level single-iteration schedule [Orca]; MLFQ token preemption [FastServe].
3. **Goodput modeling:** M/D/1 Avg_TTFT = D + R·D^2/(2·(1-R·D)) etc., searching inter/intra parallelism per phase [DistServe].
4. **Prediction early rejection:** Advances decoding load assessment before prefill + system-level uniform td prediction to mitigate anti-phase fluctuation (Baseline 4183->3589 rejects) [Mooncake].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Sarathi 2024):** Tight SLO requires offline Vidur profiling; chunk overhead 25% at 512; no disaggregated comparison.
- **Author-stated (DistServe 2024):** Assumes Poisson predictable over hours/days; single-GPU/2-GPU limited search space; not for throughput-optimized batch jobs.
- **Author-stated (Mooncake 2025):** System-level uniform td coarse; request-level prediction too costly/inaccurate left future; goodput counts only fully completed.
- **Inferred (repeated):** P99 not primary in early schedulers; real bursty production traces may exceed Poisson tail more; fairness vs SLO tradeoff not integrated.

---

## Bottleneck 12 — Long-Context Memory Pressure (Quadratic Attention & OOM)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **GPU HBM + Attention kernel**. Secondary: **Cluster** (elastic scaling needed 1K->1M variance).

### Trigger condition (when does it appear, workload/hardware condition)
- Prompt 1K-1M variance: ESP must handle 1K vs 1M dynamically [LoongServe 2024]; prefill compute superlinear due to quadratic attention [Mooncake Fig2].
- Context 128K per request -> KV tens of GB; ShadowKV shows 1.56% budget needed but still 380K max on A100 with 1024 budget vs OOM at 33K vanilla [SnapKV, ShadowKV].
- LongBench GovReport 1,235-18,409 tokens, Ruler 1M, LV-Eval >15K tokens each push HBM.
- Autoregressive generation up to 1M tokens with 200K prompts [ShadowKV].

### Why it hurts (mechanism)
Attention complexity O(n^2) memory and compute; prefill latency and KV size scale superlinear, saturates HBM and bandwidth. Middle retrieval lost-in-the-middle when cache evicts; heavy-hitter drift during decode [SCOPE 2025] degrades. Single-GPU cannot fit context -> must offload or distribute, but transfer and coherence add cost.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Max context before OOM** (e.g., 33K vanilla vs 380K SnapKV).
- **TTFT seconds** vs length (4K 3-6 s on A40).
- **Ruler/LongBench/Needle retrieval accuracy** % (SnapKV negligible drop before 140K, PyramidKV 12% cache matches full 41.46).
- **Throughput tokens/s** vs batch under 128K (ShadowKV 60K 455 vs 160 tokens/s).
- **SVD relative overhead** decreasing with length (Fig1 ShadowKV right).

### Papers addressing it (2-6, with year)
- ShadowKV (2025) — low-rank pre-RoPE 6x batch 3.04x throughput
- StreamingLLM (2023) — attention sinks 4M stable 22.2x speedup
- SnapKV (2024) — voting + pooling 380K on A100
- PyramidKV (2024) — pyramidal funneling 12% matches full
- SCOPE (2025) — phase-aware prefill/decode drift fix
- LoongServe (2024) — ESP elastic sequence parallelism 3.85x

### Existing solution families (summarize approaches)
1. **Low-rank / landmark:** SVD rank160 on pre-RoPE keys (S·r+r·M vs 2·S·M) + chunk mean landmarks (C=8) + 48 outliers [ShadowKV].
2. **Sink / sliding window:** Keep 4 initial sinks pinned + rolling window, cache-relative RoPE re-encoding [StreamingLLM]; observation-window last segment voting [SnapKV].
3. **Layer-adaptive / semantic chunk:** Arithmetic pyramid k^l = k0-(k0-k^{m-1})/(m-1)·l [PyramidKV]; semantic chunk c=10 sum attention preserving order [ChunkKV].
4. **Elastic / phase-aware:** ESP dynamic DoP per request proactive scale-down + multi-master decode overlapping [LoongServe]; separate prefill Lambda^p vs decode Lambda^d adaptive hatLambda [SCOPE].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (StreamingLLM):** Does NOT extend context, only recent coherence; fixed 4 sinks heuristic may be wasteful/insufficient future.
- **Author-stated (ShadowKV):** SVD cost for short prompts; uniform rank 160 not adaptive; CPU offload OOM risk beyond 48 batch x128K needs SSD.
- **Author-stated (PyramidKV):** Fixed alpha=20 pyramid shape not optimal; no dynamic per-input adaptation; fixed chunk 10 ignores boundaries.
- **Author-stated (SCOPE):** Phase-aware still needs offline alpha tuning; not evaluated beyond 1M with heterogeneous hardware.
- **Inferred (repeated SnapKV, PyramidKV, ShadowKV):** Uniform budget wastes lower layers; no joint quantization; middle-retrieval guarantees not formal; per-layer/head dynamic tuning missing.

---
## Bottleneck 13 — Hierarchical Tiering & Heterogeneous Memory Management Complexity

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Node (GPU HBM + CPU DRAM + CXL pool + SSD/Remote)**. Secondary: **Cluster (global tiering policy)** and **Scheduler**.

### Trigger condition (when does it appear, workload/hardware condition)
- Local HBM insufficient for 75% more requests (+50 GB+ contexts) -> must spill to CPU DRAM 192-2000 GB, CXL 8 TB pool at 1 TB/s, SSD/NVMe 1.5 TB @2 GB/s, Remote DRAM pooled 4 TB via RDMA.
- Heterogeneous bandwidth: GPU HBM 2 TB/s vs CPU DRAM 50 GB/s vs PCIe 64 GB/s vs CXL 750 ns 64B vs RDMA 200-400 Gbps vs S3 1 GBps.
- CXL2 lacks host-host coherence [Beluga 2025]; needs software ntstore+CLFLUSH or UC +mfence.
- Dynamic workload needs decision where to place new KV (GPU/CPU/disk) per LP solver [FlexGen, DynamicPlacement].

### Why it hurts (mechanism)
Each tier has distinct latency/BW/capacity/consistency. Wrong placement -> excessive sync (75% overhead), swap thrashing, or recomputation cheaper than fetch (remote S3 22-32% faster than full prefill only for long contexts at 1 GBps [LMCache]). Coherence software must avoid stale reads across sockets while avoiding UC slowdown (281 ms for 16 KB [Beluga]). Tiering logic interacts with scheduler predictions.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Tier hit rate** % per tier (e.g., HBM 14.6% peak with 28.3 GB [Beluga]).
- **Placement throughput upper bound** GH200 formal min sum max(t^h,t^e) s.t. P_H <=100% ->5.87x upper bound vs HBM-only [DynamicPlacement 2025].
- **BW realized** GB/s per path (CPU RC 33 vs 46.2 expected, GPU->CXL 26 vs 55.4 PCIe [Beluga]).
- **Latency overhead** % transfer vs compute (1.52x vs Ideal vs 3.9-18x others [InfiniGen]).
- **Cost/energy vs capacity** not yet standard.

### Papers addressing it (2-6, with year)
- Beluga (2025) — CXL 2.0 switch 2x PCIe5 + XC50256 256-lane 2 TB/s
- DynamicPlacement (2025) — GH200 theoretical model + SA search over W,R
- InfiniGen (2024) — rehearsal prefetch CPU pool
- CachedAttention / AttentionStore (2024) — DRAM->SSD multi-turn hierarchical
- Shared RAG-DCache (2025) — disk cross-instance pre-generation + queuing window
- LMCache (2025) — enterprise controller + 8+ tiers (CPU/SSD/Remote/Redis/S3/NFS/WEKA) + batched admit/evict

### Existing solution families (summarize approaches)
1. **CXL switched pool:** 2 adapters PCIe5 x16 per server, 2 chips XC50256 2 TB/s ->16 servers 8 TB at 1 TB/s, mmap DAX, GPU direct P2P eliminating bounce buffer, Uncacheable+DDIO-disable coherence [Beluga].
2. **Formal placement:** GH200 model + SA over window W and ratio R [DynamicPlacement].
3. **Hierarchical DRAM->SSD:** Async save, layer-wise preload, scheduler-aware fetch/eviction, decoupled positional encoding 87% TTFT [CachedAttention].
4. **Enterprise layer:** Connector abstraction + 256-token chunk streaming coalesce + centralized Controller lookup/move/pin/compress + batched p2p [LMCache].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (Beluga 2025):** Software coherence needed (RC bottleneck), future direct GPU-to-CXL fabric needed (O7); capacity vs BW tradeoff beyond 16 servers saturates 2 TB/s/chip not measured; switch single point of failure.
- **Author-stated (DynamicPlacement 2025):** Perfect pre-knowledge not realizable; offline SA not online; PCIe5/CXL3 not evaluated beyond GH200.
- **Author-stated (LMCache 2025):** Low BW remote only wins for >256K at 32 Gbps; Python->Rust pending; 500 GB CPU DRAM still insufficient for many tokens—needs auto-tiering not formalized.
- **Author-stated (CachedAttention 2024):** Hierarchical limited to chat; eviction under real pressure not measured beyond ShareGPT.
- **Inferred (repeated):** Cost/power of CXL switch vs RDMA NIC not stable; multi-tier prefetch scheduling not jointly optimized with network transfer.

---

## Bottleneck 14 — Compression-vs-Quality Degradation (Quant / Eviction Error Accumulation)

### Where it occurs (layer: GPU HBM / PCIe / network / scheduler / cluster)
Primary: **Attention kernel / Memory manager**. Secondary: **Scheduler** (must trade memory for SLO).

### Trigger condition (when does it appear, workload/hardware condition)
- Extreme eviction (0.7% cache =64 tokens avg input ~13K) -> SnapKV/H2O collapse, TREC drop 38->58 (+20.5) after PyramidKV [PyramidKV 2024].
- Aggressive quantization 2-bit per-token for keys mixes outlier channels -> error 13.67 vs per-channel 4.55, attention error 47.00 vs 9.60 [KIVI 2024]; at 2-bit CoT loss 40.52->25.25 vs GEAR 40.20.
- Long chain CoT (GSM8K) or multi-hop retrieval where all tokens needed -> dense attention assumption breaks.
- Joint need: MQA single KV head more sensitive (Falcon 2-bit 59.83->57.48 larger drop [KIVI]).

### Why it hurts (mechanism)
Attention sparsity power-law suggests few heavy hitters dominate, but recent tokens always matter and outliers in fixed channels dominate per-token quantization. Heavy-hitter drift during decode [SCOPE] and varying KV count per layer/query [InfiniGen C1-3] make fixed budget inadequate. Error compounds autoregressively, disproportionately affecting tasks requiring whole-context aggregation vs retrieval.

### Observable metrics (TTFT/TPOT/P95/memory hit rate etc)
- **Accuracy drop** % / delta F1/ROUGE vs full cache (H2O 5% retains, below ->35% drop; KIVI 2-bit near lossless +2.17 GSM8K window; PyramidKV 64->41.49 vs Full 41.46).
- **Reconstruction error** ||X-X''''||_F / ||X||_F (Key per-token 13.67 vs per-channel 4.55), ||A-A''''||_F.
- **Perplexity** near parity at 20% H2O, diverges beyond.
- **Memory vs accuracy Pareto** curve.

### Papers addressing it (2-6, with year)
- KIVI (2024) — asymmetric per-channel key / per-token value 2-bit residual R=128
- GEAR (2024) — quant + low-rank (r=4/2) + sparse (s/2% outliers) 2.39x peak
- H2O (2023) — heavy-hitter submodular + recent window 5% budget
- PyramidKV (2024) — layer pyramidal allocation alpha=20
- SnapKV (2024) — observation-window voting + pooling clustering
- SCOPE (2025) — prefill/decode separated optimization fixing drift

### Existing solution families (summarize approaches)
1. **Asymmetric grouping:** Key per-channel G=32 vs Value per-token G=32 split grouped quantized + residual buffer R=128 sliding window [KIVI].
2. **Composite error compensation:** Optimize min||X-D^q-L-S||_F where D^q quantized backbone (KCVT 4-bit / KIVI g64 2-bit), S top/bottom s/2% outliers FP16, L head-wise low-rank via power iteration [GEAR].
3. **Heavy-hitter + pyramid:** Dynamic submodular F_score(T)=sum o_s (1-1/e bound) + recent window [H2O]; pyramidal arithmetic decreasing k^l across layers [PyramidKV]; observation-window sum attention per head top-k per head [SnapKV].
4. **Phase-aware:** Separate prefill Lambda^p invariant vs decode Lambda^d adaptive hatLambda [SCOPE].

### Remaining limitations reported in literature (author-stated or repeated inferred, cite sources)
- **Author-stated (KIVI 2024):** Uniform G/R not optimal per layer/head; no integration with eviction; Falcon MQA needs 4-bit.
- **Author-stated (GEAR 2024):** Uniform rank/sparsity (r=4/2, s=2%); no joint eviction; power-iteration jitter.
- **Author-stated (H2O 2023):** Inference-only, submodular assumption may not hold universally; sequential dependency risk aggressive budget.
- **Author-stated (PyramidKV 2024):** Fixed alpha=20 heuristic, not dynamic per input; fixed chunk 10.
- **Inferred (repeated):** No theoretical bound for landmark-chunk error; uniform group size impact under long input due to scale granularity; per-layer budget still manual not learned.

---

## Cross-Bottleneck Synergy & Trade-offs (Organizational Evidence Only)

- **Capacity vs Bandwidth:** Offloading solves capacity (Bottleneck1) but creates PCIe/Network (5,6) -> compensated by sparse fetch (InfiniGen <10%) or CXL (Beluga) or coalescing (FlowKV).
- **Hit Rate vs Load:** Higher hit via KV-aware routing (10) risks hot-spot imbalance -> needs threshold tuning (Mooncake) and replication.
- **Recompute vs Transfer:** For RAG non-prefix case, choose min(T_recompute, T_load) per chunk (CacheBlend vs CacheGen codec) -> controller picks storage vs recompute.
- **Interference vs Throughput:** Chunked stall-free (7) trades prefill speed for TBT bound; Disaggregated (7) trades double weight memory for interference elimination.
- **Compression vs Memory:** Token eviction (14) + quant (14) + paging (2) are orthogonal stacking per TMLR Survey taxonomy; LMCache/Dynamo demonstrate via connectors.
- **Prediction accuracy (9) underlies all:** Miss prediction turns 5,6,10 into extra recompute; no paper claims >90% reuse in real skewed workloads (Mooncake 50% plateau) — fundamental workload limit.

## Evidence Gaps & Repeated Limitations (Summary)

Repeated author-stated gaps across >=2 papers:
- **Manual thresholds:** kvcache_balancing_threshold (Mooncake), admission 10 (HotPrefix), alpha=20 (PyramidKV), tau=512/2048 (Sarathi), R=128 (KIVI) all offline/static; no online auto-tuning proven.
- **Single-GPU PCIe 3.0 / limited heterogeneity:** Many evaluations A6000/A100 only; H100/NVL72/GB200 30x (Dynamo) not generalizable; consumer GPU gaps (Beluga) not covered.
- **Evaluation scale <=32 GPUs / 1M max:** No 100s-node Conductor bottleneck or RDMA incast measured; real burst multi-tenant SLOs underrepresented (synthetic Poisson).
- **Storage cost not quantified:** Dollar/power cost of CPU/CXL/SSD pooling vs adding GPUs missing (FlexGen, Beluga, LMCache all note cost favorable but no TCO).
- **Isolation/privacy:** Cross-request CoW (vLLM) and cross-instance sharing (Shared RAG-DCache) risk timing side-channel leakage; not evaluated.

## Sampling Evidence for Audit

Read >=24 notes covering all manifest categories: serving foundations (Orca 2022, vLLM 2023, SGLang 2024, LMCache 2025), memory/fragmentation (vLLM, SGLang, LMCache, FlexGen), eviction (StreamingLLM 2023, H2O 2023, SnapKV 2024, PyramidKV 2024), compression (KIVI 2024, GEAR 2024, ChunkKV 2025, ShadowKV 2025), prefix reuse (SGLang, RAGCache 2024, HotPrefix 2026, CacheBlend 2025, KVLink 2025, Cache-Craft 2025), offloading (FlexGen, InfiniGen 2024, CachedAttention 2024, Beluga 2025), scheduling (Sarathi-Serve 2024, FastServe 2023, OnlineScheduling 2025, Llumnix 2024), disaggregation (Splitwise 2024, DistServe 2024, DejaVu 2024, FlowKV 2025, AMPD 2026, Mooncake 2025), RAG (RAGCache, CacheBlend, Cache-Craft, KVLink, SharedRAG-DCache 2025), agent (KVFlow 2025, Continuum 2025), heterogeneous (Beluga, DynamicPlacement 2025, ShadowKV), surveys (KVSurvey TMLR 2025, FromAttention 2025), plus taxonomy_draft.md and papers.md (43).

*End of bottleneck_map.md — 14 bottlenecks full sections; ready for Stage 2B idea generation but not generating ideas here.*
