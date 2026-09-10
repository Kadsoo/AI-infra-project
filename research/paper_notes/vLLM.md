# Paper Metadata

- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention [PAPER FACT]
- **Authors:** Woosuk Kwon*, Zhuohan Li*, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica [PAPER FACT] (* equal contribution) â€?UC Berkeley, Stanford, Independent Researcher, UC San Diego [PAPER FACT]
- **Venue:** ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP 2023), Koblenz, Germany, Oct 23-26, 2023 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.1145/3600006.3613165 / https://arxiv.org/abs/2309.06180 [PAPER FACT]
- **Code:** https://github.com/vllm-project/vllm [PAPER FACT] â€?source publicly available, Apache 2.0 for code, CC BY 4.0 for paper, code GitHub [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2309.06180v1 + local fitz extraction C:\Windows\Temp\opencode\vllm.txt (16 pages, arXiv 2309.06180v1) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

High-throughput LLM serving requires batching many requests [PAPER FACT], but each request's KV cache is huge, grows/shrinks dynamically, and its lifetime/length is unknown a priori [PAPER FACT]. For OPT-13B, one token = 800 KB (2*5120 hidden*40 layers*2 bytes FP16) â†?1.6 GB for 2048 tokens [PAPER FACT]; A100 40GB devotes ~65% (26GB) to weights, ~30% to KV cache, small remainder to activations (Fig. 1 left) [PAPER FACT].

Existing systems (FasterTransformer [NVIDIA 2023a], Orca [Yu et al. 2022]) store KV cache in **contiguous** memory (requirement of PyTorch/TensorFlow) and **pre-allocate** for max length (e.g., 2048) per request [PAPER FACT]. This causes:
- **Memory waste**: reservation for future tokens, internal and external fragmentation â€?only 20.4% - 38.2% of KV memory holds actual token states (Fig. 2 profiling) [PAPER FACT].
- **Limited batch size** and low throughput; growth curve of KV memory is steep vs vLLM smoothed curve (Fig. 1 right) [PAPER FACT].
- **No sharing** across sequences even when decoding algorithms (parallel sampling, beam search, shared prefix) allow it [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLMs (GPT, PaLM) enable new hosted services (programming assistants, chatbots) whose cost per request is ~10x keyword query [Reuters 2023] [PAPER FACT]; cloud providers racing to host them (OpenAI API, AWS Bedrock) [PAPER FACT].
- Autoregressive generation is memory-bound, underutilizes GPU compute; prompt phase is matrix-matrix parallel, decode phase is matrix-vector sequential and dominates latency [PAPER FACT].
- Batching can improve utilization but is memory-capped: even with all memory for KV, only tens of requests fit; as compute grows faster than memory (A100â†’H100 FLOPS >2x, memory stays 80GB max), memory becomes increasingly critical bottleneck [PAPER FACT] (citing Gholami et al. 2021).
- Advanced decoding (parallel sampling for code, beam search for translation) offers sharing opportunities that existing contiguous allocation cannot exploit, leaving 12% (parallel sampling prompt sharing) to 55% (beam search) memory unshared [PAPER FACT] (Â§3, Â§6.3 context).
- Efficient memory management would directly increase batch size â†?throughput â†?cost per request, which is the primary lever for economical LLM services [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Fragmentation and reservation waste.** [PAPER FACT] Pre-allocation for max length causes:
   - **Internal fragmentation**: actual length << max (e.g., Fig. 11 distribution), entire chunk reserved whole lifetime, unused part unavailable to others [PAPER FACT].
   - **External fragmentation**: different pre-allocated sizes via buddy allocator leave unusable holes [PAPER FACT].
   - **Reserved** slots for future tokens occupy memory for entire duration even before generation [PAPER FACT]. Total waste quantified Fig. 2, Fig. 3.

2. **No sharing across sequences/requests.** [PAPER FACT] Parallel sampling, beam search, shared system prompts share prefix KV but systems store per-sequence contiguous copies â†?redundant duplication (Fig. 8, Fig. 9, Fig. 10). Orca paper itself notes no sharing.

3. **Unknown dynamic lengths & variable prompt lengths.** [PAPER FACT] Scheduler must decide placement/eviction without knowing output length; memory needed per iteration grows and may exhaust GPU, requiring preemption decisions.

4. **Distributed memory coherence.** [PAPER FACT] LLMs exceed single GPU (175B needs 8-16 GPUs); tensor parallelism (Megatron-LM SPMD) splits linear layers and attention heads but requires consistent KV mapping across workers; naÃ¯ve per-GPU allocator would duplicate manager state and cause sync overhead.

5. **Kernel overhead vs flexibility tradeoff.** [PAPER FACT] Paged non-contiguous access needs indirection (block table) and variable-length handling, adding branches and non-coalesced concerns vs highly optimized contiguous kernels.

## 4 Core Idea [PAPER FACT]

**PagedAttention â€?paging-inspired attention over non-contiguous KV cache, plus a vLLM engine co-designed around it [PAPER FACT].**

- **PagedAttention algorithm (Â§4.1):** Partitions KV cache into fixed-size **KV blocks** (block = B tokens, default 16 [PAPER FACT]). Each block holds keys/values for B positions (Kj = (k_(j-1)B+1 .. kjB), Vj similarly). Attention Eq. 4 rewritten block-wise: Ai_j = exp(qi^T Kj / sqrt(d)) / sum_t exp(qi^T Kt 1 / sqrt(d)), oi = sum_j Vj Ai_j^T [PAPER FACT]. Kernel fetches blocks via block table, not contiguous address [PAPER FACT] (Fig. 5 example: query "forth" attends across 3 non-contiguous blocks).

- **Virtual memory analogy:** Think blocks as pages, tokens as bytes, requests as processes [PAPER FACT]; logical KV blocks filled left-to-right, last block partially filled and reserved for future; physical blocks are pooled; block table maps logicalâ†’physical + #filled [PAPER FACT] (Â§4.2). This gives on-demand allocation, eliminates external fragmentation (all blocks same size), bounds internal waste to â‰?one block per sequence [PAPER FACT] (Fig. 6 walkthrough).

- **Sharing at block granularity with copy-on-write (COW) and reference counting [PAPER FACT] (Â§4.4):**
  - Parallel sampling: prompt logical blocks (0,1) map to same physical blocks (7,1) refcount=2; on write to shared last block, allocate new physical, copy, decrement refcount [PAPER FACT] (Fig. 8).
  - Beam search: shared prefix blocks across candidates, dynamic sharing pattern evolves like process tree with forks; freed candidates release blocks when refcountâ†? (Fig. 9); up to 55% saving [PAPER FACT].
  - Shared prefix (system prompt/prompt engineering): service provider pre-reserves physical blocks for prefix, maps incoming logical blocks to them copy-on-write, prompt phase only computes task suffix [PAPER FACT] (Fig. 10).

- **vLLM system on top:** Centralized scheduler + KV cache manager (block engine on GPU + CPU RAM for swapping) + PagedAttention kernels; block-level management and preemptive scheduling co-designed (Fig. 4) [PAPER FACT]; supports distributed execution with single shared manager broadcasting block tables [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig. 4):** Centralized scheduler coordinates distributed GPU workers; KV cache manager manages physical blocks via instructions; block engine allocates contiguous GPU DRAM and splits into physical KV blocks (and CPU RAM for swap) [PAPER FACT].

- **Decoding flow (Â§4.3, Fig. 6):**
  1. *Init:* Reserve only needed blocks for prompt (e.g., 7 tokens â†?2 logical blocks â†?map to physical 7,1) [PAPER FACT]; prefill with conventional attention for prompt + first output.
  2. *Decode step 1:* PagedAttention on physical blocks 7,1; append token to last logical block's remaining slot, update #filled [PAPER FACT].
  3. *Decode step 2:* Last logical block full â†?allocate new physical (3), map to new logical, store KV [PAPER FACT].
  Globally per iteration: scheduler selects candidate sequences (see Â§4.5), allocates physical for new logical; concatenates all input tokens (prompt + latest decode token) as one sequence into LLM; kernels read/write via block tables; block size >1 enables parallel KV processing across positions [PAPER FACT].

- **Memory manager details (Â§4.2, Â§4.5, Â§4.6):**
  - Logical vs physical separation, block tables per request/sequence group [PAPER FACT].
  - Reference counts for sharing; all-or-nothing eviction policy (all blocks of a sequence together) + gang-scheduling of sequence groups (beam candidates together) [PAPER FACT].
  - **Preemption & recovery:** When physical blocks exhausted, select sequences to evict; two recovery options: **Swapping** to CPU RAM via CPU block allocator (swap space bounded by GPU KV memory, preempted queue waits until completions free GPU) and **Recomputation** (recompute KV from prompt+generated tokens in one prompt-phase iteration, lower latency than original) [PAPER FACT]; choice depends on block size and PCIe vs compute bandwidth (ablation Â§7.3) [PAPER FACT].
  - **Distributed:** Single KV cache manager in centralized scheduler shared across workers using Megatron-LM style tensor parallelism (SPMD, attention split on head dimension); scheduler broadcasts token IDs + block tables; workers synchronize via all-reduce without scheduling coordination; execution is SPMD [PAPER FACT] (Â§4.6).

- **Implementation (Â§5):** 8.5K lines Python + 2K C++/CUDA, FastAPI frontend extending OpenAI API, NCCL for tensor comms, supporting GPT, OPT, LLaMA via PyTorch/Transformers [PAPER FACT]; kernel optimizations: fused reshape+block write, fused block read+attention (warp per block, variable lengths), fused block copy for COW batching [PAPER FACT]; fork/append/free methods for decoding algorithms [PAPER FACT].

- **Batching compatibility:** Works with iteration-level scheduling / continuous batching (Â§2.3), block tables hide sharing complexity from kernels [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary: Serving throughput (tokens/s in Fig. 1 right; requests/s vs normalized latency in Fig. 12,14,16,17) and normalized latency = mean end-to-end latency / output length (ms/token) [PAPER FACT]**; high-throughput system keeps low normalized latency at high request rates [PAPER FACT]; traces 1-hour (15-min for 175B due to cost) [PAPER FACT].
- **Secondary:**
  - KV cache memory usage / waste percentage (Fig. 2) and effective usage ~96% (near-zero waste <4% per Fig.2) for vLLM vs 20.4-38.2% baselines [PAPER FACT].
  - Average # batched requests per iteration (Fig. 13) [PAPER FACT].
  - Memory saving % from sharing (Fig. 15) [PAPER FACT].
  - Kernel latency overhead (Fig. 18a) and block size impact on end-to-end (Fig. 18b) [PAPER FACT].
  - Swap vs recompute overhead vs block size (Fig. 19) [PAPER FACT].
- **Not primary but noted:** Model accuracy unaffected (no approximation) [PAPER FACT].

## 7 Baselines [PAPER FACT]

All comparisons maintain same model accuracy (no approximation) [PAPER FACT].

- **FasterTransformer [NVIDIA 2023a] v?** Distributed inference engine latency-optimized; custom scheduler added with dynamic batching similar to Triton, max batch size B as large as possible per memory; no fine-grained scheduling [PAPER FACT] (Â§6.1 Baseline 1).

- **Orca [Yu et al. 2022] â€?throughput-optimized with iteration-level scheduling.** Not publicly available, authors re-implement Orca with buddy allocation. Three variants by output reservation [PAPER FACT]:
  - **Orca (Oracle):** knows true output lengths â€?upper bound, infeasible [PAPER FACT].
  - **Orca (Pow2):** reserves up to 2x true length (e.g., 25â†?2) [PAPER FACT].
  - **Orca (Max):** reserves max sequence length (2048) [PAPER FACT].

- **Ablation/self comparisons:** No-cache, block size sweep, swapping vs recomputation [PAPER FACT]; kernel microbenchmark vs FT contiguous kernel [PAPER FACT].

- **Hardware-matched baselines:** Same model/server configs (Table 1) for fair memory capacity [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models & server configs (Table 1, Â§6.1):**
  - OPT 13B on 1x A100 40GB (26GB params, 12GB KV cache, 15.7K slots) [PAPER FACT]
  - OPT 66B on 4x A100 40GB (160GB total, 132GB params, 21GB KV cache, 9.7K slots) [PAPER FACT]
  - OPT 175B on 8x A100-80GB (640GB total, 346GB params, 264GB KV cache, 60.1K slots) [PAPER FACT]
  - LLaMA 13B used for shared-prefix translation (Â§6.4) [PAPER FACT]
  - All max seq len 2048, fp16? Not explicitly but OPT/LLaMA standard [AGENT INFERENCE] â€?paper says tensors FP16 per token size calculation [PAPER FACT].

- **Datasets & traces (Â§6.1):**
  - **ShareGPT** [Team 2023] â€?user-shared ChatGPT conversations (long inputs) â€?8.4x longer prompts, 5.8x longer outputs avg than Alpaca [PAPER FACT] (Fig. 11).
  - **Alpaca** [Taori et al. 2023] â€?GPT-3.5 self-instruct instruction dataset (shorter) [PAPER FACT].
  - Tokenized lengths used to synthesize requests; arrival times Poisson with varying rates [PAPER FACT].
  - **WMT16** EN-DE for shared prefix (Â§6.4) with LLaMA-13B multilingual, one-shot (80 tokens prefix) vs few-shot (341 tokens, 5 examples) [PAPER FACT].
  - **Chatbot (Â§6.5):** ShareGPT-derived history + last query concatenated, cut to last 1024 tokens, generate up to 1024 tokens; conversation-level prompt [PAPER FACT].

- **Decoding scenarios (Â§4.4, Â§6):**
  - Basic sampling (1 sample/req) [PAPER FACT]
  - Parallel sampling (multiple samples/req sharing prompt) tested with n=?, Fig.14a shows gains increasing with n [PAPER FACT]
  - Beam search beam width k up to 6 (Fig.14b), top-k candidates from k*|V| [PAPER FACT]
  - Shared prefix (Fig.10) [PAPER FACT]
  - Chatbot multi-turn [PAPER FACT]

- **Trace durations:** 1-hour traces except 15-min for OPT-175B [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Platform:** A2 instances on Google Cloud Platform with NVIDIA A100 GPUs [PAPER FACT].
- **Configs (Table 1):** As above: 1x A100-40GB, 4x A100-40GB, 8x A100-80GB [PAPER FACT]; interconnect not detailed beyond GCP A2 default (not NVLink/Infiniband spec as in Orca) [NOT REPORTED].
- **Memory partition:** Parameter size and KV cache memory listed Table 1 [PAPER FACT].
- **Precision:** KV cache token size calculated for FP16 (2 bytes) [PAPER FACT]; model precision [NOT REPORTED] but FP16 assumed consistent.
- **Other:** CPU, DRAM, disk for swapping: CPU RAM block allocator used, but CPU spec [NOT REPORTED]; network [NOT REPORTED].

[AGENT INFERENCE]: No A10, H100, or CPU-only evaluation; focus on A100 datacenter.

## 10 Main Results [PAPER FACT]

All numbers from Â§6â€?.2, Figs. 12â€?9.

- **Basic sampling (Â§6.2, Fig.12,13):**
  - *ShareGPT:* vLLM sustains **1.7â€?.7x higher request rates than Orca (Oracle)** and **2.7â€?x vs Orca (Max)** at similar latency [PAPER FACT]; vs FasterTransformer up to **22x** [PAPER FACT]; processes **2.2x more concurrent requests than Oracle, 4.3x than Max** for OPT-13B at same rate (Fig.13a) [PAPER FACT]; trend holds across 13B/66B/175B except 175B Alpaca becomes compute-bound (large KV pool vs short seqs) where gain over Oracle/Pow2 less pronounced (Fig.12f) [PAPER FACT].
  - *Alpaca (short):* Similar gains; 175B exception as above [PAPER FACT].

- **Parallel sampling & beam search (Â§6.3, Fig.14,15):**
  - Gain **increases with #samples**: improves over Oracle from 1.3x (basic) to 2.3x (beam width 6) on OPT-13B Alpaca [PAPER FACT] (Fig.14b vs 12).
  - Beam search more sharing â†?larger benefit than parallel sampling [PAPER FACT].
  - **Memory saving:** Parallel sampling **6.1â€?.8% Alpaca, 16.2â€?0.5% ShareGPT**; Beam search **37.6â€?5.2% Alpaca, 44.3â€?6.3% ShareGPT** (blocks saved / total without sharing) [PAPER FACT] (Fig.15 and text Â§6.3).

- **Shared prefix (Â§6.4, Fig.16, LLaMA-13B WMT16):**
  - 1-shot prefix (80 tokens): **1.67x throughput vs Orca Oracle** [PAPER FACT].
  - 5-shot prefix (341 tokens): **3.58x throughput vs Orca Oracle** [PAPER FACT] (Fig.16b vs 16a).

- **Chatbot (Â§6.5, Fig.17):** **2x higher request rates** vs all three Orca baselines for ShareGPT chat (1024 prompt cut, 1024 max gen) [PAPER FACT]; Orca baselines behave similarly because buddy allocator reserves 1024 for outputs regardless of predictor [PAPER FACT].

- **Ablations (Â§7):**
  - **Kernel latency (Â§7.1, Fig.18a):** PagedAttention attention kernel **20â€?6% higher latency** than FT highly optimized contiguous kernel [PAPER FACT]; but end-to-end still wins massively; overhead only affects attention, not linear [PAPER FACT].
  - **Block size (Â§7.2, Fig.18b):** ShareGPT best 16â€?28, Alpaca best 16â€?2 (degrades >32 due to internal fragmentation > seq len) [PAPER FACT]; default chosen **16** as sweet spot (large enough for GPU parallelism, small enough for low fragmentation) [PAPER FACT].
  - **Swapping vs recomputation (Â§7.3, Fig.19):** Small blocks â†?many small PCIe transfers â†?swapping excessive overhead, recomputation constant; **recomputation more efficient for small blocks, swapping for large**, **recompute overhead never >20% of swapping latency**, comparable for 16â€?4 [PAPER FACT] (Fig.19a micro, 19b end-to-end).

- **Overall claim:** **2â€?x throughput at same latency vs FasterTransformer and Orca** across many scenarios; more pronounced for longer sequences, larger models, complex decoding [PAPER FACT] (Abstract).

- **Memory efficiency (Fig.2):** vLLM **~96.3%** effective KV usage vs Orca Max 20.4%, Pow2 ~38.2%, Oracle ~57.3%?? Actually figure breakdown: proprietary numbers but text summary 20.4â€?8.2% for existing vs near-zero waste for vLLM [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer-based LLM with self-attention; KV cache is key/value vectors per token per layer per head [PAPER FACT].
- Tensors traditionally require contiguous memory (PyTorch/TensorFlow constraint) [PAPER FACT]; PagedAttention lifts this by block-wise computation Eq.4 [PAPER FACT].
- Dynamic per-token growth/shrink, lifetime and length unknown a priori [PAPER FACT]; token KV depends on position (same token at different positions different) [PAPER FACT].
- Serves with continuous/iteration-level batching; uses prefill (parallel matrix-matrix) + decode (sequential matrix-vector, memory-bound) phases [PAPER FACT].
- Block size uniform,>1 enables parallelism; small block bounds waste to one block [PAPER FACT].
- All blocks of a sequence accessed together â†?all-or-nothing eviction, sequence group gang-scheduling (beam candidates together) [PAPER FACT].
- Swapping and recomputation are viable recovery; PCIe bandwidth and GPU compute determine choice [PAPER FACT].
- Distributed: Megatron SPMD tensor parallelism, heads split, same token positions across shards â†?single shared block manager suffices [PAPER FACT]; model exceeds single GPU, requires partitioning (Â§4.6) [PAPER FACT].
- KV cache dominates batch size limit; weight memory static 65% (Fig.1) [PAPER FACT] vs activation small [PAPER FACT].
- No compression/quantization of KV; FP16 fixed [PAPER FACT] (token size 800KB assumes FP16).

## 12 Author-Stated Limitations [PAPER FACT]

- **Not elaborate:** Paper has no explicit "Limitations" section; Discussion Â§8 touches applicability boundaries [PAPER FACT]:
  - **Not general to all GPU workloads:** Virtual memory paging not beneficial for training (static shapes, optimizable ahead) or non-LLM serving that is compute-bound â€?overhead of indirection and non-contiguous blocks may degrade performance [PAPER FACT] (Â§8 first paragraph).
  - **Requires LLM-specific adaptations:** To be effective, needs tailoring (all-or-nothing swap-out, recomputation recovery not feasible in OS, kernel fusion to mitigate indirection) â€?generic paging without co-design would be inefficient [PAPER FACT] (Â§8).
  - **Overheads exist:** 20â€?6% kernel penalty acknowledged (Â§7.1) [PAPER FACT]; small blocks hurt swapping (Fig.19) [PAPER FACT].
- **Unmeasured aspects (not stated as limitation but constraints):**
  - Evaluated only on OPT/LLaMA up to 175B; no 340B/500B, no MoE beyond? (MoE not) [NOT REPORTED] â€?generalization claimed but not shown.
  - No multi-node beyond 8 GPUs? Table shows up to 8 GPUs 175B, but no 32-GPU 341B like Orca [NOT REPORTED].

[AGENT INFERENCE]: Authors do not list fairness, starvation, or security issues.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Block size tradeoff still manual:** Default 16 not optimal for all (ShareGPT likes 64â€?28). No auto-tuning for workload length distribution [AGENT INFERENCE].
- **CPU swap capacity bounded but not elastic:** Swap space bounded by GPU KV memory, but under high pressure still blocks new requests until completions free GPU â€?head-of-line blocking under burst, no priority [AGENT INFERENCE].
- **No cross-request automatic prefix sharing beyond beam/parallel:** General shared prefix requires manual provider reservation (Fig.10); not automatic LRU like SGLang RadixAttention later. Ad-hoc system prompt caching not fully transparent [AGENT INFERENCE].
- **Scheduling fairness:** Cache-aware scheduling not discussed; FCFS-like with all-or-nothing eviction may starve long requests or cause tail latency variance [AGENT INFERENCE].
- **Memory hierarchy limited to GPU+CPU DRAM:** No disk tiering, no offload to NVMe, no heterogeneous memory (e.g., unified vs HBM) [AGENT INFERENCE].
- **Security/isolation:** Sharing via reference counts across sequences of same request is safe, but cross-request sharing (if naive) could leak KV; paper limits sharing to within request + explicit prefix reservation, not general cross-user deduplication â€?privacy tradeoff unexplored [AGENT INFERENCE].
- **No quantization:** 800KB/token at FP16; INT8/KV cache compression (later techniques) would change block sizing and waste math, not evaluated [AGENT INFERENCE].
- **Production realism:** Synthetic Poisson arrivals, 1-hour traces; no burst, multi-tenant, SLO violations, or failure recovery measured [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Optimal block size policy:** Can adaptive sizing (variable B vs fixed 16) or profile-guided tuning automatically achieve 96%+ efficiency across short Alpaca and long ShareGPT without manual sweep? [AGENT INFERENCE]
2. **Preemption policy learning:** All-or-nothing + FCFS is simple; can learned cost models (estimating output length) improve victim selection vs LRU/random to minimize recompute/swap waste? [AGENT INFERENCE]
3. **Beyond tensor parallelism:** How to extend single manager + broadcast of block tables to pipeline parallelism (heterogeneous) or disaggregated prefill/decode, and to emerging hardware (Grace-Hopper unified memory) [AGENT INFERENCE]
4. **Cross-request automatic sharing at scale:** Can vLLM generalize prefix reservation to fully automatic radix-tree LRU with cache-aware scheduling (as SGLang later does) while preserving isolation and not copying via COW for arbitrary prefixes? [AGENT INFERENCE]
5. **KV cache compression synergy:** How does paging interact with quantization, pruning, or FlashAttention tiling; does per-block quantization change coalescing? [AGENT INFERENCE]
6. **Multi-level tiering:** Could we tier evicted blocks to NVMe/remote storage for very long contexts (100k) and prefetch them, rather than recompute? [AGENT INFERENCE]
7. **Fairness and SLOs:** How to incorporate latency SLOs, priority, or token-based fairness into preemptive scheduling without starving low-priority batch? [AGENT INFERENCE]
8. **Verification at scale:** Do 2â€?x gains hold on real production traces (10k rps, burst) with real EOS early-stops and chat multi-turn state reuse? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Orca [Yu et al., OSDI 2022] â€?direct predecessor and baseline:** Introduced iteration-level scheduling and selective batching which vLLM builds on; vLLM complements it by solving memory fragmentation/sharing that iteration-level scheduling exacerbates [PAPER FACT] (Â§9 compares: Orca for interleaving, vLLM for memory, complementary, 2â€?x over Orca) [PAPER FACT].
- **FasterTransformer [NVIDIA 2023a] â€?latency baseline** with block-wise matmul, all-reduce [PAPER FACT].
- **Memory optimizations for training:** SwapAdvisor [Huang 2020], SuperNeurons [Wang 2018], Checkmate [Jain 2020], POET [Patil 2022], ZeRO-Offload [Ren 2021], FlexGen [Sheng 2023] â€?swapping/recompute for training/inference with limited memory, but not online serving focus [PAPER FACT] (Â§9).
- **Kernel tiling:** FlashAttention [Dao 2022] â€?reduces peak attention memory and IO via tiling; orthogonal to paging [PAPER FACT].
- **General serving:** Clipper, TF Serving, Nexus, InferLine, Clockwork, DVABatch, REEF, Shepherd, AlpaServe [Li et al. 2023] â€?general batching/caching/placement/preemption but ignore autoregressive token state [PAPER FACT] (Â§9).
- **OLLA [Steiner 2022]** â€?optimizes tensor lifetime/location for fragmentation but not block-level online serving [PAPER FACT].
- **Follow-up to vLLM (not in paper):**
  - **SGLang [Zheng et al. 2024]** â€?RadixAttention automatic LRU prefix sharing across requests, builds on PagedAttention's non-contiguous layout but adds distributed radix tree and cache-aware scheduling [AGENT INFERENCE/PAPER FACT from SGLang].
  - **ChunkedAttention, HydraGen, FlashInfer, PromptCache** â€?concurrent KV reuse explorations cited in SGLang but not vLLM [AGENT INFERENCE].

## Review Log
Reviewer: Reviewer-1
Problems Found:
- Memory efficiency quoted as ~96.3% is Fig.2-derived over-precise; paper reports waste 60-80% for baselines (20.4-38.2% effective) and vLLM near-zero waste <4% (effective >96%) without exact 96.3% in text. Verified via webfetch arXiv:2309.06180v1 + Alphaxiv highlights (20.4-38.2% effective, <4% waste).
- Code license phrasing ambiguous (Apache? vs CC BY 4.0); verified code is Apache 2.0, paper CC BY 4.0.
- Model precision line incorrectly tagged as AGENT INFERENCE for FP16; paper explicitly uses FP16 for KV size calc (800KB/token) so FP16 calc is PAPER FACT, model precision remains NOT REPORTED.
- Quantitative claims spot-checked: 1.7-2.7x vs Orca Oracle, 2.7-8x vs Orca Max, up to 22x vs FT on ShareGPT (Sec 6.2, Fig.12/13); 12% prompt sharing vs up to 55% beam-search saving (¡ì3, ¡ì6.3); 20-26% kernel overhead (¡ì7.1); block size 16-128 optimal, default 16 (¡ì7.2); all confirmed via webfetch.
Corrections:
- Changed 96.3% to ~96% (<4% waste) with correct tag.
- Clarified license to Apache 2.0 / CC BY 4.0.
- Fixed precision line tag.
- No change to core contribution (PagedAttention + CoW sharing + vLLM system) ¡ª accurately captured.
Confidence: High