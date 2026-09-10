# Paper Metadata

- **Title:** Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve [PAPER FACT]
- **Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee [PAPER FACT] -- Georgia Institute of Technology (Agrawal* internship at MSR India, Tumanov) + Microsoft Research India (Kedia, Panwar, Mohan, Kwatra, Gulavani, Ramjee) [PAPER FACT]
- **Venue:** arXiv preprint arXiv:2403.02310v3 [cs.LG, cs.DC], submitted 4 Mar 2024 v1, last revised 17 Jun 2024 v3, 12 pages [PAPER FACT]; arXiv perpetual non-exclusive license [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2403.02310 / https://arxiv.org/abs/2403.02310 / HTML https://arxiv.org/html/2403.02310v3 [PAPER FACT]
- **Code:** https://github.com/microsoft/sarathi-serve [PAPER FACT] -- fork of vLLM, FlashAttention v2 + FlashInfer kernels, NCCL for comms [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2403.02310 + https://arxiv.org/html/2403.02310v3 (v3, 17 Jun 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM inference throughput can be increased via batching, but existing schedulers force a tradeoff between throughput and latency [PAPER FACT]. Each request has prefill (processes entire prompt in parallel, high latency but saturates GPU compute, compute-bound) and decode (one token per iteration, low latency but low compute utilization, memory-bound, benefits heavily from batching) [PAPER FACT]. Batching interleaves prefills and decodes; how to schedule them determines whether system optimizes for throughput or latency, with no current solution achieving both [PAPER FACT]. Under load, vLLM shows generation stalls lasting several seconds (Fig1a, Yi-34B on 2 A100, 128 arxiv-summarisation requests) and tail latency spikes with increasing load (Fig1b) [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLMs (GPT-3, LLaMA, Falcon, etc.) dominate GPU workload due to widespread chatbot/search/code-assistant usage; inference cost must be reduced while meeting interactive latency [PAPER FACT].
- Prefill latency is high but compute-efficient; decode latency is low but compute-inefficient -- batching effective for decodes, hence overall throughput [PAPER FACT].
- Production serving needs both high capacity (low cost) and low tail latency (TTFT initial responsiveness, TBT streaming fluidity); existing systems compromise one [PAPER FACT].
- Need scalable deployment across nodes: DGX A100 supports 8-GPU TP with NVLink, but cross-node lacks hyper-cluster bandwidth, so pipeline parallelism (PP) is required for commodity ethernet; PP introduces pipeline bubbles [PAPER FACT].
- Goal: support large batch sizes for throughput without violating TBT SLOs, and enable efficient PP without micro-batch imbalance [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Prefill-decode interference / generation stalls.** [PAPER FACT] Prefill-prioritizing iteration-level batching (Orca, vLLM) eagerly schedules prefills whenever memory available -- when requests C,D arrive while A,B decoding, their long prefills stall decodes of A,B; since prefill time depends on prompt length (100s-1000s tokens, median 1730 openchat_sharegpt4 vs 7059 arxiv_summarization Table2), iteration latency spikes to seconds -- high TBT tail latency (Fig2, Fig7) [PAPER FACT]. Orca hybrid batches still stall because long-prompt execution remains high; vLLM non-hybrid batches stall even more [PAPER FACT].
2. **Decode-prioritizing throughput collapse.** [PAPER FACT] Request-level batching (FasterTransformer) finishes all decodes before new prefills -- low TBT but severely reduced throughput: early-finishing requests leave batch with reduced size until last finishes, and pending prefills wait [PAPER FACT] (Alg1 vs Alg2 comparison).
3. **Batch size vs latency tradeoff with static policy.** [PAPER FACT] Lowering max batch size reduces TBT but hurts throughput; no single batch size navigates tradeoff well (Fig12: vLLM batch 32/64/128 have identical capped capacity) [PAPER FACT]. Prefill throughput saturates early (Fig3: prefill throughput flat after 1 request at 1024 tokens), decode throughput scales almost linearly with batch (Fig3) [PAPER FACT].
4. **Arithmetic intensity slack wasted.** [PAPER FACT] Linear ops dominate >80% time even at high seq len (Fig4 breakdown); decode batches are memory-bound (low arithmetic intensity), prefill compute-bound (Fig5-6). Decode iterations leave compute underutilized: adding 128 prefill tokens costs ~ same as 1 decode token (Fig4 note) [PAPER FACT]; naive coalescing (Decode+Full Prefill) increases TBT up to 28.3x vs decode-only (Fig9) [PAPER FACT].
5. **Pipeline bubbles in PP.** [PAPER FACT] Three types: PB1 varying prefill tokens across micro-batches, PB2 prefill vs decode time mismatch, PB3 varying KV-cache length affecting decode time (Fig8) [PAPER FACT]. Falcon-180B example: 4k prompt ~1150 ms vs decode-only batch32 ~200 ms -- ~950 ms bubble [PAPER FACT]; bubbles wasted GPU cycles and lower throughput, aggravated by longer prompts / larger batches [PAPER FACT].
6. **Tile-quantization and fixed overheads.** [PAPER FACT] Matmuls tiled to thread blocks; dimensions not divisible by tile size cause extraneous compute: e.g., 257 chunk 32% slower than 256 (Sec 4.3) [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Sarathi-Serve = chunked-prefills + stall-free batching + token-budget-controlled hybrid batches -> uniform compute iterations -> minimal stalls and pipeline bubbles [PAPER FACT].**

- **Chunked-prefills (Sec 4.1):** Split large prefill into near-equal compute-sized chunks, each computed over multiple iterations [PAPER FACT]. Insight: prefill saturates GPU at modest length ~512 tokens (Fig4), while median prompts are 1730-7059 -- opportunity to break into chunks still saturating compute but small enough to bound latency [PAPER FACT]. A 4k prompt becomes e.g., 8 chunks of 512 [PAPER FACT conceptually].
- **Stall-free scheduling / piggyback-with-budget (Sec 4.2, Alg3):** Iteration-level scheduler that in each iteration: 1) packs all running decodes (1 token each), 2) optionally adds one chunk from partially-done prefill, 3) admits new requests chunk-by-chunk until token budget tau reached, using get_next_chunk_size(R, tau, n_t) [PAPER FACT]. Token budget = maximum tokens per batch without violating TBT SLO, determined by profiling (Sec 4.3) [PAPER FACT]. Restricting tokens per iteration bounds latency and makes it near-independent of prompt length [PAPER FACT]. Unlike Orca/vLLM which stall decodes for prefills, Sarathi coalesces chunked prefills with decodes within budget -- decodes never experience generation stall [PAPER FACT].
- **Uniform hybrid batches -> PP efficiency:** Since each batch has ~ tau tokens, inter-batch variance collapses -> balanced micro-batches -> pipeline bubbles minimized (Fig8 Sarathi vs Orca) [PAPER FACT].
- **Token budget tuning (Sec 4.3):** Smaller budget lowers TBT but increases chunking overhead (lower GPU utilization, repeated KV-cache loads: N chunks -> KV of first chunk loaded N-1 times etc., O(N^2) reads) [PAPER FACT]; larger budget improves efficiency but risks SLO violation and more bubbles; tile-quantization favors multiples of tile size; Vidur simulator used to pick optimal tau per deployment (hardware, parallelism, SLO) [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Scheduler architecture:** Iteration-level loop modifying vLLM scheduler; implements stall-free Alg3 on centralized scheduler [PAPER FACT].
- **Chunked prefill kernels:** Support paged chunk prefill using FlashAttention v2 [Dao 2023] and FlashInfer [Ye et al. 2024]; currently uses FlashAttention backend for evaluated models [PAPER FACT]. Fuses reshape+attention for variable-length handling [AGENT INFERENCE].
- **Hybrid batch processing:** process_hybrid_batch(B) that processes mixed decode tokens + prefill chunks in single iteration; token budget enforcement via compute_token_budget(Tmax) from SLO (Sec 4.3) [PAPER FACT].
- **Token budget policy:** One-time profiling of batches with different token counts per model/GPU/parallelism; set tau to max tokens meeting TBT SLO; evaluation uses tau=512 for strict SLO and 2048 for relaxed (except LLaMA2-70B relaxed 1536 to reduce PP bubbles) [PAPER FACT]. Dynamically varying tau left for future work [PAPER FACT].
- **Pipeline parallelism support:** Extended vLLM to support PP (TP4-PP2 etc.) with NCCL point-to-point comms; uniform batches enable balanced PP scheduling without micro-batch fragmentation [PAPER FACT].
- **Telemetry and evaluation harness:** Extensive telemetry for TTFT/TBT per request, capacity via Poisson arrival sweeps [PAPER FACT].
- **Base system:** Fork of vLLM [Kwon et al.] retaining PagedAttention, adding chunked prefill, stall-free policy, PP, telemetry [PAPER FACT]; uses NCCL for TP/PP [PAPER FACT]; source at github.com/microsoft/sarathi-serve [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary: Capacity = maximum sustainable QPS while meeting SLOs and median scheduling delay <2s [PAPER FACT]** [PAPER FACT]; defined as max arrival rate where queuing delay not blowing up (limit 2 sec median scheduling delay) and where P99 TBT SLO met [PAPER FACT]. Higher capacity = lower cost per query [PAPER FACT].
- **Latency SLOs:**
  - **TTFT (time-to-first-token):** responsiveness, measured median (P50) [PAPER FACT].
  - **TBT (time-between-tokens):** streaming fluidity per token, measured P99 [PAPER FACT] (median for TTFT, P99 for TBT per Sec 5) [PAPER FACT].
  - **E2E:** rarely used except ablation Table4 [PAPER FACT].
  - **SLO definitions:** Strict vs relaxed derived as 5x and 25x decode iteration execution time (with prefill 4k and batch 32, no interference) per model; absolute thresholds Table3: Mistral-7B 0.1s/0.5s, Yi-34B 0.2s/1s, LLaMA2-70B 1s/5s, Falcon-180B 1s/5s (P99 TBT strict/relaxed) [PAPER FACT].
  - **Throughput-latency tradeoff curves:** capacity vs SLO (Fig12) [PAPER FACT].
- **Secondary:**
  - Prefill/decode throughput vs batch size (Fig3) and time breakdown linear/attention/others (Fig4) [PAPER FACT].
  - Arithmetic intensity and linear layer execution time vs tokens (Fig5-6) [PAPER FACT].
  - Incremental cost of hybrid batches (Fig9: TBT with Decode+Full vs Decode+Chunked) [PAPER FACT].
  - Pipeline bubble existence and capacity with PP (Fig8, Fig13) [PAPER FACT].
  - Chunked-prefill overhead normalized to no-chunking (Fig14) [PAPER FACT].
  - Ablation TTFT/TBT for hybrid-only vs chunked-only vs combined (Table4) [PAPER FACT].
- **Not elaborate:** Cost $/query, energy, KV memory fragmentation details beyond PagedAttention baseline [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **vLLM [Kwon et al. SOSP23, Kwon et al. 2309.06180]:** Iteration-level batching, prefill-prioritizing, PagedAttention, supports large batch sizes; evaluated with max batch size 32/64/128 sweeps for tradeoff; does not support inter-op/PP in vanilla (authors implement PP variant for comparison) [PAPER FACT]; supports hybrid? Actually paper says vLLM only supports either all-prefill or all-decode batches, not hybrid (Fig7) [PAPER FACT]; same parallelism as Sarathi for fair comparison (Mistral-7B TP1, Yi-34B TP2, LLaMA2-70B TP4-PP2, Falcon-180B TP4-PP2 or TP8) [PAPER FACT].
- **Orca [Yu et al. OSDI22]:** Iteration-level batching with hybrid batches (prefill+decode coalesced), FCFS, eager admission; limited to smaller batch due to lack of PagedAttention and large activation footprint (max seq len * batch) [PAPER FACT]; evaluated alongside vLLM in Fig10-11 [PAPER FACT].
- **FasterTransformer [NVIDIA]:** Request-level decode-prioritizing, discussed conceptually (Sec 2.5 Alg1) but not used in quantitative capacity evaluation (only in motivation) [PAPER FACT].
- **Ablation variants:** Hybrid-batching-only (Orca-like mixed without chunking) vs Chunked-prefills-only vs Sarathi-Serve combined (Table4) [PAPER FACT]; overhead sweep for chunk sizes 512/1024/2048 (Fig14) [PAPER FACT].
- **Disaggregated systems (discussed not benchmarked):** Splitwise [Patel 2311.18677], DistServe [Zhong 2401.09670], TetriInfer [Hu 2401.11181] classified as third category "disaggregated", mentioned as requiring KV migration and underutilizing prefill GPU memory, left for future quantitative comparison (Sec 6) [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models and GPU configs (Table1, Sec 5):**
  - Mistral-7B (GQA-SW) on 1x A100 80GB (80GB total) [PAPER FACT]
  - Yi-34B (GQA) on 2x A100 TP2 (160GB) [PAPER FACT]
  - LLaMA2-70B (GQA) on 8x A40 48GB (384GB, TP4-PP2) [PAPER FACT]
  - Falcon-180B (GQA) on 8x A100 80GB across 2 nodes (640GB, TP4-PP2 hybrid: 4-way TP intra-node + 2-way PP inter-node via 100Gbps ethernet) [PAPER FACT]
  - For PP comparison: Falcon-180B also TP8 vs hybrid [PAPER FACT].
- **Datasets (Table2, Sec 5):**
  - **openchat_sharegpt4** [Wang et al.]: user-shared ChatGPT-4 conversations, median prompt 1730, P90 5696, std 2088; median output 415, P90 834, std 101 [PAPER FACT]. Multi-round chat, high variance [PAPER FACT].
  - **arxiv_summarization** [Cohan et al.]: arXiv papers+abstracts, median prompt 7059, P90 12985, std 3638; median output 208, P90 371, std 265 [PAPER FACT]. Long prompts, low output variance, representative of M365 Copilot/Duet AI [PAPER FACT].
  - Both filtered to total length <=8192 and 16384 respectively [PAPER FACT].
  - Arrival process Poisson with varying QPS, outlier removal, sustainable load definition median scheduling delay <2s [PAPER FACT].
- **Generation settings:** [NOT REPORTED] exact decode sampling method, output length distribution derived from datasets; no beam search evaluated [NOT REPORTED].
- **Trace scale:** 128 requests example Fig1, capacity sweeps use Poisson-generated traces from dataset length distributions [PAPER FACT]; long prompts stress TTFT and TBT [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Platform:** Azure NC96ads v4 VMs, each with 4x NVIDIA 80GB A100, pairwise NVLink [PAPER FACT] (for Mistral-7B, Yi-34B, Falcon-180B cross-node); interconnect 100 Gbps ethernet between nodes [PAPER FACT].
- **Alternate hardware:** LLaMA2-70B on 8x NVIDIA 48GB A40 GPUs, pairwise connected [PAPER FACT].
- **Interconnect specifics:**
  - Intra-node: NVLink (pairwise) [PAPER FACT]
  - Inter-node (Falcon PP): 100 Gbps ethernet (Sec 5, Sec 5.3) [PAPER FACT]; TP cross-node incurs high all-reduce overhead (~2x higher median TBT, Fig13a) [PAPER FACT].
- **Precision:** [NOT REPORTED] but FP16 implied via model sizes / batch sizing; not explicitly stated per experiment [NOT REPORTED].
- **Software/stack:** CUDA 12.1 [PAPER FACT per artifact], vLLM base, FlashAttention v2/FlashInfer kernels, NCCL for TP/PP [PAPER FACT]; Vidur simulator for token budget selection [PAPER FACT]; artifact tested on A100/A40 [PAPER FACT].
- **Not detailed:** CPU, DRAM, OS, exact NCCL version beyond citation [NOT REPORTED].

## 10 Main Results [PAPER FACT]

All numbers from Abstract, Sec 5, Figs 10-14, Table4.

- **Capacity evaluation vs Orca/vLLM (Fig10-11):**
  - **Single-GPU Mistral-7B:** Up to **2.6x higher capacity vs vLLM** [PAPER FACT] (abstract: 2.6x for Mistral-7B single A100; Sec 5.1 notes up to 4.0x vs Orca and 3.7x vs vLLM for Yi-34B openchat) [PAPER FACT].
  - **Yi-34B on 2x A100 TP2:** **Up to 3.7x vs vLLM and 4.0x vs Orca under strict SLO** (openchat_sharegpt4) [PAPER FACT]; abstract summarises 3.7x for Yi-34B on 2 A100 [PAPER FACT]; Fig10 shows capacity QPS bars for openchat and arxiv under strict/relaxed: Sarathi consistently top [PAPER FACT].
  - **LLaMA2-70B TP4-PP2 on 8x A40:** **Up to 6.3x vs Orca and 4.3x vs vLLM** (openchat_sharegpt4) due to reduced pipeline bubbles [PAPER FACT] (text Sec 5.1) [PAPER FACT].
  - **Falcon-180B TP4-PP2 on 8x A100 x2 nodes:** Up to **5.6x gain in end-to-end serving capacity** with pipeline parallelism (abstract, Sec 5.3) [PAPER FACT]; specifically Fig13b shows 4.3x vs vLLM TP-only and 3.6x vs vLLM hybrid under strict SLO, 1.48x under relaxed [PAPER FACT].
  - **Relaxed vs Strict:** Orca/vLLM capacity grows with relaxed SLO (since they violate P99 early), Sarathi advantage shrinks but remains >1.65x (Fig11, Sec 5.1) [PAPER FACT]; Sarathi uses smaller tau=512 for strict, larger tau=2048 (or 1536 for LLaMA2-70B relaxed) [PAPER FACT].
  - **Why vLLM > Orca in relaxed?** vLLM supports larger batch via PagedAttention; Orca batches prompts together (max seq len * batch vs max seq len) leading to higher tail [PAPER FACT].

- **Throughput-latency tradeoff (Fig12, Sec 5.2):** Sweeping P99 TBT SLO for Mistral-7B/Yi-34B on openchat, vLLM capacity capped and identical for max batch 32/64/128 (cannot leverage large batch due to tradeoff) [PAPER FACT]; Sarathi with tau=512 achieves **3.5x higher capacity vs vLLM at strict 100ms Mistral-7B**, and **1.65x at relaxed 1s Yi-34B with tau=2048** [PAPER FACT].

- **Pipeline viability (Fig13, Sec 5.3 Falcon-180B):**
  - Median TBT for decode-only batch: TP8 ~2x higher than TP4-PP2 (cross-node TP overhead) [PAPER FACT].
  - Capacity: Sarathi PP gives **1.48x vs vLLM hybrid (relaxed)** and **3.6x vs vLLM hybrid (strict)** and **4.3x vs vLLM TP-only (strict)** [PAPER FACT]; shows chunked uniform batches reduce bubbles [PAPER FACT].

- **Ablations (Sec 5.4):**
  - **Chunk overhead (Fig14, Yi-34B TP2):** At small chunk 512, **~25% overhead** vs no-chunking (bar near 1.25); larger tau=2048 negligible overhead (~1.0) [PAPER FACT]; overhead decreases with prompt length? Bars show slight variation [PAPER FACT].
  - **Hybrid vs chunked in isolation (Table4, Yi-34B 2xA100, tau=1024, 128 requests):**
    - Hybrid-batching-only: P50 TTFT 0.53s openchat / 3.78s arxiv, P99 TBT 0.68s /1.38s [PAPER FACT].
    - Chunked-only: 1.04s/5.38s TTFT, 0.17s/0.20s TBT [PAPER FACT].
    - **Combined Sarathi:** **0.76s/3.90s TTFT, 0.14s/0.17s TBT** -- best on both dimensions: lowers both TTFT vs chunked-only and TBT vs hybrid-only [PAPER FACT].
  - **Incremental cost (Fig9):** Decode+Full Prefill (Orca hybrid) raises TBT up to 28.3x vs decode-only; Decode+Chunked (Sarathi) tightly bounds increase, impact reduces with higher decode batch size and context length [PAPER FACT].

- **Overall claim:** Up to **order-of-magnitude (10x) improvement** across models/hardware under tail latency constraints (Conclusion) [PAPER FACT]; specific headline: 2.6x Mistral-7B 1xA100, 3.7x Yi-34B 2xA100, 5.6x Falcon-180B PP [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder-only models with prefill (parallel, compute-bound) vs decode (sequential, memory-bound) split; KV-cache per token per layer [PAPER FACT].
- Token-level batching possible: linear ops can process hybrid prefill+decode tokens with similar cost if total tokens ~ tau; attention dominates quadratically only at extreme lengths but linear >80% time [PAPER FACT].
- Batch execution time ~ max(Tmath, Tmem); decode memory-bound, prefill compute-bound; arithmetic intensity quantifies balance [PAPER FACT].
- PagedAttention available for large batches (vLLM base) [PAPER FACT].
- Poisson arrival for capacity sweeps, trace length distributions fixed per dataset; prompt lengths 100s-1000s, output lengths variable [PAPER FACT].
- Pipeline parallelism with micro-batching can be balanced if batch token counts uniform [PAPER FACT].
- SLO attainment requires median scheduling delay <2s and P99 TBT under threshold [PAPER FACT].
- TTFT tolerated via median, TBT via P99 (SLOs 5x/25x baseline) [PAPER FACT]; lower TBT prioritized over TTFT for streaming [PAPER FACT].
- Model fits with TP within node or TP+PP across nodes; NVLink intra-node, ethernet inter-node [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Discussed in Sec 6 Related Work and Sec 7 Conclusion / Sec 4.3:
  - **Not compared quantitatively to disaggregated approaches:** Splitwise, DistServe, TetriInfer eliminate interference by separating phases but require KV migration and underutilize prefill GPU memory; authors leave quantitative comparison for future work [PAPER FACT].
  - **Prefill efficiency tradeoff:** Chunked prefills are somewhat slower than full prefills (KV cache reload overhead, extra kernel launch); disaggregated can execute prefills with max efficiency and better TTFT; Sarathi sacrifices some prefill speed for TBT guarantees [PAPER FACT].
  - **Tile-quantization and overhead sensitivity:** Chunk size must consider tile size (e.g., 256 multiple) to avoid 32% penalty; very small tau causes overhead (Fig14) [PAPER FACT].
  - **Token budget requires profiling:** Optimal tau depends on TBT SLO, parallelism, hardware; selected via Vidur simulator; no dynamic adaptation to workload yet (future work: dynamically varying tau) [PAPER FACT].
  - **Prototype not full-featured:** Artifact is lightweight research prototype fork of vLLM, lacks complete feature parity with open-source vLLM, intended for faster research iterations [PAPER FACT] (Artifact Appendix).
  - **Fairness/complementary schedulers:** Fairness among tenants (e.g., Sheng et al.) and preemption (FastServe) are complementary, not integrated; could benefit from Sarathis lower interference [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Static token budget per model/hardware:** 512 for strict, 2048 for relaxed is coarse; optimal tau varies with prompt length distribution (openchat vs arxiv) and arrival rate, but paper uses only two values (three for LLaMA2) [AGENT INFERENCE]; no automatic per-request tau or online controller.
- **No KV-cache reuse/prefix sharing:** No prefix cache, RadixAttention-like deduplication, or prompt caching; repeated system prompts would recompute chunked prefills repeatedly [AGENT INFERENCE].
- **TTFT may increase:** Chunked-only shows TTFT 1.04s vs hybrid-only 0.53s (openchat); Sarathi combined 0.76s still higher than hybrid-only -- under TTFT-critical SLO, Sarathi may be slightly slower despite TBT win [AGENT INFERENCE].
- **Evaluation scale limited to <=8 GPUs, <=2 nodes:** No multi-tenant, multi-replica load balancer beyond 8 GPUs; no measurement of scheduler overhead at high QPS (100+ RPS) [AGENT INFERENCE].
- **No cost/energy or heterogeneous hardware analysis:** Unlike Splitwise, no Perf/$ or Perf/W analysis; no comparison of A100 vs A40 efficiency [AGENT INFERENCE].
- **Workload coverage:** Only two datasets (sharegpt, arxiv) filtered to 8k/16k, no coding (HumanEval), long-context (LongBench 100k) or MoE models evaluated [AGENT INFERENCE].
- **Reliance on FlashAttention specifics:** Implementation tied to FlashAttention v2 tile size; portability to other attention kernels (e.g., MLA, MQA variants) not discussed [AGENT INFERENCE].
- **No integration with memory optimizations beyond PagedAttention:** No interaction with quantization, sparsity, or KV eviction (H2O, Scissorhands) [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Dynamic token budget:** Can tau be auto-tuned online based on queue length, SLO slack, and prompt length to simultaneously minimize TBT tail and maximize goodput without offline profiling? [AGENT INFERENCE]
2. **Disaggregation vs chunked hybrid:** In which regime does disaggregation (Splitwise/DistServe) strictly dominate Sarathis hybrid approach when KV transfer bandwidth is abundant vs constrained? What hybrid-disaggregated hybrid is optimal? [AGENT INFERENCE]
3. **Prefix-aware chunking:** How to combine chunked prefills with RadixAttention/prompt cache to avoid recomputing shared prefixes chunk-by-chunk, especially for multi-turn chat? [AGENT INFERENCE]
4. **Heterogeneous clusters and power:** Can chunked scheduling be extended to heterogeneous prompt/decode hardware or power-capped GPUs while keeping uniform batch property? [AGENT INFERENCE]
5. **Long context (100k+):** Does repeated KV reload O(N^2) overhead become prohibitive at 100k prompts; would sequence parallelism or ring attention complement chunking? [AGENT INFERENCE]
6. **Fairness and preemption:** How to integrate fairness (Sheng et al.) or preemptive SJF (FastServe) with stall-free guarantee without reintroducing stalls? [AGENT INFERENCE]
7. **PP scaling:** Can uniform-batch idea extend to larger PP degrees (e.g., 8-stage) or to decoder-encoder models where prefill/decode asymmetry differs? [AGENT INFERENCE]
8. **TTFT SLOs:** When TTFT and TBT SLOs are both strict, is there a Pareto-optimal tau or should scheduler prioritize differently (e.g., TTFT-aware chunk priority)? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Orca [Yu et al., OSDI22]:** Iteration-level batching, predecessor; Sarathi augments it with chunked prefills and stall-free hybrid scheduling to bound TBT [PAPER FACT].
- **vLLM / PagedAttention [Kwon et al., SOSP23]:** Memory manager enabling large batches, base codebase; Sarathi uses it and shows Orca fails without it due to activation memory [PAPER FACT].
- **FasterTransformer [NVIDIA]:** Request-level decode-prioritizing baseline, high TBT-optimal but low throughput [PAPER FACT].
- **Splitwise [Patel et al. arXiv 2311.18677] and DistServe [Zhong et al. arXiv 2401.09670] and TetriInfer [Hu et al. 2401.11181]:** Disaggregated prefill-decode on separate replicas to eliminate interference, classified as third category; authors argue disaggregation transfers KV and underutilizes prompt GPU memory; complementary but not directly evaluated [PAPER FACT].
- **SARATHI [Agrawal et al. 2308.16369, precursor]:** Original chunked prefill with piggyback decodes idea; Sarathi-Serve is its serving system instantiation [PAPER FACT].
- **Vidur [Agrawal et al. MLSys2024]:** Large-scale LLM inference profiler/simulator used to pick token budget maximizing capacity [PAPER FACT].
- **FlexGen [Sheng 2023], FastServe [Wu 2023], SGLang/RadixAttention [Zheng 2024], PromptCache, AttentionStore, APIServe [Abhyankar 2402.01869]:** Memory/scheduling optimizations orthogonal; APIServe adopted chunked prefill for multi-turn recomputation [PAPER FACT] (Sec 6).
- **FlashAttention v2 [Dao 2023], FlashInfer [Ye 2024], Megatron-LM [Shoeybi 2019], GPipe [Huang 2019]:** Kernels and parallelism foundations [PAPER FACT].
- **Follow-up to Sarathi (not in paper):** Mooncake [Qin 2407.00079] builds on chunked-pipeline + KV-centric disaggregation; discussion notes Sarathi inspiration [AGENT INFERENCE].


## Review Log
Reviewer: Reviewer-1
Problems Found:
- Token budget tuning correctly notes tau=512 strict, 2048 relaxed, 1536 for LLaMA2-70B relaxed ¡ª verified ¡ì4.3 and Fig10-12 notes; note previously correct but clarified capacity definition (median scheduling delay <2s) which is part of capacity metric per ¡ì5.
- Throughput gains spot-checked: Mistral-7B up to 2.6x vs vLLM (single A100), Yi-34B up to 3.7x vs vLLM /4.0x vs Orca (2xA100 TP2), LLaMA2-70B TP4-PP2 up to 6.3x vs Orca /4.3x vs vLLM, Falcon-180B TP4-PP2 up to 5.6x (1.48x relaxed/3.6x strict vs vLLM hybrid, 4.3x vs TP-only) ¡ª all numbers trace to Abstract, ¡ì5.1, ¡ì5.3 Fig10/13.
- Chunk overhead ~25% at 512 vs negligible at 2048 (Fig14) and ablation Table4 hybrid-only 0.53s/0.68s vs chunked-only 1.04s/0.17s vs combined 0.76s/0.14s verified.
- Hardware: Azure NC96ads v4 4xA100 80GB pairwise NVLink + 100Gbps ethernet inter-node, alternate A40 for LLaMA2-70B; correctly marked CPU/DRAM/NCCL NOT REPORTED.
- Baseline: vLLM with max batch 32/64/128 sweep and Orca hybrid correctly captured; disaggregated Splitwise/DistServe mentioned as concurrent not benchmarked ¡ª correctly not claimed as measured.
Corrections:
- Reworded tile-quantization 257 vs 256 example for clarity.
- Explicitly added median scheduling delay <2s to capacity definition.
- No core contribution misinterpretation (chunked-prefills + stall-free token-budget hybrid batches -> uniform compute, pipeline bubble reduction correctly described).
Confidence: High