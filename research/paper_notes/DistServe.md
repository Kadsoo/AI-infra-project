# Paper Metadata

- **Title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving [PAPER FACT]
- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang [PAPER FACT] -- Peking University, StepFun, UC San Diego [PAPER FACT]
- **Venue:** OSDI 2024 (arXiv preprint arXiv:2401.09670v3 [cs.DC], submitted 18 Jan 2024 v1, last revised 6 Jun 2024 v3) [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2401.09670 / https://arxiv.org/abs/2401.09670 / HTML https://arxiv.org/html/2401.09670v3 [PAPER FACT]
- **Code:** https://github.com/LLMServe/DistServe [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2401.09670 + https://arxiv.org/html/2401.09670v3 (v3, 06 Jun 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Existing LLM serving co-locates prefill and decoding on same GPUs and batches them together to maximize overall throughput (tokens/s) [PAPER FACT]. However this leads to strong prefill-decoding interference and couples resource allocation/parallelism for both phases [PAPER FACT]. LLM apps emphasize per-phase latency: TTFT for prefill and TPOT (time per output token, aka TBT) for decoding, with varying strictness (e.g., chatbots need low TTFT, summarization needs low TPOT) [PAPER FACT]. Under stringent TTFT+TPOT SLOs, colocated systems must prioritize one over the other or over-provision GPUs, failing to maximize per-GPU goodput (max rps meeting >90% SLO attainment) [PAPER FACT]. Fig1 (OPT-13B, 512 in/64 out on 1x A100) shows colocated vLLM achieves only ~1.6 rps per-GPU, while prefill-only achieves 5.6 rps and decode-only 10 rps -- ideal disaggregated 2P+1D could achieve 3.3 rps per-GPU, 2.1x higher [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLM services differentiate TTFT vs TPOT: TTFT = prefill duration, TPOT = avg decode per token; overall latency = TTFT + TPOT * generated tokens [PAPER FACT]; optimizing per-GPU goodput under SLO attainment (e.g., 90% requests meet both) directly reduces cost per query [PAPER FACT].
- Prefill is compute-bound (e.g., 512 tokens makes A100 near compute-bound for 13B) and superlinear with token count; decode is memory-bandwidth-bound with similar I/O as prefill despite 1 token, hence low utilization [PAPER FACT]; Fig3 shows prefill throughput flat beyond batch 1 at 512 tokens, decode throughput keeps rising [PAPER FACT].
- Growing need to host larger models (66B, 175B) that exceed single GPU and require model parallelism; parallelism choices affect TTFT vs TPOT differently [PAPER FACT].
- Service providers must meet diverse SLOs without over-provisioning; interference forces wasted GPUs [PAPER FACT].
- Modern clusters have high-bandwidth interconnects (NVLink 600 GB/s intra-node, InfiniBand 800 Gbps inter-node in many clusters, 25-50 GB/s per GPU pair) making KV-cache transfer feasible [PAPER FACT] (Sec 3.3).
- Opportunity: disaggregate phases onto separate GPUs/instances, eliminate interference, scale each phase independently with tailored parallelism and replication [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Prefill-decoding interference (strong).** [PAPER FACT] Adding 1 prefill to decode batch significantly slows both: decodes wait for long prefill (TPOT elongation worsens with 1024 vs 128 length, Fig2), and prefills slow when GPU already saturated by decodes (Fig2 blue curves) [PAPER FACT]. Sequential scheduling still causes queuing delay: decodes wait for ongoing prefills and vice versa; priority scheduling fails one phases SLO [PAPER FACT].
2. **Chunked-prefill piggyback insufficient.** [PAPER FACT] Sarathi-style chunked-prefill with piggyback alleviates but does not eliminate slowdown, trades TTFT for TPOT, adds overhead: if chunk size << saturation point, prefill longer due to contention; if near saturation, few decode slots remain; also causes O(N^2) KV loads (N chunks -> N+(N-1)+... = O(N^2) vs O(N) non-chunked) increasing HBM reads, worsening with long context [PAPER FACT] (Sec 2.3).
3. **Resource and parallelism coupling.** [PAPER FACT] Colocated forces shared parallelism plan; but prefill prefers intra-op at low rate (reduces execution time, Fig4a) vs inter-op at high rate (reduces queuing), while decode batch size limited by memory and prefers intra-op for stringent TPOT vs inter-op for throughput (Fig5) [PAPER FACT]; optimal config differs per phase, colocated must pick compromise for more demanding SLO, causing over-provisioning [PAPER FACT].
4. **TTFT vs TPOT tradeoff forcing over-provision.** [PAPER FACT] To meet both SLOs, systems over-provision GPUs (e.g., Fig1 requires separate optimization) [PAPER FACT].
5. **Communication overhead if disaggregated naively.** [PAPER FACT] KV cache size e.g., 1.13 GB per 512-token request on OPT-66B; at 10 rps need 90 Gbps inter-node bandwidth to hide; cross-node bandwidth limited to 25 Gbps in testbed, requiring careful placement to use NVLink (600 GB/s) intra-node [PAPER FACT].
6. **Variable prompt length causing pipeline bubbles.** [PAPER FACT] Non-uniform lengths cause inter-op pipeline bubbles for prefill instances; M/D/1 model assumptions break [PAPER FACT] (Sec 3.3).
7. **Queuing theory complexity:** No simple analytic SLO attainment due to diverse lengths and Poisson arrivals; profiling on testbed time-prohibitive, need simulator [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Disaggregate prefill and decoding onto different GPUs (prefill instances vs decoding instances), each with its own resources, parallelism strategy, and scaling, plus bandwidth-aware placement to minimize KV transfer cost, co-optimized for per-GPU goodput under TTFT/TPOT SLOs [PAPER FACT].**

- **Disaggregation eliminates interference:** Prefill instances only do prefill to generate first token then send KV caches + token to decode instance; decode instances batch only decodes, achieving higher utilization via larger batches (e.g., multiple prefills per decode) [PAPER FACT]. Naturally resolves interference and enables independent optimization [PAPER FACT].
- **Co-optimized resource allocation + parallelism per phase:** Given model, workload distribution, TTFT/TPOT, SLO attainment target (90%), DistServe searches parallelism configs (inter_op, intra_op) per phase to maximize per-GPU goodput (goodput / num_gpus) via simulation + binary search, then replicates instances to meet total rate R: n = ceil(R / goodput_p), m = ceil(R / goodput_d) [PAPER FACT] (Alg1 for high node-affinity, Alg2 for low affinity) [PAPER FACT].
- **Queuing analysis guides intuition:** Models prefill as M/D/1 queue: Avg_TTFT = D + R*D^2 / (2*(1 - R*D)) [Eq1] [PAPER FACT]; with 2-way inter-op: Avg_TTFT_inter = D + R*D^2 / (4*(2 - R*D)) [Eq2]; with intra-op speedup K (1<K<2): Avg_TTFT_intra = D/K + R*D^2 / (2*K*(K - R*D)) [Eq3] [PAPER FACT]; at low rate intra-op wins (execution time dominates), at high rate inter-op wins (queuing dominates), and stringent SLO favors intra-op [PAPER FACT].
- **Bandwidth-aware placement:** Two algorithms: High node-affinity (InfiniBand negligible) -- enumerate all configs independently then replicate; Low node-affinity (25 Gbps testbed, common) -- constrain prefill and decode stages of same index to co-locate on same node to use NVLink for KV transfer, reducing transmission to <0.1% total latency [PAPER FACT] (Sec 4.1-4.2, Sec 6.3) [PAPER FACT].
- **Online scheduling optimizations:** FCFS with adaptive batching (prefill batch only if length < Lm saturation threshold, else sequential; decode batch as large as memory allows) plus handling of pipeline bubbles for variable lengths [PAPER FACT] (Sec 3.1-3.2, Sec 4.3).

## 5 System Changes [PAPER FACT]

- **Architecture:** Disaggregated instances: prefill instance = unit managing one complete model copy (may span multiple GPUs via inter/intra-op); decoding instance similarly; each type scales independently [PAPER FACT]; orchestration layer on top of inference engine [PAPER FACT].
- **Placement algorithm (Alg1 high-affinity, Alg2 low-affinity):** Enumerates intra_op in {1..M} and inter_op in {1.. N*M / intra_op}, checks GPU memory capacity G.size/(inter*intra) < C (can fit), parallels model, runs simu_prefill / simu_decode simulators to binary-search max goodput meeting SLO attainment, keeps config with max per-GPU goodput [PAPER FACT]; then replicates to meet rate R [PAPER FACT]; uses workload traces resampled from history, fitting distribution over hours/days (assumes predictability) [PAPER FACT].
- **Simulator:** Event-driven discrete simulator using latency model (Appendix A): prefill latency = C1*(4*t*h^2+2*t*h*m)+C2*3*h*t^2/b +C3 (GEMM compute + FlashAttention memory) [PAPER FACT] (compute-bound GEMMs + memory-bound FlashAttention with block size b) [PAPER FACT]; decode latency = C4*(4*h^2+2*h*m) + C5*3*h*t (memory-bound) [PAPER FACT]; C1-5 via profiling/interpolation, error <2% vs real system Table2 [PAPER FACT]; used to estimate SLO attainment without testbed runs [PAPER FACT].
- **Communication:** KV cache transfer between phases; optimized to use NVLink intra-node when co-located stages, otherwise InfiniBand; transmission time <30ms for 95% requests even for 175B (Fig10) [PAPER FACT].
- **Online scheduling (Sec 4.3):** Prefill: FCFS, batch if total tokens < Lm else sequential; Decode: FCFS, batch up to memory; replication load balancing; scheduling to minimize pipeline bubbles for variable lengths [PAPER FACT].
- **Implementation:** Built as orchestration layer; supports OPT models with PagedAttention, GQA/MQA aware (but evaluated on MHA for pressure), FP16 [PAPER FACT]; tested on 32 GPUs (4 nodes x 8 A100-80GB, NVLink, 25 Gbps cross-node, plus high-affinity sim) [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary: Per-GPU goodput = (max RPS meeting 90% TTFT+TPOT SLO attainment) / num_gpus [PAPER FACT] (e.g., 90% or 99% of requests meet both TTFT and TPOT SLOs) divided by number of GPUs [PAPER FACT]; also minimal SLO scale achievable at fixed rate [PAPER FACT]; SLO attainment curves vs rate and vs SLO scale (SLO scale multiplies both TTFT and TPOT, smaller = more stringent) shown in Fig8-9 [PAPER FACT].
- **Latency SLOs:**
  - **TTFT:** responsiveness for prefill; per-request, e.g., 0.25s for OPT-13B ShareGPT, 2.5s-15s for LongBench etc. Table1 [PAPER FACT].
  - **TPOT (TPOT/TBT):** per output token time, e.g., 0.1s for OPT-13B, 0.15-0.2s for larger models Table1 [PAPER FACT].
  - **Definitions vary per application/dataset/model:** Table1 lists 6 workloads: chatbot OPT-13B 0.25s/0.1s, OPT-66B 2.5s/0.15s, OPT-175B 4.0s/0.2s on ShareGPT; code completion OPT-66B 0.125s/0.2s on HumanEval; summarization OPT-66B 15s/0.15s on LongBench [PAPER FACT].
  - SLO attainment target 90% (main) and 99% (Appendix C) [PAPER FACT].
- **Secondary:**
  - TTFT and TPOT CDFs, P90 curves (Fig1, Fig8) [PAPER FACT].
  - Latency breakdown: prefill queuing, prefill execution, transmission, decode queuing, decode execution proportions (Fig10a) [PAPER FACT].
  - KV transmission CDF time (<30ms 95%, Fig10b) [PAPER FACT].
  - Throughput for prefill/decode vs batch size/input length (Fig3) [PAPER FACT].
  - Algorithm running time vs GPUs (Fig12, minutes, parallelizable) [PAPER FACT].
  - Simulator accuracy vs real system error <2% (Table2) [PAPER FACT].
  - Ablation placement High vs Low affinity, vLLM vs vLLM++ (Fig11) [PAPER FACT].
- **Not elaborate:** Cost $ per query, energy per token, dollar cost of 2x model weight copies (prefill+decode each hold weights) [NOT REPORTED] exact cost numbers.

## 7 Baselines [PAPER FACT]

- **vLLM [Kwon et al. SOSP23, PagedAttention]:** Representative colocated system with continuous batching and paged KV management; maximizes overall throughput but suffers interference; supports only intra-op parallelism; tested with intra_op =1,4,8 for 13B/66B/175B respectively following original paper [PAPER FACT].
- **vLLM++ (ablation):** Enumerates different parallelism strategies for vLLM and picks best per-GPU goodput (to isolate disaggregation benefit vs parallelism tuning) [PAPER FACT].
- **DeepSpeed-MII [Microsoft, chunked-prefill]:** Supports chunked-prefill decomposing long prompts to fill token budget, piggybacks decodes; mitigates but cannot eliminate interference, slower than full prefill so trades TTFT [PAPER FACT]; set intra_op same as vLLM; cannot serve OPT-175B with intra_op 8 due to vocab_size/intra_op multiple-of-8 constraint and OOM with 4 [PAPER FACT].
- **DistServe variants:** DistServe-Low (Alg2 constrained to NVLink co-location, used on physical 25 Gbps cluster) vs DistServe-High (Alg1 no constraint, assumes high cross-node bandwidth, simulated) [PAPER FACT].
- **Other colocated systems mentioned not directly benchmarked:** Orca, Sarathi, FasterTransformer, AlpaServe etc. discussed in Related Work [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models:** OPT series [Zhang et al. 2022] chosen as representative, widely used in academia/industry; OPT uses classic MHA (multi-head attention) to pressure transmission (newer GQA/MQA would lower KV size and benefit DistServe more) [PAPER FACT]; tested:
  - OPT-13B (13B params) [PAPER FACT]
  - OPT-66B (66B) [PAPER FACT]
  - OPT-175B (175B, similar to GPT-3) [PAPER FACT]
  - FP16 precision [PAPER FACT].
- **Datasets / Applications (Table1, Fig7):**
  - **ShareGPT [ShareGPT teams 2023]:** User-shared ChatGPT conversations, chatbot app; input/output length distributions Fig7a (median not numerically labeled but shows short-medium inputs, outputs variable) [PAPER FACT]; arrival Poisson, used for chatbot on all three OPT sizes [PAPER FACT].
  - **HumanEval [Chen et al. 2021]:** 164 programming problems, code completion task, personal real-time assistant, stringent TTFT 0.125s and TPOT 0.2s for OPT-66B, Fig7b [PAPER FACT].
  - **LongBench [Bai et al. 2023]:** Summarization task, long articles, capped input lengths to 2048 due to OPT positional embedding limit, Fig7c, loose TTFT 15s but stringent TPOT 0.15s for OPT-66B [PAPER FACT].
  - Input lengths sampled from datasets; output lengths also dataset-driven; timestamps Poisson with varying rates [PAPER FACT]; due to space, chatbot on all 3 models, other two tasks only on 66B (largest open-source) [PAPER FACT].
- **Workload characteristics:** Chatbot medium inputs, code completion short, summarization very long inputs (Fig7) [PAPER FACT]; Poisson arrival for goodput sweeps [PAPER FACT].
- **Scale:** Sustained rate sweeps; e.g., Fig8 chatbot tests up to maybe 4 rps per GPU for 13B? Exact max rates not labeled but curves show attainment drop from 100% to ~20% as rate increases [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Cluster testbed (Sec 6.1):** 4 nodes x 8 NVIDIA SXM A100-80GB GPUs = 32 GPUs total [PAPER FACT]; each node NVLink interconnected [PAPER FACT]; cross-node bandwidth 25 Gbps (limited) [PAPER FACT]; due to limited bandwidth, DistServe uses low-affinity Alg2 in most physical experiments, high-affinity simulated [PAPER FACT].
- **High-affinity simulated cluster:** Assumes InfiniBand 800 Gbps or 600 GB/s NVLink negligible transmission, used for ablation DistServe-High [PAPER FACT] (Sec 4, Sec 6.4) [PAPER FACT].
- **Per-model GPU allocation examples (Appendix B Table3):**
  - OPT-13B ShareGPT: Prefill TP2 PP1, Decode TP1 PP1 [PAPER FACT]
  - OPT-66B ShareGPT/HumanEval/LongBench: Prefill TP4 PP1, Decode TP2 PP2 [PAPER FACT]
  - OPT-175B ShareGPT: Prefill inter3 intra3 (9 GPUs), Decode inter3 intra4 (12 GPUs) -- text says prefill inter3 intra3 and decode inter3 intra4 under Alg2 with 32 GPU cluster [PAPER FACT].
- **Precision:** FP16 [PAPER FACT].
- **Simulator runs:** AWS m5d.metal 96 cores, algorithm runtime minutes, highly parallelizable [PAPER FACT] (Fig12) [PAPER FACT].
- **Not detailed:** CPU, DRAM, OS, exact CUDA/NCCL versions [NOT REPORTED].

## 10 Main Results [PAPER FACT]

All numbers from Sec 6, Figs 8-14, Tables 2-3, Abstract.

- **Chatbot ShareGPT:**
  - **OPT-13B/66B/175B:** DistServe sustains **2.0x-4.6x higher request rate** vs vLLM while maintaining >90% SLO attainment [PAPER FACT] (Sec 6.2 first paragraph, Fig8 row1). Specific: vs DeepSpeed-MII 1.6x-7.4x higher rate [PAPER FACT].
  - **SLO tightness:** At fixed rate, DistServe can achieve **1.8x-3.2x more stringent SLO scale** vs vLLM and **1.7x-1.8x** vs DeepSpeed-MII (Fig8 row2, varying SLO scale linearly) [PAPER FACT].
  - **Example 175B placement:** prefill inter3 intra3, decode inter3 intra4 effectively balances load, proving algorithm effectiveness [PAPER FACT].
  - **Overall headline:** Abstract claims **up to 7.4x more requests or 12.6x tighter SLO** vs state-of-art [PAPER FACT]; Sec 6.2 confirms ranges across workloads [PAPER FACT].

- **Code completion HumanEval OPT-66B (Fig9a):**
  - **5.7x higher request rate and 1.4x more stringent SLO** vs vLLM [PAPER FACT].
  - **1.6x higher rate and 1.4x tighter SLO** vs DeepSpeed-MII [PAPER FACT]; constrained by TTFT due to stringent 0.125s requirement; DistServe wins via eliminating decode interference and increasing intra-op for prefill via search [PAPER FACT].

- **Summarization LongBench OPT-66B (Fig9b):**
  - **4.3x higher rate and 12.6x tighter SLO** vs vLLM [PAPER FACT]; abstracts 12.6x highlights this case [PAPER FACT].
  - **1.8x higher rate and 2.6x tighter SLO** vs DeepSpeed-MII [PAPER FACT]; long inputs pressure prefill but TPOT critical, vLLM fails TPOT due to long prefill slowdown [PAPER FACT].

- **99% attainment (Appendix C Fig13-14):** Under more stringent 99% goal, DistServe still sustains **3x-8x higher rate and 1.24x-6.67x tighter SLO** vs vLLM, and **1.32x-8x rate, 1.20x-1.58x SLO** vs DeepSpeed-MII [PAPER FACT].

- **Latency breakdown (Fig10, OPT-175B ShareGPT):**
  - KV transmission accounts for **<0.1% total latency** even for 175B [PAPER FACT].
  - CDF: **>95% requests <30ms transmission** despite 25 Gbps cross-node, thanks to NVLink co-location constraint [PAPER FACT] (Sec 6.3) [PAPER FACT].

- **Ablation (Fig11, OPT-66B ShareGPT simulated):**
  - **vLLM++ == vLLM** (intra4 best), proving parallelism tuning alone insufficient; interference limits gain [PAPER FACT].
  - **DistServe-High > DistServe-Low** because High not constrained to co-locate stages, can use tailored parallelism per phase more freely [PAPER FACT].
  - Simulator error **<2%** vs real system across rates 1.0-4.0 rps (Table2: e.g., vLLM 97.0% vs sim 96.8% at 1.0 rps, 23.6% vs 24.1% at 4.0 rps; DistServe 100% vs 100% at 1.0-1.5, etc.) [PAPER FACT] -- verifies accuracy.

- **Algorithm runtime (Fig12):**
  - Alg Low and High run in **minutes** on 96-core m5d.metal, independent of model size (discrete event sim), highly parallelizable linear speedup with cores [PAPER FACT].
  - Low takes slightly longer than High with many GPUs due to enumerating intra-node combos for co-location constraint, but still minutes for up to many GPUs [PAPER FACT].
  - Executed once before redeployment, overhead acceptable [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder LLM (OPT) with distinct prefill (compute-bound, superlinear, saturates GPU at ~512 tokens for 13B) and decode (memory-bound, low utilization) phases, sharing weights and KV caches [PAPER FACT].
- Goodput = per-GPU max rate meeting both TTFT and TPOT SLO attainment; SLOs vary per app (chatbot vs code vs summarization) and model size [PAPER FACT].
- Workload length distributions and Poisson arrival predictable over hours/days (fitting distribution from history and resampling) [PAPER FACT]; input lengths non-uniform but handled via simulation [PAPER FACT].
- Simulator can estimate SLO attainment via latency models (C1-5) and event-driven queuing, with high predictability of DNN execution [PAPER FACT].
- GPU memory capacity C limits batch size; model size / (inter*intra) must fit [PAPER FACT].
- Communication overhead negligible if placement respects bandwidth: intra-node NVLink 600 GB/s, inter-node InfiniBand 800 Gbps (or 25 Gbps limited case with co-location) [PAPER FACT]; KV size linear with prompt length, prefill quadratic [PAPER FACT].
- Replication linearly scales rate capacity (requests equally dispatched) [PAPER FACT]; FCFS scheduling [PAPER FACT].
- OPT with MHA represents worst-case KV size; GQA/MQA would only improve DistServe [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Discussed in Sec 7 Discussion:
  - **Throughput-optimized offline scenarios:** For latency-insensitive batch jobs, maximizing overall throughput (tokens/s) rather than goodput may favor chunked-prefill piggyback which keeps GPU compute-bound each iteration, while DistServes disaggregation may be less effective [PAPER FACT].
  - **Resource-constrained scenarios:** With few or single GPU, design space limited, cannot adjust parallelism/allocations effectively; simpler colocated (vLLM) may be better due to lower deployment complexity [PAPER FACT].
  - **Long-context serving (1M windows):** KV transmission grows linearly with prompt length, prefill grows quadratically so relative transmission cost decreases, but interference exacerbates; approach remains promising but authors note extremely long contexts beyond 2048 not evaluated due to OPT limit [PAPER FACT] (Sec 7, Sec 6.1 note vocab/positional limit).
  - **Algorithm assumes workload predictability:** Requires history to fit distribution; short-term unpredictability acknowledged but long-term predictable claim [PAPER FACT].
  - **No cost/energy evaluation of extra weight copies:** Disaggregation requires each phase hold full model weights (2x memory vs colocated), not quantified [NOT REPORTED as limitation but discussed as overhead].
  - Future work: integrating new parallelism optimizations, handling extremely long contexts [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Double weight memory overhead:** Prefill and decode instances each store full model weights, doubling total weight memory vs colocated for same total GPUs; for 175B (350GB) this is significant, not quantified in cost vs per-GPU goodput gain [AGENT INFERENCE].
- **Requires many GPUs to be effective:** Evaluation uses 32 GPUs (4 nodes); single-GPU or 2-GPU deployments show limited placement search space and may show minimal gain; not evaluated at scale <8 GPUs [AGENT INFERENCE].
- **Workload coverage limited to OPT and 2048 max length:** No evaluation on modern GQA models (LLaMA2-70B), Mixtral MoE, or 32k-128k contexts where KV transfer overhead changes dramatically [AGENT INFERENCE].
- **Poisson arrival and static SLOs:** Real traces bursty, multi-tenant, with varying TTFT/TPOT per request priority; static Poisson may underestimate tail [AGENT INFERENCE].
- **Simulator dependency:** Accuracy relies on profiling C1-5 constants per hardware/model; new hardware/model requires re-profiling; simulator <2% error shown only for limited rates, not for all workloads [AGENT INFERENCE].
- **No integration with prefix sharing:** No RadixAttention/prefix cache; repeated system prompts would transfer same KV repeatedly [AGENT INFERENCE].
- **Placement search exhaustive but may not scale:** Enumerating all inter*intra combos grows with N*M, though runtime minutes now, for 100s GPUs and heterogeneous clusters search may explode [AGENT INFERENCE].
- **Transmission still needs NVLink co-location:** Low-affinity placement constrains stages to same node, limiting flexibility and potentially causing fragmentation if nodes heterogeneous [AGENT INFERENCE].
- **No fault tolerance/elasticity:** Failure of prefill or decode instance mid-request requires restart; no checkpointing evaluated [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Heterogeneous hardware disaggregation:** Can prefill (compute-heavy) use H100/TPUs while decode (memory-heavy) uses cheaper/memory-optimized GPUs (A100, LPDDR) to optimize Perf/$ and Perf/W beyond homogeneous A100? [AGENT INFERENCE]
2. **Dynamic disaggregation:** How to handle bursty workloads where prefill/decode ratio shifts rapidly (e.g., chat vs summarization mix) without redeploying placements requiring minutes? [AGENT INFERENCE]
3. **Long context scaling:** At 100k-1M contexts, KV transfer size ~ 1M * 2 * h * layers * 2 bytes = 100s GB -- can NVLink/InfiniBand still hide transfer, or is compression/quantization needed? [AGENT INFERENCE]
4. **KV cache compression synergy:** Would combining DistServe with KIVI, GEAR quantization or MLA (DeepSeek-V2) increase effective goodput by reducing transfer and memory pressure? [AGENT INFERENCE]
5. **Scheduling beyond FCFS:** Could SLO-aware or priority scheduling (e.g., EDF, fair queuing) improve goodput under mixed TTFT/TPOT SLOs vs FCFS used? [AGENT INFERENCE]
6. **Multi-model serving:** How to extend disaggregated pools to serve multiple models multiplexed on same cluster with shared prefill/decode pools? [AGENT INFERENCE]
7. **Simulator generalizability:** Can latency model (C1-5) be auto-learned online to adapt to hardware variance, kernel updates (FlashAttention v3) without offline profiling? [AGENT INFERENCE]
8. **Goodput vs throughput tradeoff theory:** Formal Pareto frontier analysis between goodput (SLO-aware) and raw throughput (tokens/s) across disaggregated vs chunked-piggyback approaches? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Orca [Yu et al. OSDI22]:** Continuous batching, iteration-level scheduling predecessor; colocated, suffers interference which DistServe eliminates via disaggregation [PAPER FACT].
- **vLLM / PagedAttention [Kwon et al. SOSP23]:** Fine-grained KV management enabling large batches; baseline colocated system, DistServe builds orchestration on similar engines [PAPER FACT].
- **Sarathi / Sarathi-Serve [Agrawal et al. 2308.16369 / 2403.02310]:** Chunked-prefill + stall-free hybrid batching to mitigate interference but not eliminate; trades TTFT for TPOT with O(N^2) KV reload overhead; DistServe argues disaggregation superior for goodput [PAPER FACT].
- **Splitwise [Patel et al. 2311.18677]:** Concurrent disaggregation idea, phase splitting onto separate machines, heterogeneous A100/H100 pools, focuses on cost/throughput/power rather than goodput/SLO placement search; DistServe emphasizes goodput optimization and bandwidth-aware placement search [PAPER FACT].
- **TetriInfer [Hu et al. 2401.11181] and DejaVu [Strati et al. 2024]:** Concurrent disaggregation with similar goals, mentioned as corroborating effectiveness [PAPER FACT] (Sec 8).
- **FasterTransformer [NVIDIA], AlpaServe [Li et al.], Clockwork [Gujarati OSDI20], Shepherd [Zhang 2023]:** Serving systems; AlpaServe uses model parallelism for statistical multiplexing but non-autoregressive focus [PAPER FACT].
- **Pollux [Qiao OSDI21], Sia [Jayaram SOSP23]:** Goodput-optimized DL cluster scheduling, heterogeneous-aware, inspiring definition of per-GPU goodput [PAPER FACT].
- **Resource disaggregation works [LegoOS, Mira, DistMind etc.]:** Concept of separating hardware resources into pools, analogous to DistServes phase pools [PAPER FACT].
- **Follow-up to DistServe (not in paper):** Mooncake [Qin 2407.00079] extends disaggregation with KVCache-centric CPU DRAM pool and chunked pipeline parallelism, building on DistServes placement insights [AGENT INFERENCE].


## Review Log
Reviewer: Reviewer-1
Problems Found:
- Per-GPU goodput definition slightly simplified; corrected to explicit division by num_gpus as per ¡ì4/¡ì6.
- Throughput claims verified: chatbot ShareGPT 2.0-4.6x vs vLLM (1.6-7.4x vs DeepSpeed-MII), SLO 1.8-3.2x tighter; HumanEval 5.7x/1.4x vs vLLM, 1.6x/1.4x vs DS-MII; LongBench 4.3x/12.6x vs vLLM, 1.8x/2.6x vs DS-MII; Abstract 7.4x/12.6x headlines; 99% attainment Figs13-14 3-8x etc. ¡ª all trace to ¡ì6.2 Fig8-9.
- Queuing model Eqs 1-3 (M/D/1 Avg_TTFT) correctly captured with K factor for intra-op speedup.
- Bandwidth-aware placement Alg1 (high affinity) vs Alg2 (low affinity co-locate same node for NVLink 600GB/s, cross-node 25Gbps testbed) verified ¡ì4.1-4.2; latency breakdown <0.1% total and 95% <30ms transmission Fig10 confirmed.
- Simulator error <2% Table2 and runtime minutes on 96-core m5d.metal Fig12 verified.
- Hardware: 4 nodes x8 A100-80GB =32 GPUs, NVLink intra, 25Gbps cross (limited), high-affinity 800Gbps simulated ¡ª correct.
Corrections:
- Clarified per-GPU goodput definition.
- Fixed latency model formula spacing/tag.
- No hallucinated KV sharing or prefix cache (correctly notes none).
Confidence: High