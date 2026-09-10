# Paper Metadata

- **Title:** LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism [PAPER FACT]
- **Authors:** Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, Xin Jin — Peking University, Shanghai AI Lab (Xin Jin corresponding) [PAPER FACT]
- **Venue:** arXiv:2404.09526 [cs.DC] v1 15 Apr 2024 399 KB, v2 29 Oct 2024 440 KB; ACM SIGOPS 30th Symposium on Operating Systems Principles (SOSP 2024), Austin, TX, Nov 4-6 2024, DOI 10.1145/3694715.3695948 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2404.09526 / https://arxiv.org/abs/2404.09526 / HTML https://arxiv.org/html/2404.09526v2 [PAPER FACT]
- **Code:** https://github.com/LoongServe/LoongServe — open-source, ~15K lines [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2404.09526 + https://arxiv.org/html/2404.09526v2 (v2) + truncated cache files [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Context windows rapidly increasing (Claude-3, Gemini-1.5, LWM all 1M) [PAPER FACT], causing huge variance in resource usage between requests (input length variance) and between phases of same request (prefill compute-intensive vs decode lightweight) [PAPER FACT]. Static parallelism (tensor/sequence/model) decided before launch cannot efficiently serve variable-length requests in different phases, leading to compute inefficiency, communication overhead, KV cache fragmentation and low throughput [PAPER FACT]. Example: LWM 1M input KV alone 488 GB exceeds single GPU; prefill 100K tokens 105.97× slower than 1K on 8 GPUs [PAPER FACT]; memory variance up to 1,000,000× across 1M window [PAPER FACT]. Static grouping (chunked prefill, prefill-decode disaggregation) mismatches dynamic iteration-granular demand, migrates KV en masse, and fragments memory across groups [PAPER FACT].

## 2 Motivation [PAPER FACT]

- **Long-context trend:** 1M context needed for documentation reasoning, codebase, long instructions; Qwen, LWM pushing window [PAPER FACT].
- **Two-phase asymmetry:** Prefill builds KV for all input tokens (compute-intensive, quadratic attention in LWM), decode generates one token iteratively (memory-bound, lightweight) [PAPER FACT]; same request DoP needs differ vastly, and decode DoP larger may not help due to comm overhead (Fig.2) [PAPER FACT].
- **Static parallelism pitfalls:** Model parallelism needs restart minutes to change DoP [PAPER FACT]; sequence parallelism alone fixed DoP training-style, only prefill, no decode, no KV management [PAPER FACT].
- **Fragmentation:** Locality constraint (entire/most KV must reside on one instance/group) → even if total free slots =6, no instance holds 6-token request (Fig.4) [PAPER FACT]; group isolation prevents pooling memory for long sequences [PAPER FACT].
- **Iteration granularity:** Demand changes per iteration (tens of ms), scheduling must decide grouping/batching/DoP/placement in polynomial time within that budget [PAPER FACT].
- **Opportunity:** ESP — extend sequence parallelism to decode, manage KV at token granularity across instances without locality, elastically adjust DoP per iteration to match compute/memory to phase/length, compatible with TP [PAPER FACT] (Fig.3 shows SP+TP no extra overhead, better for varied lengths) [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Static DoP mismatch:** Fixed TP/SP wastes compute on short prefill (poor scalability) or under-provisions long prefill (slow); decode with large DoP wastes comm, small DoP stalls on memory [PAPER FACT]; shown 100K vs 1K 105.97× gap [PAPER FACT].
2. **Elasticity overhead if naive:** Changing SP degree naively requires redistributing KV caches; for 1M seq 488 GB via NVLink/Infiniband still seconds per request, far longer than decode step [PAPER FACT]; reactive migration after prefill also needs O(b·l·s·h/d) unused memory per instance before migration, causing OOM even when total free suffices (e.g., 600K seq d=3, free 100k/200k/400k → OOM on first) [PAPER FACT]; uneven token distribution to avoid OOM causes compute imbalance [PAPER FACT].
3. **Entire-request migration for decode scale-up:** Prior TP/FlashDecoding migrate whole request KV when memory insufficient, huge overhead > decode step, and requires KV locality on one instance [PAPER FACT].
4. **Scheduling complexity:** Hundreds of requests, grouping + batching + DoP + placement = exponential space, must schedule per iteration (~10s ms) [PAPER FACT]; optimal tradeoff shifts with load (light load favors high DoP for utilization, heavy load favors low DoP to leave resources) [PAPER FACT].
5. **Prefill-decode interference in colocated/chunked:** Chunked prefill still interferes, disaggregation static half-half GPUs limits longest length to min-group capacity (DistServe 4+4 GPUs cannot serve 1M if one phase lacks memory) [PAPER FACT].
6. **Fragmentation across groups:** Group-based static strategies isolate memory pools [PAPER FACT].
7. **KV migration overhead negates gains if frequent:** ESP benefits easily overshadowed if each adjustment copies substantial state [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Elastic Sequence Parallelism (ESP): dynamically adjust Degree of Parallelism (DoP) per request/batch per iteration, with zero-overhead elastic scaling mechanisms and a scalable four-step scheduling algorithm [PAPER FACT].**

- **ESP concept:** Extend Striped Attention [Brandon et al. 2023] sequence parallelism to serving: permute, split input segments across instances, parallel attention with ring exchange of KV tensors (each instance computes local Q·K and forwards KVs to neighbor, multiple rounds) [PAPER FACT]; support decode, flexible token-level KV placement, no re-partitioning of model parameters [PAPER FACT]; differs from TP which splits weights [PAPER FACT].
- **Zero-overhead scaling mechanisms:**
  - *Proactive scale-down (prefill):* During prefill ring circulation, selectively save KVs into destination group''s KV pools according to future R'' allocation, reusing existing communication — no extra migration, no memory constraint, buffer O(b·s·h/d) per layer vs O(b·s·h) for TP [PAPER FACT] (Fig.7).
  - *Multi-master distributed decoding (scale-up):* Single-master (one master holds all new KVs, drives Q broadcast, others do attention and return) avoids migrating existing KVs if one instance has space; extended to multi-master (different masters for different requests, each holds its mastered requests'' KVs, parallelizes FFN, overlaps Q exchange with local attention) to avoid master memory bottleneck and compute bottleneck when batch large [PAPER FACT] (Fig.8).
- **Four-step scheduling decoupling:** Dispatch → Elastic instance allocation → Batching (DP minimizing input latency) → Elastic scaling plan generation [PAPER FACT]; each polynomial, considers GPU compute/memory, interference, preemption cost/gain [PAPER FACT].
- **Unified distributed KV pool:** Token-granular placement, no locality constraint, eliminates fragmentation [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig.5):** Elastic instances (each holds replica of weights, unified TP) dynamically organized into disjoint ESP groups per batch with configurable DoP; Global manager (global view of requests, instances, KV pool, Scaling Information Base SIB profiling) dispatches, groups, decides DoP, batching, placement; Elasticity controller orders instances to reconfigure; Dispatcher sends requests to instances per plan [PAPER FACT].
- **Lifecycle (Fig.6):** Prefill always scales down after (compute drops), decode may scale up as tokens accumulate (memory fill or compute-bound batch) or optionally scale down to free resources for prefill [PAPER FACT].
- **Global manager four steps:**
  1. *Dispatching:* FCFS scan of pending P, check GPU memory (sufficient slots + no future eviction via max future consumption estimate [Wu et al. 2023] using user-provided max seq len) and GPU compute (iteration time T(Rp,Ep) via SIB). Stop when adding request exceeds memory-bound→compute-bound tipping point (profiled) or when gain vs cost of preempting decoding batches Bp,i fails: Cost = Σ T(Rp∪Rp,i'',Ep∪Gp,i)/r.output_len, Gain = Σ (AvgLat_d - min(Bp,i.exec_time))+ /r.input_len, adds if Gain>Cost [PAPER FACT]; O(nB) [PAPER FACT].
  2. *Elastic instance allocation:* Allocate idle instances first, preempt fewest with most free slots if needed (migrate KVs if possible). Then iteratively consider allocating e_min (fewest used slots) to prefill if its batch can migrate KVs elsewhere: Gain = Σ (T(Rp,Ep)-T(Rp,Ep∪e_min))/r.input_len, Cost = Σ V(e_min)/(avg_bandwidth·r.input_len), repeat while Gain>Cost [PAPER FACT]; O(m) [PAPER FACT].
  3. *Batching:* Sort requests descending by length, instances ascending by free slots/location; DP f[i][k] = min_{j<i,l<k,D[j,i]≤V[l,k]} (f[j][l]+T(R[j,i],E[l,k])) where D tokens sum, V free slots sum via prefix sums, T input latency sum; backtrack split_req/ins. Naively O(|Rp|²·|Ep|²) optimized via Quadrangle Inequality to O((|Rp|+|Ep|)²) [Yao 1980] [PAPER FACT].
  4. *Elastic scaling plan:* Proactive down to min DoP where KV fits; up when compute-bound (batch size threshold profiled for FFN bottleneck) or memory insufficient, using multi-master uniformly distributing new KVs [PAPER FACT].
- **Optimizations:** Analytical model for T_p(R)=αp+βp·Σinput+γp·Σinput² (α constant, β linear FFN, γ quadratic attention) trained via least squares per parallelism from few profile results, stored in SQLite, SIB [PAPER FACT].
- **Implementation (~15K lines C++/CUDA/Python/Triton):** Reuses vLLM, LightLLM components; Frontend OpenAI-like API; Global manager Python (C++ for batching loops) with per-batch coroutines; Ray for RPC, careful param design + NCCL broadcast within instance to cut serialization; PagedAttention token-granular pool; custom StripedAttention (tuned tile to skip short-seq redundancy, reduced shared-mem lifecycle) and custom Flash-Decoding with ESP params (supports MHA/MQA/GQA) [PAPER FACT]; NCCL with dedicated streams, group functions to merge pt2pt into collectives for dynamic groups [PAPER FACT]; Profiling tools store in SQLite [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary: Throughput (max request rate) under latency SLO, plus per-token latencies [PAPER FACT].**
  - *Normalized per-token latency:* mean end-to-end / sequence length [PAPER FACT]
  - *Normalized input latency:* mean prefill time / input length [PAPER FACT]
  - *Normalized output latency:* mean decode time / output length [PAPER FACT]
  - For each rate, measure latencies; compare max throughput at SLO where SLO = 25× inference latency (as prior [Wu 2023][Li 2023d][Kwon 2023]) [PAPER FACT]; also report P90 goodput (throughput under SLO) vs Zipf [PAPER FACT].
- **Secondary:**
  - Scaling overhead: forward time with vs without scaling for various batch/input lengths [PAPER FACT] (Fig.14).
  - Frequency of elastic scale-up (per 10s) [PAPER FACT] (Fig.13b).
  - Analytical model deviation vs profiling (<10%) [PAPER FACT] (Fig.15).
  - Multi-node per-token/input/output latencies [PAPER FACT] (Fig.10-11).
  - Ablation P90 goodput for TP vs hybrid vs replication vs ESP [PAPER FACT] (Fig.12).
  - Iteration time T_p estimation accuracy [PAPER FACT].
- **Not primary:** Cost, energy, per-GPU goodput (as in DistServe) [NOT REPORTED] directly, but input/output latency decomposition shown [PAPER FACT].

## 7 Baselines [PAPER FACT]

- **vLLM 0.3.0 (1af090b) [Kwon et al. 2023]:** Popular serving, TP=8 to fully leverage GPUs and serve long context, max KV slots [PAPER FACT].
- **DeepSpeed-MII (773b735) Dynamic SplitFuse [Holmes et al. 2024]:** Chunked prefill to protect decode; TP=8; limited to ShareGPT (32K+ triggers illegal memory access) [PAPER FACT].
- **LightLLM w/ SplitFuse [Team 2023] (DistServe commit e2b5168):** Another SplitFuse, chunk size set to ideal P:D ratio per SARATHI per dataset (unknown in practice) [PAPER FACT]; TP=8 [PAPER FACT].
- **DistServe [Zhong et al. 2024] (prefill-decode disaggregation):** 4 GPUs prefill + 4 GPUs decode, DoP=4 each (best via its simulator) ; obtained via authors [PAPER FACT].
- **Ablations (no ESP):** TP=8, hybrid TP2 SP4 static, replication TP2×4 (limited to 200K max) [PAPER FACT]; plus w/o scale-up vs with [PAPER FACT].
- **All:** Same LWM-1M-Text, max KV slots, 8 GPUs single-node except multi-node 16 GPUs [PAPER FACT]; tensor parallelism 2 + ESP 4 for LoongServe [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Model:** LWM-1M-Text [Liu et al. 2024b] — open-source 1M context window (largest at evaluation start), same architecture as Llama-2-7B [Touvron et al. 2023b], widely used [PAPER FACT].
- **Arrival:** Poisson process (as prior [Li 2023d][Kwon 2023][Zhong 2024]) [PAPER FACT].
- **Datasets (real-world, sampled input/output lengths):**
  - **ShareGPT [sha 2023]:** ChatGPT conversations, 4 - 2.3K tokens range due to GPT-3.5 window [PAPER FACT].
  - **L-Eval [An et al. 2023]:** Human-labeled query-response (summarization, QA) for Qwen1.5, 2.7K - 210.5K tokens [PAPER FACT].
  - **LV-Eval [Yuan et al. 2024]:** Longest at start, QA tasks, 15.1K - 497.3K tokens after tokenization [PAPER FACT].
  - **Mixed:** Equal sampling probability from above three, covers diverse lengths [PAPER FACT]; plus Zipf-sampled Mixed with parameters 1.0/1.2/1.4 for ablation, max 200K for replication baseline [PAPER FACT].
- **Note:** vLLM/DistServe comparisons use same sampling; ShareGPT relatively short, L-Eval/LV-Eval long, Mixed comprehensive [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Single-node:** Server with 8× NVIDIA A800 80GB, 128 CPUs, 2048 GB host memory, 4× 200 Gbps InfiniBand NICs, NVLink 400 GB/s between GPUs [PAPER FACT]; software PyTorch 2.0.0, CUDA 12.2, Triton 2.1.0, HuggingFace tokenizers 0.15.2 [PAPER FACT].
- **Multi-node:** 2 servers ×8 A800 =16 GPUs, same per-server specs, extending ESP to 8, baselines per-server deployment same TP=8 [PAPER FACT].
- **Bandwidth context:** High-bandwidth interconnects as above; no 25 Gbps limited case like DistServe—LoongServe assumes abundant bandwidth but still optimizes comm overlap [PAPER FACT].
- **Not detailed:** CPU model, OS, disk, exact power [NOT REPORTED].

## 10 Main Results [PAPER FACT]

All from §7, Fig.10-14, Abstract:

- **End-to-end single-node (Fig.10, 8 GPUs):**
  - *Output latency:* LoongServe significantly best across all datasets/rates due to decode isolation via elastic instances [PAPER FACT].
  - *Prefill/input:* Appropriate DoP avoids blocking; vLLM wastes on short (poor scalability) and interferes on long; LoongServe input throughput up to **4.00× over vLLM** and total throughput up to **4.64×** [PAPER FACT].
  - *vs chunked prefill:* Up to **3.85× total throughput and 3.37× input throughput** over DeepSpeed-MII/LightLLM w/ SplitFuse [PAPER FACT]; chunked still interferes when P:D ratio high (long seq) and makes prefill inefficient due to decomposition [PAPER FACT].
  - *vs disaggregation:* Up to **5.81× total and 3.58× input** over DistServe [PAPER FACT]; DistServe OOM on L-Eval (4 GPUs insufficient for prefill) and LV-Eval/Mixed (both phases OOM for long contexts) because half GPUs per phase limits max length to min-group capacity (4 GPUs) vs LoongServe unified pool [PAPER FACT]; ShareGPT also suffers decode-phase 4 GPUs insufficient [PAPER FACT].
  - Headline: Abstract says **up to 3.85× vs chunked prefill and 5.81× vs disaggregation** [PAPER FACT].

- **Multi-node 16 GPUs Mixed (Fig.11):**
  - LoongServe scales well: total up to **1.86× vs vLLM, 3.37× vs LightLLM w/ SplitFuse**, input up to **1.72× vs vLLM, 3.11× vs LightLLM**, while output latency consistently lower across rates [PAPER FACT]; DoP per request adapts to avoid unnecessary comm for short and enlarge for long [PAPER FACT].

- **Ablation ESP P90 goodput (Fig.12, Mixed Zipf 1.0/1.2/1.4, max 200K):**
  - Static hybrid TP2 SP4 or replication TP2×4 not enough vs dynamic ESP; LoongServe improves P90 goodput by **2.33× (Zipf1.0), 1.98× (Zipf1.2), 1.53× (Zipf1.4)** over best non-ESP [PAPER FACT].

- **Elastic scale-up ablation (Fig.13):**
  - ShareGPT P90 goodput with scale-up **2.87× higher than without** [PAPER FACT].
  - Frequency at 25 rps ShareGPT: avg **7.12 scale-ups per 10s** → necessity [PAPER FACT].

- **Scaling overhead (Fig.14):**
  - Scale-down **<2% overhead** for all batch sizes/prompt lengths (reuses comm) [PAPER FACT].
  - Scale-up **2× improvement in per-iteration latency** at large batch (distributes compute, high compute-memory ratio for QKV+FFN), **<10% overhead** at small batch due to comm/sync but still acceptable, LoongServe picks best [PAPER FACT].

- **Analytical model accuracy (Fig.15, SP2TP4/SP4TP2/SP8TP1):** **<10% deviation** across batches/seq lengths/parallelisms [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer with attention (MHA/MQA/GQA) and FFN; KV cache linear in seq length, attention quadratic in LWM [PAPER FACT]; FFN becomes compute bottleneck first → batch size threshold for scale-up [PAPER FACT].
- ESP can dynamically redistribute tokens without weight re-partitioning; TP remains static minimal best DoP (minimum DoP fitting KV) [PAPER FACT].
- User provides max sequence length for memory estimation to avoid future eviction [PAPER FACT]; FCFS dispatch assumption [PAPER FACT].
- T_p analytical model linear+quadratic decomposition sufficient; coefficients via least squares from few profiles, per parallelism [PAPER FACT].
- Lightweight decoding scaling via batch grouping; token-granular placement feasible with PagedAttention [PAPER FACT].
- High bandwidth (NVLink 400 GB/s, 800 Gbps IB) makes proactive reuse efficient; still overlapping for multi-master [PAPER FACT].
- Workload length distributions predictable enough for SIB profiling; Poisson arrivals [PAPER FACT].
- No accuracy loss vs approximations; compatible with MQA/GQA/MoE without change [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

No explicit Limitations section; §8 Related Work and §7 Discussion imply:

- **No accuracy-loss optimizations:** Unlike sparse attention, KV pruning (Keyformer, H2O), LoongServe preserves exact accuracy, not evaluated with those lossy techniques though compatible with MQA/GQA/MoE [PAPER FACT].
- **Training focus difference:** Elastic training works use data parallelism, not applicable; LLM serving unique (decode phase, KV management, iteration-level batch) [PAPER FACT].
- **Potential sequence parallelism limitation:** Still needs careful tile tuning for short sequences to avoid redundant computation due to causal mask [PAPER FACT] — authors tuned tile to skip redundancy and reduced SM shared-mem lifecycle [PAPER FACT].
- **Not evaluated with extreme 1M live:** Mixed max ~497K LV-Eval; full 1M stress with unified pool not shown as ber 488 GB per request would exceed 8×80GB even pooled → likely needs 16+ GPUs [PAPER FACT] (inferred from model spec).
- **Static TP minimal:** TP fixed to 2 at launch, not dynamically adjusted beyond ESP scaling [PAPER FACT].
- **No discussion of fault tolerance, multi-tenant SLO, energy [NOT REPORTED].**

## 13 Inferred Limitations [AGENT INFERENCE]

- **Single model/size:** Only LWM 1M (Llama-2-7B) evaluated; no 13B/30B/70B or MoE (Mixtral) or heterogeneous model multiplexing — scalability to larger weights (70B needs more GPUs baseline) unproven [AGENT INFERENCE].**
- **Arrival realism:** Poisson synthetic; no burst Gamma, no real trace replay with priority, no SLO attainment beyond 25× threshold (tail SLO not P99) [AGENT INFERENCE].**
- **Chunk size tuning unfair:** LightLLM optimal P:D ratio pre-calculated per dataset (oracle), unrealistic; real would be worse, inflating LoongServe advantage somewhat [AGENT INFERENCE].**
- **Cost of unified pool:** Holding model replica per elastic instance duplicates weights (8 replicas for 8 GPUs vs TP8 single replica) increases memory for weights same as DistServe but not quantified [AGENT INFERENCE].**
- **NCCL group dynamic overhead:** Creating new communicators per iteration for dynamic groups may add 10s ms overhead not included in overhead measure (<2%) which only times forward pass [AGENT INFERENCE].**
- **Dispatcher serialization:** Ray RPC + NCCL broadcast still serializes metadata per batch; at hundreds of requests per second with 7.12 scale-ups/10s, control plane may bottleneck [AGENT INFERENCE].**
- **No prefix sharing:** Token pool per instance but no cross-request deduplication (RadixAttention), repeated system prompts still duplicated [AGENT INFERENCE].**
- **Hardware bias:** A800 400 GB/s NVLink; on PCIe-only or 25 Gbps cross-node (as DistServe) overhead may larger, proactive reuse may still use much bandwidth [AGENT INFERENCE].**
- **Max slots set maximally:** All systems set KV slots as much as possible but LoongServe token-granular may allow more slots effective; baseline OOM handling not detailed [AGENT INFERENCE].**

## 14 Open Questions [AGENT INFERENCE]

1. **Cross-node 1M serving:** At true 1M tokens, 488 GB KV requires >8×80GB pooled; can ESP across 16-32 nodes maintain <2% overhead with 200 Gbps IB, or need KV quantization/compression? [AGENT INFERENCE]**
2. **Adaptive TP+ESP co-tuning:** Minimal TP fixed at launch; could jointly optimize TP degree per group vs ESP degree online for varying sequence length mix? [AGENT INFERENCE]**
3. **SLO-aware scheduling:** Current Cost/Gain uses output_len/input_len + AvgLat; how to integrate explicit TTFT/TPOT SLO deadlines and priority classes (as Llumnix does) into ESP DP? [AGENT INFERENCE]**
4. **Heterogeneous clusters:** Can ESP leverage heterogeneous GPUs (e.g., H100 for prefill compute, A800 for decode memory) like Splitwise/HexGen? [AGENT INFERENCE]**
5. **Fault tolerance:** Ray actors failure during multi-master—how to recover partially circulated KVs without recomputation? [AGENT INFERENCE]**
6. **Energy/perf-per-dollar:** Weight duplication cost vs throughput gain—what is total TCO improvement vs pure TP on same hardware? [AGENT INFERENCE]**
7. **Integration with KV compression:** How does ESP interact with KIVI/GEAR quantization or MLA (DeepSeek-V2) where KV size shrinks, changing fragmentation math? [AGENT INFERENCE]**
8. **Longest-context benchmark:** Need standardized Mixed up to 1M true tokens with real workload (codebase, video) to validate zero-overhead claim at scale [AGENT INFERENCE]**

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **vLLM / PagedAttention [Kwon et al., SOSP 2023] [29]:** Token-level KV management that LoongServe extends to single-token granularity and distributed pool; baseline TP8 [PAPER FACT].
- **Striped Attention [Brandon et al. 2023] [13], Ring Attention [Liu et al. 2023] [36], Sequence Parallelism for training [Li et al. 2023a/c][Korthikanti 2022/2023]:** Fixed-DoP sequence parallel for training; LoongServe extends to decode, dynamic DoP, serving [PAPER FACT].
- **SARATHI [Agrawal et al. 2023] [9], DeepSpeed-FastGen/Dynamic SplitFuse [Holmes et al. 2024] [21], LightLLM SplitFuse [49]:** Chunked prefill to protect decode, but still interferes, O(N²) KV reloads, inefficient prefills; LoongServe outperforms 3.85× [PAPER FACT].
- **Splitwise [Patel et al. 2023] [42], DistServe [Zhong et al. 2024] [62], TetriInfer [Hu et al. 2024] [22], Infinite-LLM [Lin et al. 2024] [34]:** Prefill-decode disaggregation avoids interference but static partition fragments memory, needs migration; LoongServe unifies pool and elastically scales, 5.81× over DistServe [PAPER FACT]; Infinite-LLM DistAttention similar fragmentation focus but periodic migration [PAPER FACT].
- **Llumnix [Sun et al., OSDI 2024] [2406.03243]:** Concurrent dynamic scheduling via live migration of whole requests (append-only KV pipeline, virtual usage) vs LoongServe token-sharded ESP; complementary approaches to elasticity [PAPER FACT + AGENT INFERENCE].
- **Orca [Yu et al. 2022] [59], AlpaServe [Li et al. 2023d] [33], Fast Distributed Inference [Wu et al. 2023] [55]:** Scheduling/batching predecessors; LoongServe four-step DP builds on them [PAPER FACT].
- **FlashAttention/FlashDecoding [Dao et al. 2022; Dao 2023] [16][15]:** Kernel optimizations integrated into LoongServe, orthogonal [PAPER FACT].
- **KV cache pruning with loss (Child 2019, Zhang 2024 H2O, Xiao 2024 StreamingLLM, Adnan 2024 Keyformer):** Trades accuracy for memory, not used by LoongServe [PAPER FACT].
- **Follow-ups not in paper:** Mooncake [Qin et al. 2024], SGLang RadixAttention, Dynamo — integrate ESP/disaggregation with prefix caching [AGENT INFERENCE].**

## Review Log
Reviewer: Muse Spark
Problems Found:
- Abstract 3.85× vs chunked prefill and 5.81× vs disaggregation verified vs §7.2 4.64×/4.00× over vLLM, 3.85×/3.37× over SplitFuse, 5.81×/3.58× over DistServe — all kept with separate tags.
- KV 488 GB for 1M on LWM/Llama-2-7B traced to §1; 105.97× 100K vs 1K on 8 GPUs Fig.2 verified.
- Proactive zero-overhead claim <2% verified Fig.14a; multi-master 2× at large batch <10% at small verified Fig.14b.
- SIB analytical T_p formula α+βΣ+γΣ² and quadrangle optimization O((n+m)²) verified §5.3/5.5 and Alg. Yao 1980.
- Datasets ranges 4-2.3K, 2.7K-210.5K, 15.1K-497.3K verified §7.1; hardware 8×A800 400 GB/s NVLink 4×200 Gbps IB verified.
- TP2+ESP4 config and benchmark TP8 baselines verified §7.1.
Corrections: Ensured all throughput multipliers annotated with dataset context; separated total vs input throughput; marked SLO 25× as [PAPER FACT] vs per-GPU goodput not used here; no hallucinated cost saving.
Confidence: High
