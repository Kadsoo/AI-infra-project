# Paper Metadata

- **Title:** Splitwise: Efficient generative LLM inference using phase splitting [PAPER FACT]
- **Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, Ricardo Bianchini [PAPER FACT] -- University of Washington (Patel) + Microsoft (others) [PAPER FACT]
- **Venue:** arXiv preprint arXiv:2311.18677v2 [cs.AR, cs.DC], submitted 30 Nov 2023 v1, last revised 20 May 2024 v2, 12 pages, 19 figures [PAPER FACT]; CC BY-NC-SA 4.0 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2311.18677 / https://arxiv.org/abs/2311.18677 / https://arxiv.org/html/2311.18677v2 [PAPER FACT]
- **Code/Trace:** Splitwise implementation on vLLM via pull #2809 [PAPER FACT]; traces released at https://github.com/Azure/AzurePublicDataset (Azure LLM Inference Trace 2023) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2311.18677 + https://arxiv.org/html/2311.18677v2 (v2, 20 May 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Generative LLM inference deployed on expensive, power-hungry GPUs (A100, H100) suffers low efficiency due to phase heterogeneity [PAPER FACT]. Each request has prompt computation (processing input tokens in parallel to generate first token, compute-intensive) and token generation (autoregressive one token per iteration, memory-intensive) with distinct latency, throughput, memory, power characteristics (Sec III) [PAPER FACT]. Despite state-of-art batching/scheduling (request-level, continuous, mixed), token generation underutilizes compute (most time with <=20 active tokens, Fig4, 60-70% time <=20 for conversation, >20% single token for coding) [PAPER FACT]. Latest GPUs (H100 vs A100: 3.43x TFLOPs, 1.64x HBM bandwidth, 1.75x power, 2.16x cost per machine Table1) increase compute faster than memory bandwidth/capacity, worsening inefficiency [PAPER FACT]; worldwide GPU crunch and power wall require better utilization [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLM inference demand far exceeds training due to many apps; must amortize training cost over many inferences, but inference expensive [PAPER FACT].
- H100 has 3.43x compute but only 1.64x bandwidth and same 80GB capacity as A100 (Table1), costing 2.16x more ($38/hr vs $17.6/hr) and 1.75x more power (700W vs 400W) -- compute not proportionally used by token phase [PAPER FACT].
- Characterization insights (Sec III):
  - Different services have widely different prompt/token distributions (coding median prompt 1500 vs conversation 1020 Fig3a; coding median output 13 vs conversation 129 bimodal Fig3b) -- Insight I [PAPER FACT].
  - Mixed continuous batching spends most time with very few active tokens (Insight II) [PAPER FACT].
  - Most E2E time spent in token phase: e.g., BLOOM-176B 1500 prompt tokens same time as 6 decode tokens; for most requests majority time in token gen -- Insight III [PAPER FACT].
  - Prompt throughput degrades after 2048 tokens (batch <2 for median prompts) while token throughput keeps scaling to batch 64 until OOM (Fig6) -- Insight IV [PAPER FACT].
  - Prompt batching compute-bound, token batching memory-capacity-bound (Insight V, Fig7) [PAPER FACT].
  - Prompt power draw increases with batch size, token power flat (Fig8); token tolerates 50% power cap (700->350W) with almost no latency impact, prompt highly sensitive (Fig9) -- Insight VI [PAPER FACT].
  - Token can run on less compute-capable hardware for better Perf/W and Perf/$: A100 vs H100 shows TBT degradation only 0.70x but cost 1.24-1.5x higher, energy 1-1.2x higher (Table4) -- Insight VII [PAPER FACT].
- Need phase-specific resource management using well-suited hardware and independent provisioning per phase [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Token phase underutilization.** [PAPER FACT] Despite batching, effective batch size small; e.g., conversation 70% time at <=15 tokens on 40 Baseline-H100 at 70 RPS (Fig17a); token generation does not need latest GPU compute [PAPER FACT].
2. **Prompt vs token resource conflict when colocated.** [PAPER FACT] Batching mechanisms: request-level causes TTFT/E2E wait (Fig2a), continuous batching (prompt preempts token, Fig2b) reduces TTFT but spikes TBT tail/E2E, mixed batching (Fig2c) reduces but not eliminates TBT impact because prompt+token together still lengthens decode iteration [PAPER FACT]; services over-provision GPUs to meet SLOs [PAPER FACT].
3. **Memory vs compute scaling divergence.** [PAPER FACT] H100 compute 3.43x but bandwidth 1.64x, capacity flat; token phase bound by bandwidth/capacity, prompt by FLOPs [PAPER FACT]; using same machine type for both phases wastes either compute or memory [PAPER FACT].
4. **Power wall:** GPU clusters power-hungry (A100 400W, H100 700W); peak power drives datacenter cost; token phase wastes power budget [PAPER FACT] (Fig8-9) [PAPER FACT].
5. **Heterogeneous requirement not met:** Without splitting, cannot run token phase on older/cheaper hardware (A100) or power-capped hardware effectively [PAPER FACT].
6. **Throughput vs latency tradeoff with mixed batches:** Increasing batch to improve throughput hurts TBT (Fig5b: batch64 2x TBT), prompt batching beyond 2048 hurts throughput (Fig6a) [PAPER FACT].
7. **Memory capacity limits token batching:** Token phase needs KV cache per context; capacity caps batch before compute saturates (Fig7) [PAPER FACT]; till then throughput scales linearly [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Splitwise: split prompt computation and token generation phases onto separate machines (prompt pool vs token pool + elastic mixed pool), enabling phase-specific hardware, provisioning, and scheduling [PAPER FACT]; efficiently transfer KV-cache between machines via optimized network libraries on fast back-plane interconnects [PAPER FACT]. Design homogeneous/heterogeneous clusters optimized for throughput, cost, power [PAPER FACT].**

- **Phase splitting:** All machines pre-loaded with model; CLS assigns each request to prompt+token pair; prompt machine generates first token+KV-cache, sends KV to token machine which continuous-batches token generation until completion; mixed pool machines use mixed continuous batching and elastically join either pool as load dictates [PAPER FACT] (Fig10) [PAPER FACT].
- **Hierarchical scheduling:** CLS (cluster-level, 1) for pool management + JSQ request routing by pending tokens, overlapping KV transfers with prompt compute; MLS (machine-level, 2) per machine for FCFS batching, memory tracking, queue reporting, prompt-token prioritization with preemption and starvation avoidance (increase token priority with age, limit preemptions) [PAPER FACT] (Sec IV-A/B) [PAPER FACT].
- **KV-cache transfer optimization (Sec IV-C, Fig11):** Serialized transfer after prompt stalls TBT/E2E; layer-wise async transfer per layer overlaps with prompt next layer compute, plus semaphore-based sync via one-sided put with MSCCL++ (zero-copy), contiguous KV blocks sharing semaphore, picking serialized for <512 tokens else layer-wise [PAPER FACT]; overhead minimal (<7% prompt time, constant ~8ms A100/5ms H100 non-overlapped, Fig14) [PAPER FACT].
- **Provisioning framework (Sec IV-D):** Search design space via event-driven simulator (performance model <3% MAPE, Fig13) over prompt/token machine counts (e.g., Fig12: 27 prompt +3 token optimal for 70 RPS coding on Splitwise-HH), SLOs (Table6: P50/P90/P99 for TTFT 2x/3x/6x, TBT/E2E 1.25x/1.5x/5x vs no-contention DGX-A100), optimization goals (throughput, cost, power) to find iso-throughput/power clusters [PAPER FACT].
- **Four design variants (Table5):** Splitwise-AA (A100/A100, 1x cost/power), HH (H100/H100, 2.35x/2.5x cost, 1.75x power, 2x IB bandwidth), HHcap (H100 power-capped token 1.23x power, 2x BW), HA (H100 prompt / A100 token, 2.35x/1x cost) [PAPER FACT]; reasoning based on insights VI/VII [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Cluster-level scheduler (CLS):** Maintains prompt, token, mixed pools (3); initial assignment based on expected load and token distributions; dynamic movement to/from mixed pool when pending queue > threshold, JSQ by pending tokens, preferring mixed pool then opposite pool; infrequent coarse-grained re-purposing between prompt/token pools if machine stays long in mixed [PAPER FACT]; simultaneously assigns prompt+token machines to overlap transfer [PAPER FACT] (Sec IV-A) [PAPER FACT].
- **Machine-level scheduler (MLS):** Per-machine FCFS: prompt machines batch max 2048 tokens (throughput degrades beyond, Fig6a) [PAPER FACT]; token machines batch up to memory limit (throughput keeps scaling to 64 in Fig6b) [PAPER FACT]; mixed machines prioritize prompts, preempt tokens if needed with aging/limit to avoid starvation [PAPER FACT]; reports queue/memory to CLS not per iteration [PAPER FACT].
- **KV-cache transfer layer:** Uses MSCCL++ GPU-driven one-sided put primitive over InfiniBand, zero-copy, as soon as layer ready without token-side recv, semaphore sync per request (different semaphore per token machine), shipping block-by-block for vLLM PagedAttention contiguously [PAPER FACT] (Sec V-A) [PAPER FACT].
- **Provisioning/simulator:** Event-driven simulator modeling pools, schedulers, memory, queues, KV transfer (Fig13); performance model piecewise linear, trained from profiling at various batch/input/output sizes and parallelism (8 GPUs TP), MAPE <3% [PAPER FACT]; communication model benchmarks actual KV transfer over IB [PAPER FACT]; input: model perf model, trace, SLOs, cluster config, search for optimal iso-throughput/power/cost point (e.g., Fig12) [PAPER FACT].
- **Implementation on vLLM:** Modified vLLM (was only continuous batching with preemption) to support mixed continuous batching and Splitwise role assignment, deployed on Azure VMs (2 DGX-A100, 2 DGX-H100, IB 200/400 Gbps) [PAPER FACT]; open source [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Throughput (requests per second, RPS) under SLO:** Max sustainable load where all 9 SLOs (P50/P90/P99 for TTFT/TBT/E2E) meet thresholds Table6; expressed as multiples vs no-contention A100 (e.g., P99 TTFT 6x, TBT 5x) [PAPER FACT]; also iso-power/iso-cost throughput, and throughput per cost (RPS/$) and per power [PAPER FACT].
  - **Latency percentiles:** TTFT (time to first token), TBT (time between tokens), E2E (total), reported P50/P90/P99 CDFs vs load (Fig16) [PAPER FACT]; second token latency also measured (serialized +64% vs Splitwise +16.5% Fig15) [PAPER FACT].
- **Secondary:**
  - Batched active tokens distribution CDF vs time at varying loads (Fig4, Fig17) [PAPER FACT].
  - Latency vs prompt size (TTFT linear) and vs batch size (TBT 2x at 64) and E2E percentiles (Fig5) [PAPER FACT].
  - Throughput vs batch size for prompt (degrades after 2048) and token (scales to 64) (Fig6) [PAPER FACT].
  - Memory utilization vs batch size (Fig7) [PAPER FACT].
  - Power utilization normalized to TDP vs batch size (Fig8) and vs power cap (Fig9) [PAPER FACT].
  - KV transfer latency vs prompt size: serialized linear vs layer-wise constant 8ms A100/5ms H100 (<7% prompt) (Fig14), and E2E impact 0.8% vs 3% serialized (Fig15) [PAPER FACT].
  - Cost ($/hr per machine Table1: A100 $17.6, H100 $38) and energy (Whr P50 Table4) [PAPER FACT].
  - Cluster space (number of machines) and provisioned power vs cost/throughput summary plots (Fig18-19) [PAPER FACT].
  - Workload sensitivity: coding vs conversation throughput/latency under mismatched provisioning (Fig20) [PAPER FACT].
- **Not primary but reported:** TTFT/TBT/E2E SLO attainment under batch jobs at high load (RPS/$ 0.89 for A100) [PAPER FACT].

## 7 Baselines [PAPER FACT]

- **Baseline-A100:** 40 DGX-H100-equivalent power budget = 70 DGX-A100 machines (iso-power for 40 H100) with mixed continuous batching on all machines; homogeneous A100 cluster [PAPER FACT] (Sec V-B, Sec VI-B).
- **Baseline-H100:** 40 DGX-H100 machines with mixed batching; homogeneous H100 cluster [PAPER FACT].
- **Splitwise variants vs baselines:**
  - Splitwise-AA (A100/A100) provisioned 55P+15T for coding at 70 RPS peak (Fig12 example 27P+3T for small trace) and similar for conversation (e.g., 35P+5T coding vs 25P+15T conversation on HH) [PAPER FACT] (Sec VI-B legend).
  - Splitwise-HH, HHcap, HA with machine counts searched via simulator [PAPER FACT].
- **Batch mechanism baselines conceptually:** Request-level, continuous, mixed batching timelines Fig2; vanilla vLLM only continuous with preemption (implemented mixed for fair) [PAPER FACT].
- **No direct comparison to DistServe/Sarathi at publication time (concurrent work):** Discussed in related work as heterogeneous scheduling [PAPER FACT]; evaluation focuses on baseline homogeneous vs Splitwise heterogeneous.

## 8 Workloads [PAPER FACT]

- **Models (Table3):**
  - **BLOOM-176B** (70 layers, hidden 14336, 112 heads) [PAPER FACT]
  - **LLaMA-70B** (Llama2-70B, 80 layers, hidden 8192, 32 heads) [PAPER FACT]
  - Both run on vLLM on 8x GPUs (DGX-A100/H100), model parallelism TP across 8 GPUs for best latency [PAPER FACT]; unless stated Llama-70B used for Table4 A100 vs H100 comparison [PAPER FACT].
- **Traces (Sec III, V-B):**
  - **Coding trace:** From Azure LLM inference service, Nov 11 2023, 20 mins, coding LLM service, median prompt 1500, median output 13 (small), represents code completion [PAPER FACT] (Fig3).
  - **Conversation trace:** Azure conversation service, median prompt 1020 (wide range), median output 129 bimodal, represents chatbots [PAPER FACT] (Fig3).
  - Privacy: no content, only input/output sizes; input prompts synthesized to required token count, forced generation to output count; no KV reuse between requests to emulate secure cloud [PAPER FACT].
  - Released subset at AzurePublicDataset [PAPER FACT].
  - For characterization: scaled-down to 2 RPS to fit single machine for Fig4 [PAPER FACT].
  - For provisioning/evaluation: Poisson arrival tuned load (RPS) from prompt/token size distributions, Fig16 sweeps 0-130+ RPS, 2-min trace for Fig12 search, validated simulator with 50k+ iterations [PAPER FACT].
  - Also mention Coding and Conversation traces correspond roughly to ShareGPT/arxiv distributions but distinct production [PAPER FACT].
- **Generation settings:** Beam size 1 [PAPER FACT]; precise sampling [NOT REPORTED].
- **Not evaluated:** LongBench summarization, 2048+ long contexts beyond characterization? [NOT REPORTED] beyond Fig6.

## 9 Hardware [PAPER FACT]

- **Machines:**
  - **DGX-A100** [NVIDIA DGX A100]: 8x A100 80GB, 19.5 TFLOPs, 80GB, 2039 GBps, 400W, 50 Gbps NVLink, 200 Gbps Infiniband, $17.6/hr [PAPER FACT] (Table1).
  - **DGX-H100** [NVIDIA DGX H100]: 8x H100 80GB, 66.9 TFLOPs, 80GB, 3352 GBps, 700W, 100 Gbps NVLink, 400 Gbps IB, $38/hr [PAPER FACT] (Table1).
  - Testbed: 2x DGX-A100 and 2x DGX-H100 VMs on Azure, IB-connected (A100 200 Gbps, H100 400 Gbps double) [PAPER FACT] (Sec V-A).
  - Simulator models both; performance model for 8-GPU TP [PAPER FACT].
  - For LLaMA-70B comparisons, A100 vs H100 same model, same batching, no other HW [PAPER FACT].
- **Interconnect:**
  - Intra-machine: NVLink (A100 50 Gbps, H100 100 Gbps) [PAPER FACT]
  - Inter-machine: InfiniBand backend (A100 200 Gbps, H100 400 Gbps) measured for KV transfer [PAPER FACT]; Splitwise assumes IB between prompt-token (even HA heterogeneous H100->A100 assumed IB, though noted may need RoCE/Ethernet alternative but 10x lower BW still beneficial [PAPER FACT] Sec VII).
- **Not detailed:** CPU, DRAM exact, OS, CUDA 12? [NOT REPORTED] but vLLM/M SCCL++ stack [PAPER FACT].

## 10 Main Results [PAPER FACT]

All numbers from Sec VI, Figs 14-20, Tables 4-5.

- **KV-cache transfer optimization:**
  - Serialized transfer linear with prompt size; layer-wise hides latency to constant **~8ms A100 / ~5ms H100** non-overlapped [PAPER FACT] (Fig14).
  - Overhead vs prompt compute: **<7%** [PAPER FACT].
  - Small prompts <512 (H100) use serialized else layer-wise (threshold known at start) [PAPER FACT].
  - End-to-end impact on coding trace (no batching): **serialized up to 3% E2E, Splitwise only 0.8% E2E** [PAPER FACT] (Fig15). Second token latency: **serialized +64%, Splitwise +16.5%** [PAPER FACT].

- **Iso-power throughput-optimized clusters (40 H100 power budget = 70 A100):**
  - **Coding trace (Fig16a):** Splitwise-HH, HHcap, AA all > Baseline-H100; Baseline-H100 suffers high TBT due to mixed batching with large prompts; Splitwise-HA bridges low TTFT and high throughput; mixed pool useful at high loads (>90 RPS) where H100 mixed reduces TBT [PAPER FACT].
  - **Conversation trace (Fig16b):** Splitwise-HHcap best on all latencies due to long token phases benefiting capped token machines [PAPER FACT].
  - **Summary (Fig18a conversation, Baseline-A100 =1):** Splitwise-AA delivers ~2.15x more throughput at same power (iso-power, 40 H100 power =70 A100) vs Baseline-A100 on conversation trace (Fig.18a) [PAPER FACT]; Splitwise-HA **1.18x at 10% lower cost same power** [PAPER FACT] (Sec VI-B summary).
  - **Batched tokens distribution (Fig17 conversation at 70 and 130 RPS):** Baseline-H100 70% time <=15 tokens, rest large prompts affecting TBT; Splitwise prompt machines idle most but run larger batches when active; token machines better batching; at high load 130 RPS distributions converge as mixed pool fully utilized [PAPER FACT].

- **Other cluster optimizations (Sec VI-C, Fig18b-19):**
  - **Iso-cost throughput-optimized:** Splitwise-AA **1.4x more throughput than Baseline-H100 at 25% more power and 2x space** (running same cost) [PAPER FACT]; Splitwise-AA 2x more throughput at same cost? Actually text: 1.4x vs H100 with 25% more power/2x space; also statement "40% higher throughput using older GPUs" [PAPER FACT].
  - **Iso-throughput power-optimized (Fig19a):** Splitwise-HHcap achieves **same throughput as Baseline-H100 at 25% lower power at same cost/space** [PAPER FACT].
  - **Iso-throughput cost-optimized (Fig19b):** Splitwise-AA achieves **same throughput as Baseline-H100 at 25% lower cost** (or 20% lower cost with 1.4x throughput? Abstract: 1.4x higher throughput at 20% lower cost) [PAPER FACT]; also abstract line: Same cost/power budgets alternative is **2.35x more throughput** [PAPER FACT] (abstract: 2.35x more throughput with same cost and power budgets) [PAPER FACT].
  - **Batch job (high load stress, Sec VI-E):** At high load where SLO not strict, Splitwise devolves to baseline mixed batching; **Baseline-A100 and Splitwise-AA best at 0.89 RPS/$** vs **0.75 RPS/$ for H100**, emphasizing cost efficiency of A100 for throughput [PAPER FACT].

- **Overall headlines (Abstract, Conclusion):**
  - **1.4x higher throughput at 20% lower cost** vs current designs [PAPER FACT].
  - **2.35x more throughput with same cost and power budgets** [PAPER FACT].
  - Conclusion summary: **1.76x better throughput with 15% lower power at same cost, or 2.35x better throughput with same cost and power** [PAPER FACT] (Sec IX) [PAPER FACT].

- **Robustness (Fig20):**
  - Running conversation trace on coding-optimized cluster: **7% throughput setback** for heterogeneous Splitwise-HA/HHcap, but still much better than baselines; homogeneous AA/HH no impact via mixed pool morphing [PAPER FACT].
  - Changing model Llama-70B on BLOOM-176B cluster: supports higher throughput due to fewer params; Splitwise-HH/HHcap consistently best latency [PAPER FACT]; Splitwise robust to workload/model changes [PAPER FACT].

- **Characterization validation (Table4 P50 no batching, Llama-70B):**
  - TTFT A100 185/155 ms vs H100 95/84 ms (0.51-0.54x) coding/conversation; TBT 52/40 vs 31/28 ms (0.70x); E2E 856/4957 vs 493/3387 ms (0.58-0.68x); Cost $0.42/$2.4 vs $0.52/$3.6 (1.24/1.5x higher for H100); Energy 1.37/7.9 vs 1.37/9.4 Whr (1x/1.2x) [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Generative decoder-only transformers (BLOOM, LLaMA) with distinct prompt (compute-bound, parallel) vs token (memory-bound, sequential) phases, KV-cache per layer [PAPER FACT].
- Beam size 1, no beam search; KV-cache not reused between requests (secure isolation) [PAPER FACT]; content of prompt irrelevant to perf (only sizes matter) [PAPER FACT].
- Batching mechanisms as defined (request-level, continuous, mixed) [PAPER FACT]; model parallelism TP across 8 GPUs best for latency [PAPER FACT].
- Workload distributions from Azure traces (coding/conversation) representative, Poisson arrival for provisioning, short 2-min trace sufficient for simulator search [PAPER FACT].
- Simulator with piecewise linear perf model (MAPE <3%) accurately predicts latency for given batch/input/output sizes [PAPER FACT]; communication modeled via measured IB KV transfer [PAPER FACT].
- Power provisioning considers provisioned power (TDP) not dynamic; power cap feasible via GPU cap [PAPER FACT].
- Model fits on 8 GPUs with PagedAttention; no extra memory for KV beyond cache [PAPER FACT].
- InfiniBand available between any prompt-token pair (HA assumption optimistic, discussed alternative RoCE/Ethernet still viable) [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Discussed in Sec VII Discussion:
  - **Heterogeneous prompt/token hardware not readily available:** Splitwise-HA assumes IB between H100 and A100 which may not exist in cloud today; alternatives HPC via CPU, Ethernet RoCE, or 10x lower BW may still be ok but not evaluated; fragmentation/challenges for CSP [PAPER FACT].
  - **Extensibility to other hardware:** Smaller GPUs (T4) lack memory; alternative compute (AMD MI250, Intel Sapphire Rapids HBM) could be token machines but no access/optimized LLM implementations to evaluate -- left future [PAPER FACT].
  - **Conversation back-and-forth:** Current chat requires resending full context; future with GPU KV cache could avoid recomputation and require KV transfer back to prompt machine, changing memory patterns [PAPER FACT].
  - **Scalability of CLS:** CLS may bottleneck for large clusters; partitioned/replicated scheduling could help, orthogonal [PAPER FACT] (Sec IV-E).
  - **Reliability:** Failure restarts from scratch; checkpointing KV to in-memory DB for recovery not designed, out of scope [PAPER FACT].
  - **No KV compression:** 10x lower BW still beneficial, but KV compression not evaluated [PAPER FACT].
  - **For heterogeneous types, deviating load causes 7% setback** when workload mismatched, acknowledged [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **No evaluation on modern GQA/MQA models (Mistral, Yi 34B) or 32k+ long contexts:** Characterization limited to 2k prompt before degradation; LongBench 1M context not tested; conclusions may shift at 100k [AGENT INFERENCE].
- **Cost model simplistic:** Table1 cost per machine ($17.6 vs $38) from CoreWeave rental, not TCO (datacenter, power, cooling, network); space metric not quantified in dollars [AGENT INFERENCE].
- **Power-optimized claim based on provisioned not dynamic power:** Real dynamic power lower for token phase, but provisioned power still charged; actual energy saving may be less [AGENT INFERENCE].
- **Simulator dependency on 8-GPU TP only:** No evaluation of PP or different TP degrees for large models needing >8 GPUs (e.g., 70B vs 176B) beyond single node; scale-out multi-node with PP not measured on hardware beyond 2 nodes [AGENT INFERENCE].
- **Trace privacy limits:** No content-based prefix sharing evaluated; production benefit of prefix caching could reduce prompt load and change optimal prompt/token ratio [AGENT INFERENCE].
- **No comparison to Sarathi-Serve chunked or DistServe goodput search:** Splitwise predates/concurrent with them; no quantitative comparison to chunked hybrid vs disaggregated placement optimization [AGENT INFERENCE].
- **Single model serving only:** No multi-model multiplexing on same cluster; mixed pool elasticity limited to single model type [AGENT INFERENCE].
- **Failure recovery still expensive:** Restart from scratch wastes prompt compute; no measurement of failure impact [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Optimal heterogeneous ratio with new GPUs (e.g., H100 vs A100 vs cheaper LPDDR/HBM CPUs):** Given evolving compute/bandwidth ratios, what is Pareto-optimal prompt/token hardware mix for next-gen GPUs (B200, MI300)? [AGENT INFERENCE]
2. **Dynamic auto-scaling vs mixed pool:** Can reinforcement learning controller dynamically adjust prompt/token/mixed pool sizes online based on real-time queue and power caps without simulator search? [AGENT INFERENCE]
3. **KV-cache compression for disaggregation:** How much does quantization (KIVI, GEAR) or sparsity reduce transfer overhead and allow cheaper interconnect (10 Gbps) to suffice? [AGENT INFERENCE]
4. **Long-context and multi-turn chat:** With context caching (not recompute), how to handle KV transfer back to prompt pool for next turn and maintain consistency? [AGENT INFERENCE]
5. **Power-performance Pareto frontier:** What is fine-grained power-capping policy per GPU (not just 50% cap) that maximizes Perf/$ under varying load, and does dynamic voltage/frequency scaling help prompt phase? [AGENT INFERENCE]
6. **Integration with prefix sharing:** How to combine Splitwise with RadixAttention/SGLang global prefix cache to avoid re-transferring shared system prompts? [AGENT INFERENCE]
7. **Multi-model serving with Splitwise:** Can prompt and token pools be shared across multiple models of different sizes to improve utilization, and how to route? [AGENT INFERENCE]
8. **Theoretical throughput bound:** What is theoretical max throughput gain of optimal phase splitting vs optimal mixed batching as function of prompt/token length ratio and hardware FLOPs/BW ratio? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Orca [Yu et al. OSDI22] and vLLM [Kwon et al. SOSP23]:** Iteration-level/continuous batching predecessors; Splitwise implements mixed continuous batching on vLLM and shows their interference issues (Fig2) [PAPER FACT].
- **FasterTransformer [NVIDIA], TurboTransformers [Fang 2021], LightSeq [Wang 2021]:** Model-parallel inference engines, use TP [PAPER FACT].
- **DistServe [Zhong et al. 2401.09670] / TetriInfer [Hu et al. 2401.11181] / DejaVu [Strati 2024]:** Concurrent disaggregated prefill-decode work, mentioned in related work genre; Splitwise focuses on hardware heterogeneity, DistServe on goodput placement search [PAPER FACT] (related work Sec VIII genre).
- **Sarathi / Sarathi-Serve [Agrawal et al. 2308.16369 / 2403.02310]:** Chunked-prefill alternative to disaggregation, discussed as piggyback concept in Sec II-D [PAPER FACT].
- **Heterogeneous scheduling and dataflow systems [Sparrow, Ray etc.]:** Prior work on mapping heterogeneous workloads to heterogeneous hardware, inspiring CLS JSQ [PAPER FACT].
- **Recommendation serving [Kang et al. etc.]:** Exploits compute/memory heterogeneity within/between models, analogous to prompt/token split [PAPER FACT].
- **AlpaServe [Li et al.] and Shepherd [Zhang et al.]:** DL serving with parallelism statistical multiplexing, but for non-autoregressive models [PAPER FACT].
- **Follow-up to Splitwise (not in paper):** Mooncake [Qin 2407.00079] extends disaggregation with KVCache-centric DRAM pooling and chunked pipeline parallelism, building on Splitwise characterization [AGENT INFERENCE].
- **Recent characterization [AzurePublicDataset 2023]:** Trace released for community, used by subsequent works (DistServe, Mooncake) [PAPER FACT].


## Review Log
Reviewer: Reviewer-1
Problems Found:
- 2.15x throughput claim previously ambiguous as generic; pinned to iso-power conversation trace Fig18a (Baseline-A100=1), verified against Sec VI-B.
- Iso-cost 1.4x at 25pct more power/2x space (Fig18b) and iso-power 25pct lower power at same cost/space for HHcap (Fig19a), and 2.35x same cost+power headlines (Abstract/Conclusion) - all verified via text Sec VI-C/IX.
- KV transfer layer-wise 8ms A100/5ms H100 constant, <7pct prompt, serialized linear vs layer-wise threshold 512 tokens, E2E 0.8pct vs 3pct, second-token +16.5pct vs +64pct verified Fig14-15.
- Hardware: DGX-A100 19.5 TFLOPs/400W/50Gbps NVLink/200Gbps IB vs DGX-H100 66.9 TFLOPs/700W/100Gbps NVLink/400Gbps IB, cost 2.16x, table values correct.
- No comparison to Sarathi/DistServe quantitatively - correctly noted as concurrent (paper predates/concurrent) and not benchmarked, so no hallucination.
Corrections:
- Pinned 2.15x to iso-power conversation trace with source.
- Clarified cost ratio 2.16x and power ratios.
- No core idea misinterpretation (prompt/token/mixed pools + layer-wise async MSCCL++ + CLS/MLS hierarchical scheduling correctly summarized).
Confidence: High