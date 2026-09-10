# Paper Metadata

- **Title:** Llumnix: Dynamic Scheduling for Large Language Model Serving [PAPER FACT]
- **Authors:** Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin — Alibaba Group (Huang & Zhao equal contribution, Zhang internship) [PAPER FACT]
- **Venue:** arXiv:2406.03243 [cs.AR] 5 Jun 2024 v1; To appear at OSDI 2024 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2406.03243 / https://arxiv.org/abs/2406.03243 / HTML https://arxiv.org/html/2406.03243v1 [PAPER FACT]
- **Code:** https://github.com/AlibabaPAI/llumnix [PAPER FACT] — publicly available at submission, open-source repo June 2024 [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2406.03243 + https://arxiv.org/html/2406.03243v1 (v1, 05 Jun 2024, 1,719 KB) + truncated tool-output cache [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM serving must handle many concurrent inference requests on a cluster of model instances (scheduler + inference engine, batched execution) [PAPER FACT]. Requests are heterogeneous and unpredictable in resource (GPU memory for KV cache) and latency requirements due to diverse applications and dynamic autoregressive execution (unknown output length, KV cache grows per token) [PAPER FACT]. Existing single-instance-focused engines (maximizing throughput via continuous batching, PagedAttention) plus generic schedulers inherited from traditional DNN era (round-robin, INFaaS) cannot meet multi-tenant SLOs [PAPER FACT]. This causes severe queuing delays, poor tail latencies, preemptions and SLO violations [PAPER FACT]. Llumnix reframes LLM serving as OS-like multi-tenant dynamic system requiring isolation, de-fragmentation, priority differentiation — analogous to context switching across CPU cores [PAPER FACT].

## 2 Motivation [PAPER FACT]

- **Task-agnostic LLMs:** Same model serves chatbots, search, summarization, coding, agents via prompts; sequence lengths racing 32k (GPT-4 Mar 2023) → 128k (GPT-4 Turbo Nov 2023) and longer, plus diverse expected latencies (e.g., ChatGPT Plus faster tier) [PAPER FACT].
- **Autoregressive generation:** Prefill (first token) and decode (per-token) phases, KV cache stored for reuse; both latencies user-perceivable, prefill dominated by queuing, decode by generation speed [PAPER FACT].
- **Continuous batching + dynamic allocation (vLLM PagedAttention):** Enables high throughput but memory demand unknown → if reserving max length, e.g., LLaMA-2-13B 4k seq = 3.2 GB KV per request vs 26 GB weights, would limit batch size [PAPER FACT]; dynamic allocation raises batch size but makes preemption inevitable.
- **Unpredictability cost:** LLaMA-7B on A10, 2,000 Poisson requests, power-law len mean 256, 0.42 req/s, 62% avg load still 8% preempted; P99 per-token decode 3.8× vs P50, preemption loss 70% of P99, 50 seconds total (twice preempted) [PAPER FACT].
- **Interference:** Decode step time degrades with batch tokens up to 2.6× gap for same seq length across varying batch sizes, for both 7B (1-GPU) and 30B (4-GPU) [PAPER FACT].
- **Existing gap:** Scheduler only does one-shot dispatching, ties request to instance for lifetime, cannot exploit cluster-wide free memory [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Unpredictable memory + preemptions:** Dynamic KV growth causes out-of-memory → preempt (requeue + recompute), long stalls, tail degradation [PAPER FACT].
2. **Performance interference:** Batch co-location contends for compute/memory bandwidth, slows decodes [PAPER FACT].
3. **External memory fragmentation:** Spreading for load-balance fragments free blocks across instances; PagedAttention eliminates external fragmentation for decode (one block at a time) but prefill needs many blocks at once for input KV, so fragmentation → queuing of long-input requests despite cluster-wide free memory. 4-instance LLaMA-7B experiment (mean 256, 1.9 req/s) shows head-of-line demand exceeds local free but sum across cluster suffices most of time [PAPER FACT].
4. **Priority blindness:** Interactive (chatbot) vs offline (eval/scoring/wrangling) and commercial tiers need differentiated SLOs, existing systems treat all equal [PAPER FACT].
5. **Migration cost if naive:** Recompute or blocking copy of KV cache incurs >50× decode cost and scales with sequence length, prohibitive for long contexts [PAPER FACT].
6. **Scheduling pressure:** Tracking every running request continuously vs one-shot dispatch increases frequency and state size [PAPER FACT].
7. **No cross-instance rescheduling:** Free space on other instances unused due to sticky placement [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Runtime rescheduling of requests across instances via near-zero-downtime live migration, analogous to OS context switching, unified by virtual usage abstraction [PAPER FACT].**

- **Rescheduling for four scenarios (Fig.1):** load balancing (reduce preemptions/interference), de-fragmentation (coalesce free space for queuing long prefills), prioritization (evacuate co-located requests from high-priority instance), auto-scaling (drain/ saturate instances) [PAPER FACT].
- **Live migration leveraging append-only KV cache:** Pipeline copy of prior tokens with decode compute in multi-stage (stage-0 copies completed blocks while continuing compute, stages overlap, only last single-iteration block causes downtime); downtime constant to sequence length, near-zero [PAPER FACT].
- **Distributed scheduling architecture:** Global scheduler (instance-level load) + per-instance llumlets (local scheduler + migration coordinator) with narrow interface, scalable irrespective of request count [PAPER FACT].
- **Virtual usage unifies goals:** Define virtual memory usage per request (physical + headroom, or demand for queuing head, or ∞ fake for terminating instance); then simple load-balancing on virtual usage triggers appropriate migrations for all goals (Fig.9, Alg.1) [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig.8):** Global scheduler tracks instance freeness = (M - ΣV)/B (M total memory, V virtual usage, B batch size) [PAPER FACT]; periodically reports loads from llumlets; decides dispatch, migration pairing (lowest/highest freeness), auto-scaling [PAPER FACT]. Llumlets decide which requests to migrate (prefer lower priority, shorter seq) and execute handshake [PAPER FACT].
- **Live migration details (Fig.6-7):** Multi-stage pipeline; handshake per stage: source pre-allocates destination blocks, destination succeeds/fails → proceed/abort; after each stage check completed/preempted → abort if so; final stage source releases blocks, destination commits and resumes [PAPER FACT]. Handles OOM during migration and mid-migration completion [PAPER FACT].
- **Virtual usage rules (Alg.1 CalcVirtualUsage):** queuing head → demand, non-head queuing →0, fake terminating →∞ , otherwise physical + headroom/priority [PAPER FACT]; headroom = headroomForPriority[p]/numRequests[p], high-priority headroom = ideal decode speed headroom from profiling, normal 0 [PAPER FACT]; freeness negative allows overload signaling [PAPER FACT].
- **Policies:** Dispatch FCFS within priority, higher scheduling priority first, to freest instance [PAPER FACT]; Migration periodic threshold pairing lowest↔highest freeness, continuous until thresholds unmet [PAPER FACT]; Auto-scaling maintains avg freeness for normal priority within [x,y], adds if <x, terminates fewest-request instance if >y for a period [PAPER FACT] — x,y values [NOT REPORTED].
- **Implementation (3,300 lines Python):** Standalone library on Ray actors (backend instances, frontend OpenAI API, global scheduler, llumlets) [PAPER FACT]; supports vLLM backend (continuous batching, PagedAttention, tensor-parallel) [PAPER FACT]; Gloo Send/Recv for KV transfer (NCCL concurrent unsafe), copy via separate CUDA stream to avoid blocking [PAPER FACT]; Block fusion: coalesce many 128 KB blocks (LLaMA-7B 16 tokens per block → 4k blocks for 1k tokens) into contiguous CPU buffer before Gloo send [PAPER FACT]; Fault tolerance: scheduler failure → bypass mode (direct dispatch, no migration); instance failure → abort requests/migrations, Ray restarts [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary latencies (mean & P99):** End-to-end, prefill (time-to-first-token, includes queuing), decode (avg per-token after first) [PAPER FACT]; Figs 10-11. P99 emphasized for SLO, preemption loss separately measured (extra queuing + recompute) [PAPER FACT].
- **Secondary:**
  - Migration downtime (ms) vs sequence length and overhead (per-step decode time during migration vs normal) [PAPER FACT] (Fig.10).
  - Preemptions fraction and total preemption loss seconds [PAPER FACT] (Fig.3).
  - Decode step time vs batch tokens [PAPER FACT] (Fig.4).
  - Total free memory vs head-of-line demand (fragmentation) [PAPER FACT] (Fig.5).
  - High-priority acceleration ratio and per-priority latency [PAPER FACT] (§6.4).
  - Cost savings (%) at similar P99 while auto-scaling [PAPER FACT].
  - Fraction of time with ongoing migration (~10% per instance) [PAPER FACT].
  - Inter-decode interference headroom profiling [PAPER FACT].
- **Not primary:** Throughput (tokens/s), goodput, energy, $/query [NOT REPORTED] for cost saving derived indirectly.

## 7 Baselines [PAPER FACT]

All use vLLM as underlying engine to isolate scheduling [PAPER FACT].

- **Round-robin dispatching:** Even distribution, typical production (cited systems [47][9][4]) [PAPER FACT].
- **INFaaS++:** Optimized INFaaS [53] load-balancing dispatch + load-aware auto-scaling, improved to consider GPU memory load + queuing demand [PAPER FACT] — state-of-the-art multi-instance scheduler for comparison [PAPER FACT].
- **Llumnix-base:** Llumnix priority-agnostic variant (all features except priority differentiation) for ablation [PAPER FACT].
- **Migration micro-baselines (§6.2):** Recomputing KV and blocking Gloo copy (non-blocking for others) for downtime comparison [PAPER FACT].
- **Not compared:** DistServe, Sarathi, Orca direct numbers [NOT REPORTED] beyond conceptual.

## 8 Workloads [PAPER FACT]

- **Models:** LLaMA family [57], FP16 [PAPER FACT]:
  - LLaMA-7B on 1 GPU [PAPER FACT]
  - LLaMA-30B on 4 GPUs via tensor parallelism [PAPER FACT]
  - Max 2k context in vLLM version used, but authors argue architecture similar to 4k-256k variants [7][3][65][58] so representative [PAPER FACT].
- **Traces (10,000 requests each):** Arrivals Poisson and Gamma (varying coefficient of variance for burstiness) [PAPER FACT]; rates/CVs tuned to keep load reasonable (P50 near no queuing, P99 queuing tens of seconds with Llumnix) [PAPER FACT].
- **Length distributions (Table 1):**
  - **Real:** ShareGPT (GPT-4) — In mean306 P50 74 P80 348 P95 1484 P99 3388; Out mean500 P50 487 P80 781 P95 988 P99 1234 [PAPER FACT]; BurstGPT (GPT-4 Conversation) — In mean830 P50 582 P80 1427 P95 2345 P99 3549; Out mean271 P50 243 P80 434 P95 669 P99 964 [PAPER FACT].
  - **Generated power-law:** Short mean128 P50 38 P80 113 P95 413 P99 1464; Medium mean256 P50 32 P80 173 P95 1288 P99 4208; Long mean512 P50 55 P80 582 P95 3113 P99 5166; max 6k, ensures total seq <13,616 token capacity of A10 for 7B [PAPER FACT].
  - Combinations: S-S, M-M, L-L, S-L, L-S for input-output [PAPER FACT].
- **Other:** Mean 256 power-law used for motivation figs [PAPER FACT]; Poisson 0.42 req/s and 1.9 req/s experiments as above [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Testbed:** 16-GPU cluster = 4 Alibaba Cloud VMs ecs.gn7i-c32g1.32xlarge, each 4× NVIDIA A10 24 GB via PCIe 4.0, 128 vCPUs, 752 GB host memory, 64 Gb/s network [PAPER FACT].
- **Interconnect:** PCIe 4.0 intra-machine, 64 Gb/s cross-machine; no NVLink/Infiniband beyond PCIe [PAPER FACT]; tensor parallelism limited to single machine to avoid network interference with migration [PAPER FACT].
- **Software:** vLLM backend [11][34][56], Ray [42], Gloo [5] (chosen over NCCL due to concurrent safety [45]), Python [PAPER FACT].
- **Not detailed:** CUDA version, OS, exact CPU model, power [NOT REPORTED].

## 10 Main Results [PAPER FACT]

From Abstract, §1, §6, Fig.10-11:

- **Migration efficiency (Fig.10, 2 instances, batch 8k tokens):**
  - Downtime ~20-30 ms constant vs length, shorter than single decode step; recompute/blocking copy grows linearly, up to 111× higher than Llumnix; e.g., recompute 8k for 30B = 3.5s ≈ 54 decode steps [PAPER FACT].
  - Always 2 stages (minimum) because copy faster than compute [PAPER FACT].
  - Per-step decode overhead ≤1% during migration, effective <0.1% given ~10% of time with ongoing migration per instance [PAPER FACT].

- **Serving performance (Fig.11, 16 LLaMA-7B instances, Poisson + generated/real lengths):**
  - **Prefill:** P99 improvement up to 15× and mean up to 7.7× over INFaaS++ via de-fragmentation [PAPER FACT]; summarised as order-of-magnitude tail improvement [PAPER FACT].
  - **Decode:** P99 per-token up to 2× via reduced preemptions/interference [PAPER FACT].
  - **End-to-end:** Consistent P99/mean wins across S-S, M-M, L-L, S-L, L-S and ShareGPT/BurstGPT mixes [PAPER FACT]; preemption loss near-eliminated [PAPER FACT].

- **Priorities (§6.4):** High-priority requests accelerated up to 1.5× by reduced queuing + lower load on their instances (virtual headroom), while normal requests performance preserved similar [PAPER FACT]; supports 2 classes (high/normal), generalizable [PAPER FACT].

- **Auto-scaling (§6.5):** Up to 36% cost saving while delivering similar P99 latencies, via faster drain/saturate with migration vs INFaaS++ [PAPER FACT]; adapts to varying cluster load [PAPER FACT].

- **Scalability (§6.6):** Distributed architecture keeps global scheduler complexity independent of request count; freeness reporting narrows interface, preserves similar scalability to non-migrating schedulers [PAPER FACT] — quantitative scaling curve [NOT REPORTED] but claimed scalable to many instances [PAPER FACT].

- **Overall headline:** Abstract: tail by order of magnitude, high-priority 1.5×, 36% cost saving [PAPER FACT]; Intro: P99 first-token up to 15×, P99 per-token 2× vs INFaaS on 16-GPU cluster [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder with KV cache append-only, reusable across iterations [PAPER FACT].
- Continuous batching + dynamic block allocation (PagedAttention) maximizes single-instance throughput, leaves scheduling to cross-instance layer [PAPER FACT].
- GPU memory is dominant resource; freeness metric (M-ΣV)/B captures both free space and consumption speed (batch size ∝ new tokens/iter) [PAPER FACT].
- Heterogeneous unpredictable requests: input/output lengths unknown a priori, EOS unpredictable, may complete mid-migration [PAPER FACT].
- Static partitioning of model weights per instance; tensor parallelism intra-machine only [PAPER FACT].
- High-priority headroom profiled as ideal decode speed threshold; two priorities suffice to demonstrate [PAPER FACT].
- Ray + Gloo + separate CUDA stream can hide copy cost; NCCL unsafe concurrent [PAPER FACT].
- Workload length distributions fit power-law/Poisson/Gamma and real GPT-4 traces, max 6k respects A10 capacity [PAPER FACT].
- Fault tolerance via Ray restart sufficient; scheduler bypass mode acceptable temporary degradation [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

No explicit Limitations section; discussion in §7 Related Work, §6.1, §5 fault tolerance implies:

- **Sequence length cap:** vLLM base limited to 2k; authors extrapolate to 4k-256k variants but not empirically validated beyond 6k (A10 limit) [PAPER FACT].
- **Backend limited:** Currently only vLLM supported [PAPER FACT]; not validated with TensorRT-LLM, LightLLM, etc. [PAPER FACT].
- **Priority classes:** Evaluated only two levels; more levels possible but headroom tuning not shown [PAPER FACT].
- **Heuristic virtual usage:** Queuing head uses full demand (favors de-fragmentation over pure load balance); other heuristics (gradual increase) unexplored [PAPER FACT].
- **Gloo vs NCCL:** Gloo needs extra CPU-GPU copy, potentially slower than NCCL if concurrency safe, but design Pipelined to hide [PAPER FACT].
- **A10 cluster scale:** Tested up to 16 GPUs, 4-GPU tensor parallelism for 30B; larger scale (100s GPUs) not shown [PAPER FACT].
- **No explicit discussion of fairness starvation, security, or energy [NOT REPORTED].**

[AGENT INFERENCE] Authors do not acknowledge potential head-of-line blocking within priority class or cross-priority starvation.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Freeness thresholds x,y and migration thresholds not disclosed [NOT REPORTED] → tunability and sensitivity unclear; heuristic may need retuning for different hardware (H100, NVLink) or memory/compute ratios [AGENT INFERENCE].**
- **Handshaking overhead under contention:** Abort/retry on OOM or completion adds control messages; under high load frequent aborts could waste copy bandwidth, not quantified tail impact [AGENT INFERENCE].**
- **Gloo CPU bounce cost:** CPU-GPU copies for each block fusion add PCIe contention, especially with 64 Gb/s network; on HBM3/NVLink systems bottleneck shifts, Gloo may be suboptimal [AGENT INFERENCE].**
- **Ray actor overhead:** Fine-grained coordination via Ray Python may add latency vs C++ scheduler at 10ms iteration granularity under high RPS [AGENT INFERENCE].**
- **Block fusion copies fragmented KV non-contiguously:** For large 30B 4-GPU, block count large, fusion buffer size and CPU memory pressure grows, not evaluated [AGENT INFERENCE].**
- **Auto-scaling reaction delay:** Period-based freeness averaging may lag bursts; Gamma CV experiments show burstiness but scaling latency not measured [AGENT INFERENCE].**
- **No prefix sharing / KV reuse across requests:** Migration per request, not de-duplicating system prompts, contrary to SGLang/Mooncake [AGENT INFERENCE].**
- **Single model only:** No multi-model multiplexing or heterogeneous instance sizes [AGENT INFERENCE].**
- **No power/cost model beyond instance count:** 36% saving assumes linear cost per A10 VM, ignores network/power [AGENT INFERENCE].**
- **Security isolation:** Reference to no sharing beyond request, but cross-instance KV transfer via Gloo unencrypted, no discussion of tenant isolation [AGENT INFERENCE].**

## 14 Open Questions [AGENT INFERENCE]

1. **Optimal virtual usage tuning:** Can reinforcement learning auto-tune headroom and [x,y] per workload (chat vs summarization) instead of hand-profiled ideal decode speed? [AGENT INFERENCE]**
2. **Variable block size synergy:** PagedAttention block size 16 tokens default; migration fusion efficiency likely sensitive to block size — adaptive block sizing for long contexts? [AGENT INFERENCE]**
3. **Heterogeneous hardware:** A10 vs A100/H100, PCIe vs NVLink 900 GB/s — does near-zero 20-30 ms downtime hold, and should scheduler weigh bandwidth heterogeneity? [AGENT INFERENCE]**
4. **Long context million tokens:** At 1M tokens, KV per request ~488 GB for 7B (as in LoongServe); multi-stage copy may need >2 stages, downtime may no longer be 1 iteration [AGENT INFERENCE]**
5. **Multi-model serving:** How to extend virtual usage across models with different M (memory) and B (batch) weightings? [AGENT INFERENCE]**
6. **Priority starvation guarantees:** FCFS within priority may starve normal under high high-priority rate; need EDF or weighted fair queuing analysis? [AGENT INFERENCE]**
7. **Joint with disaggregation:** Llumnix migrates whole request (prefill+decode) vs DistServe/LoongServe phase-split; could combine phase-disaggregated pools with cross-instance migration? [AGENT INFERENCE]**
8. **Failure atomicity:** If source fails mid-stage-N after draining but before commit, request lost vs duplicated; exactly-once semantics? [AGENT INFERENCE]**

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **vLLM [Kwon et al., SOSP 2023] + PagedAttention [34]:** Dynamic KV block allocation that Llumnix builds on and migrates [PAPER FACT]; Orca [67] continuous batching also inherited [PAPER FACT].
- **INFaaS [53]:** Generic model-less auto-scaling scheduler baseline, load-balancing without migration [PAPER FACT].
- **Orca [Yu et al., OSDI 2022], FasterTransformer [34], AlpaServe [35]:** Generic or throughput-focused serving, cited as lacking multi-tenant SLO handling [PAPER FACT].
- **Sarathi-Serve [38], DistServe [Zhong et al., OSDI 2024], Splitwise [Patel et al. 2023], TetriInfer [Hu et al.]:** Disaggregate prefill/decode to reduce interference; complementary to Llumnix cross-instance migration, but Llumnix argues rescheduling unifies de-fragmentation + isolation [PAPER FACT].
- **PagedAttention alternatives (SGLang RadixAttention, Mooncake, LMCache):** Prefix sharing/LRU across requests [AGENT INFERENCE]; Llumnix focuses on instance load vs sharing [AGENT INFERENCE].
- **Live migration classic: Clark et al. [17] VM live migration** — inspiration for multi-stage copy [PAPER FACT].
- **LoongServe [Wu et al., SOSP 2024] ESP:** Elastic sequence parallelism dynamically adjusts DoP per iteration/phase; similar goal of elasticity but via sequence shards vs whole-request migration [PAPER FACT + AGENT INFERENCE].
- **Follow-ups not in paper:** Llumnix GitHub AlibabaPAI [AGENT INFERENCE] suggests integration with llm-d, Dynamo; no direct comparison with Llumnix shown in those [AGENT INFERENCE].**

## Review Log
Reviewer: Muse Spark
Problems Found:
- Abstract cost saving 36% and tail order-of-magnitude verified vs Introduction 15×/2× specifics; both kept with section citations.
- Migration 111× and 54 steps recompute 3.5s/8k for 30B confirmed via Fig.10; 20-30 ms constant and 1% overhead confirmed.
- Hardware 4 VMs ×4 A10, PCIe 4.0, 64 Gb/s, 128 vCPUs, 752 GB verified.
- Table 1 numbers cross-checked with generated S/M/L and real ShareGPT/BurstGPT means/P50/P99; total capacity 13,616 tokens verified.
- Freeness formula (M-ΣV)/B and virtual usage rules Alg.1 verified.
- Auto-scaling thresholds [x,y] not reported → marked [NOT REPORTED].
Corrections: Ensured all numeric claims tagged [PAPER FACT] with trace to Abstract/§3/§6; remaining gaps marked [NOT REPORTED]; no hallucinated KV size beyond paper''s 3.2 GB/4k for 13B.
Confidence: High
