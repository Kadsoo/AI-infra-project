# Research Tensions — Stage 2B Audit

> A tension is not an unclaimed “gap.” It is a pair of supported forces that cannot both be optimized freely over the same operating range. Each item below states the condition under which the conflict should be observable and a result that would falsify it.

## Evidence standard

- **High:** at least two local sources directly demonstrate opposing operating points or one source directly measures the trade-off.
- **Medium:** a coherent cross-paper inference, but a shared harness or common workload is missing.
- **Do not infer:** the supplied corpus does not establish the conflict strongly enough.

## T1 — KV capacity reduction versus task fidelity

| Element | Audit finding |
|---|---|
| Observation | Eviction, structural compression, and low-bit quantization reduce KV state substantially, but their quality curves differ with task, model, budget, and generation length. |
| Evidence | H2O/StreamingLLM/SnapKV/PyramidKV/SCOPE supply retention results; KIVI and GEAR show low-bit behavior; Stage 2A already records compression-versus-quality as B14. KIVI's 2-bit sensitivity and GEAR's extra low-rank/sparse repair should not be summarized as a single universal “near-lossless 2-bit” result. |
| Breakdown condition | Retrieval-dense, reasoning, MQA/GQA-sensitive, or long autoregressive workloads at aggressive budgets/bit-widths. |
| Observable consequence | Quality/F1/LongBench/PPL degradation, or a latency penalty from larger residuals, higher bits, or repair work. |
| Falsifier | One fixed retention-plus-precision setting remains within a prespecified quality tolerance across the target workload matrix while retaining its throughput/memory advantage. |
| Stage 3 value | High as a **measurement question**; low as a novelty claim until joint baselines are checked. |
| Confidence | High. |

## T2 — Exact-prefix reuse is cheap, but reusable work is often not an exact prefix

| Element | Audit finding |
|---|---|
| Observation | Radix/tree reuse is efficient when token order and prefix match; RAG and multi-history workloads contain useful chunks in other positions or histories. |
| Evidence | SGLang/RAGCache establish exact-prefix gains. CacheBlend, Cache-Craft, and KVLink directly show different repair mechanisms for non-prefix composition. Cache-Craft's production-style evidence makes it incorrect to treat low prefix hit rate as merely an eviction-policy failure. |
| Breakdown condition | Multiple retrieved chunks, reordered documents, or histories that contribute cross-attention. |
| Observable consequence | Cache hit may remain high while output quality falls, or repair/recompute/transfer eliminates the apparent TTFT benefit. |
| Falsifier | Exact-prefix-only reuse gives comparable end-to-end quality and TTFT to repaired non-prefix reuse on a dispersed-chunk workload. |
| Stage 3 value | High; compare validity-aware reuse methods at matched quality and storage, rather than claiming that one reuse mechanism dominates. |
| Confidence | High. |

## T3 — Cache affinity / hit rate versus queueing, fairness, and tail latency

| Element | Audit finding |
|---|---|
| Observation | Routing to cached state avoids prefill, but it concentrates load. Queue time can outweigh saved prefill or transfer time. |
| Evidence | SGLang notes greedy longest-prefix behavior; Mooncake explicitly estimates queue, prefill, and transfer and uses a balancing threshold; KVFlow and Continuum show locality/lifetime policies under competing workflows. The corpus contains limited direct fairness metrics. |
| Breakdown condition | Skewed hot prefixes, multi-tenant queues, burst arrivals, or a strict P95/P99 SLO. |
| Observable consequence | Higher average hit rate/throughput but worse P99 TTFT/TBT or per-class slowdown/starvation. |
| Falsifier | An affinity-only policy preserves the specified tail and fairness target over a skew/burst sweep without added replication or admission control. |
| Stage 3 value | Strong systems question, but current evidence supports the **trade-off** more directly than a particular missing algorithm. |
| Confidence | Medium. |

## T4 — P/D disaggregation removes interference but creates data-movement and duplication costs

| Element | Audit finding |
|---|---|
| Observation | Separating prefill from decode improves phase-specific utilization and SLO attainment, while KV movement, topology constraints, and duplicated model state can erase the benefit. |
| Evidence | DistServe and Splitwise demonstrate P/D benefits under modeled placement; Sarathi demonstrates the colocated alternative; FlowKV and Mooncake demonstrate transfer-path sensitivity. Stage 2A's “transfer negligible” and “transfer dominates” claims are compatible only after conditioning on topology and implementation. |
| Breakdown condition | Cross-node/low-bandwidth placement, fragmented KV, long prompts, rapid load changes, or insufficient transfer overlap. |
| Observable consequence | TTFT/TPOT tail rises, transfer becomes a material E2E fraction, or the same goodput requires more memory/GPUs. |
| Falsifier | A P/D deployment has no material transfer or duplication cost across the topology and prompt-length range under test. |
| Stage 3 value | High if framed as a conditional operating-boundary experiment, not as “P/D is universally better.” |
| Confidence | High. |

## T5 — More capacity in lower tiers versus slower and less predictable access

| Element | Audit finding |
|---|---|
| Observation | CPU DRAM, SSD, CXL, and remote pools relax HBM capacity; their access path can be slower, fragmented, non-coherent, or congested. |
| Evidence | FlexGen, InfiniGen, ShadowKV, LMCache, DynamicPlacement, and Beluga span materially different tiers and hardware. Beluga is direct evidence that an RDMA control/data path is not interchangeable with a CXL path; it is not proof that CXL wins generally. |
| Breakdown condition | Fine-grained reads, small transfer units, high concurrency, root-complex contention, heterogeneous interconnects, or short contexts. |
| Observable consequence | Tier hit rate increases but TTFT/TPOT or P99 worsens; realized bandwidth is far below nominal; recomputation becomes cheaper than fetch. |
| Falsifier | A lower-tier policy maintains its predicted throughput and tail benefit across transfer granularity and concurrency sweeps. |
| Stage 3 value | High impact but hardware-expensive; a trace-driven or emulated first pass is appropriate. |
| Confidence | High for the existence of the trade-off; Medium for any specific CXL/RDMA boundary. |

## T6 — Static/offline tuning versus non-stationary workload and hardware behavior

| Element | Audit finding |
|---|---|
| Observation | Many systems expose budgets, chunk sizes, cache-admission thresholds, placement thresholds, or cost models that are profiled offline. Some later work is adaptive, so “all policies are static” is an overstatement. |
| Evidence | Sarathi's budget, PyramidKV's shape, KIVI/GEAR parameters, CacheBlend's recompute threshold, HotPrefix admission, Mooncake balancing threshold, and DynamicPlacement's offline search are locally documented. SCOPE, Mooncake, KVFlow, and Continuum already introduce some adaptive or predictive control. |
| Breakdown condition | Workload mix changes, queue burst, prompt/output distribution shift, different model/head pattern, or topology change. |
| Observable consequence | Quality, hit rate, throughput, or P99 performance regresses relative to the best per-condition oracle; control overhead may consume the saved work. |
| Falsifier | A fixed configuration stays within a small, declared oracle gap across the intended workload/topology matrix. |
| Stage 3 value | High researchability if the experiment separates the value of adaptation from prediction overhead and instability. |
| Confidence | High for parameter sensitivity; Medium for a broadly unsolved online-control gap. |

## T7 — Throughput/utilization versus interactive latency and tail SLOs

| Element | Audit finding |
|---|---|
| Observation | Larger or more mixed batches raise utilization/throughput yet can delay a decode step or a short request behind a long prefill. |
| Evidence | Orca, Sarathi-Serve, FastServe, DistServe, and Mooncake use different SLO definitions and objectives. Sarathi's chunking makes the trade-off explicit; DistServe/Mooncake optimize goodput under their own SLOs. |
| Breakdown condition | High arrival rate, a mix of prompt lengths, aggressive batching, or a tighter percentile target. |
| Observable consequence | Mean throughput improves while TTFT/TBT P95/P99 or SLO attainment worsens. |
| Falsifier | A batching strategy improves throughput without a meaningful change in the pre-registered tail metric over the tested load range. |
| Stage 3 value | Essential evaluation tension; it is not evidence that any throughput result is invalid. |
| Confidence | High. |

## T8 — Long-context scale versus local importance assumptions

| Element | Audit finding |
|---|---|
| Observation | Attention-sparsity, sink, or local-observation policies can work well under particular prompts, but the required information may be dispersed or phase-dependent. |
| Evidence | StreamingLLM, H2O, SnapKV, PyramidKV, and SCOPE make different retention assumptions; `assumption_map.md` A9/A10/A12 records their scope. |
| Breakdown condition | Whole-document summarization, dense copy/needle tasks, instruction-at-front prompts, or long generation after a prefill-only selector. |
| Observable consequence | Retrieval/quality deteriorates despite a favorable memory curve, or a fixed cache wastes state that a phase-aware policy would replace. |
| Falsifier | A local-selector policy retains quality at the same memory budget across retrieval, summarization, and reasoning workloads. |
| Stage 3 value | Worth a controlled benchmark; a new selector is not yet justified by this evidence alone. |
| Confidence | High for conditionality; Medium for a general failure rate. |

## T9 — Workflow-aware reuse versus uncertain future workflow state

| Element | Audit finding |
|---|---|
| Observation | Agent/tool workflows create opportunities to retain and prefetch state across gaps, but future branches, tools, and output lengths are uncertain. |
| Evidence | KVFlow's step graph and Continuum's TTL give direct lifecycle evidence; both still depend on workflow/prediction assumptions and use bounded workloads. |
| Breakdown condition | Dynamic fan-out, loop/retry behavior, inaccurate dependency graph, long-tail tools, or high workflow concurrency. |
| Observable consequence | Incorrect prefetch/retention consumes bandwidth/memory, creates queueing, or fails to improve job completion time. |
| Falsifier | A workflow-agnostic LRU/TTL matches a graph-aware policy on branching/looping traces at equal memory/bandwidth budgets. |
| Stage 3 value | Promising but currently medium-confidence because the supplied corpus has limited real branching/looping evidence. |
| Confidence | Medium. |

## T10 — Local microbenchmark improvements versus distributed scalability and reliability

| Element | Audit finding |
|---|---|
| Observation | A single-node or small-cluster throughput result does not directly imply stable performance under a shared cache pool, controller contention, failure, or incast. |
| Evidence | Mooncake, LMCache, FlowKV, Beluga, and DynamicPlacement document limited topology/scale boundaries. DéjàVu directly studies a single fail-stop model in pipeline-parallel serving, so “no work addresses fault tolerance” is false. |
| Breakdown condition | Hundreds of instances, hot-key replication, controller bottleneck, multiple failures, or network partitions. |
| Observable consequence | Goodput/p99 drops, replica/metadata overhead grows, or recovery disrupts active requests. |
| Falsifier | A system's controller, cache, and recovery path preserve the target SLO across a declared scale/fault sweep. |
| Stage 3 value | First a scalability/reliability evaluation question; solution novelty is unverified. |
| Confidence | Medium. |

## T11 — Reported gains are not automatically comparable across papers

| Element | Audit finding |
|---|---|
| Observation | Capacity, throughput, goodput, and SLO attainment figures use different models, traces, GPU/topology, percentiles, and baselines. |
| Evidence | `evaluation_map.md` records heterogeneous metric definitions; `contradictions.md` already resolves apparent conflicts through workload/hardware/metric conditioning. |
| Breakdown condition | Any cross-paper ranking that treats an isolated speedup as an apples-to-apples system comparison. |
| Observable consequence | A claimed winner changes when the same trace, SLO, hardware, and quality constraint are applied. |
| Falsifier | Results retain the ordering in a matched-harness comparison. |
| Stage 3 value | A benchmark/evaluation tension, not proof that a new serving mechanism is needed. |
| Confidence | High. |

## T12 — Individually compatible optimizations may not compose additively

| Element | Audit finding |
|---|---|
| Observation | The corpus often calls paging, compression, reuse, tiering, transfer, and scheduling “orthogonal,” but their overheads and control decisions share memory, bandwidth, and tail budget. |
| Evidence | KIVI/PyramidKV explicitly motivate token-by-bit combination; CacheBlend/Craft pair reuse with recompute/transfer; FlowKV/LMCache show data-layout choices affect transfer; `limitation_map.md` L2 notes missing joint evaluation. |
| Breakdown condition | Stacking selection, low-bit kernels, remote placement, and cache-affinity routing under one SLO. |
| Observable consequence | Gains are sub-multiplicative, quality or P99 regresses, or one mechanism changes the cost model assumed by another. |
| Falsifier | A factorial ablation shows independent, additive gains within measurement error. |
| Stage 3 value | Strong motivation for a compositional evaluation matrix; it does not by itself imply a new compound algorithm. |
| Confidence | Medium. |

## What is deliberately not called a tension

- “CXL is better than RDMA” — only a topology- and implementation-conditioned comparison is supported.
- “Static policies never adapt” — later systems include adaptive/predictive components.
- “Fault tolerance is absent” — DéjàVu provides direct asynchronous replication/recovery evidence for a pipeline-parallel failure model.
- “A performance regression proves a research gap” — it may instead be a workload mismatch, an engineering limitation, or an evaluation boundary.

## Traceability

Primary synthesis inputs: `contradictions.md`, `assumption_map.md`, `limitation_map.md`, `evaluation_map.md`, `bottleneck_map.md`, and local notes for Sarathi-Serve, DistServe, Mooncake, FlowKV, Beluga, SGLang, CacheBlend, Cache-Craft, KVLink, KIVI, GEAR, PyramidKV, SCOPE, KVFlow, Continuum, and DéjàVu.
