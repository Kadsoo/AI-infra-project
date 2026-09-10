# Stage 3 Research-Question Candidates — Stage 2B

> **Scope:** These are experimentable questions distilled from the supplied corpus. They are not solution proposals and **none carries a confirmed novelty claim**. “Novelty potential” is only a screening signal for a later external novelty search.

## Ranking method

Scores are 1–5. Evidence asks whether the corpus supports the phenomenon; impact asks whether the affected serving objective matters; researchability asks whether a bounded Stage 3 experiment can test it; novelty potential asks only whether the current corpus leaves a plausible unresolved boundary.

| Rank | Candidate | Evidence | Impact | Researchability | Novelty potential | Tier | Why it is retained |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | RQ-1 Conditional P/D placement under topology and burst | 5 | 5 | 4 | 3 | Tier 1 | Strong documented trade-off; clear, low-ambiguity measurements. |
| 2 | RQ-2 Static state-budget failure boundaries | 5 | 4 | 5 | 3 | Tier 1 | Many policies expose fixed budgets; a controlled switch experiment is inexpensive. |
| 3 | RQ-8 Token × bit-width × tier interaction | 4 | 4 | 5 | 3 | Tier 1 | Falsifies the unsupported “independent multiplicative gains” premise before a new controller is proposed. |
| 4 | RQ-4 Cache affinity versus tail/fairness under skew | 4 | 5 | 4 | 3 | Tier 1 | A direct SGLang/FastServe/Llumnix boundary with a compact trace-replay test. |
| 5 | RQ-3 Quality-preserving non-prefix RAG reuse boundary | 4 | 5 | 4 | 4 | Tier 2 | High impact, but Cache-Craft already addresses the core mechanism; retain as a robustness/evaluation boundary. |
| 6 | RQ-5 Workflow-state lifetime under branching/tool uncertainty | 3 | 4 | 3 | 4 | Tier 2 | Promising workload gap, but trace semantics are not yet sufficiently grounded. |
| 7 | RQ-6 Fetch/recompute boundary for heterogeneous tiers | 4 | 5 | 3 | 3 | Tier 2 | Important hardware-aware question; validation cost/topology access is higher. |
| 8 | RQ-7 Scale and failure boundary of shared KV pools | 3 | 4 | 2 | 3 | Tier 2 | Valuable operational boundary, but starts as an engineering/evaluation study. |

# RQ-1 — When does P/D disaggregation lose to colocated execution because KV movement and burst dynamics dominate?

## Observation

P/D disaggregation can remove prefill/decode interference and improve SLO-aware goodput, while transfer, placement, and queue behavior can create a different tail bottleneck. The corpus supports both sides under different topology and workload conditions.

## Existing Approach

DistServe and Splitwise choose phase-specific resources/placement; Sarathi-Serve bounds colocated interference with chunked prefill; Mooncake and FlowKV optimize cache-aware routing/transfer. These approaches use different transfer paths, SLO definitions, and workload models.

## Fragile Assumption / Limitation

The phase split assumes transfer can be bounded or hidden and workload distributions are stable enough for offline provisioning/thresholds. This is fragile under topology heterogeneity or rapid burst changes.

## Failure Scenario

Cross-node/low-bandwidth placement, fragmented KV, mixed prompt lengths, and a burst that turns a cache-affine P/D pairing into a queue hotspot.

## Observable Consequence

- P95/P99 TTFT or TPOT rises even as mean throughput is maintained.
- Transfer time or queue time becomes a material fraction of E2E time.
- A colocated/chunked baseline overtakes the P/D configuration at equal SLO and quality.

## Relevant Papers

Sarathi-Serve (2024), DistServe (2024), Splitwise (2024), Mooncake (2025), FlowKV (2025), Beluga (2025); see `research_tensions.md` T4/T7 and `evaluation_map.md`.

## Why Worth Investigating

It turns a broad architecture debate into a falsifiable operating-boundary question with direct impact on TTFT, TPOT, goodput, and resource cost.

## Cheap Validation

Replay one trace family with controlled prompt/output distributions on one colocated and one P/D baseline. Sweep only (a) transfer latency/bandwidth, (b) burst coefficient of variation, and (c) prompt length. Report TTFT/TPOT P50/P95/P99, goodput, transfer fraction, and GPU allocation.

## Research Potential

Medium. The main Stage 3 value is robustness evidence for existing reuse methods, not a claim that their core mechanism is absent.

## Confidence

High for the conditional phenomenon; Medium for any claim about a previously unknown boundary.

# RQ-2 — How fragile are fixed retention, precision, and chunk budgets when task type or generation phase changes?

## Observation

Retention, compression, and quantization methods often expose static budgets or group/residual parameters. The corpus also shows that attention patterns and quality sensitivity differ by task, layer, model, and decode phase.

## Existing Approach

H2O/StreamingLLM/SnapKV/PyramidKV select or retain state; SCOPE separates prefill/decode behavior; KIVI and GEAR reduce numerical precision. Most published comparisons hold the chosen setting fixed while changing only part of the workload.

## Fragile Assumption / Limitation

A setting selected from an offline average is assumed to remain near-optimal and quality-safe when the workload changes. This does not follow from a good result on a single benchmark.

## Failure Scenario

Switch from retrieval QA to summarization/reasoning, change output length, use a model with MQA/GQA behavior, or move from prefill-selected to long decode state.

## Observable Consequence

- Accuracy/F1/LongBench/PPL crosses a predeclared tolerance at a fixed memory target.
- The setting that maximizes memory saving increases TPOT or kernel overhead.
- A per-condition oracle exposes a substantial loss from the static setting.

## Relevant Papers

H2O (2023), StreamingLLM (2023), SnapKV (2024), PyramidKV (2024), KIVI (2024), GEAR (2024), SCOPE (2025); `assumption_map.md` A8–A12 and `research_tensions.md` T1/T8.

## Why Worth Investigating

The experiment can rule out an overgeneralized claim cheaply and would inform whether adaptation is needed before anyone designs a controller.

## Cheap Validation

Hold implementation and memory budget constant. Test 3–4 existing methods across a small matrix of retrieval, summarization, reasoning, and long-generation tasks. Compare fixed published settings against a small offline per-condition oracle; report quality, HBM, TTFT/TPOT, and overhead.

## Research Potential

High.

## Confidence

High for sensitivity; Medium for a general online-adaptation research gap.

# RQ-3 — What is the quality–latency boundary between exact, repaired, and trainable non-prefix RAG KV reuse?

## Observation

Exact prefix caches are inexpensive but have limited coverage on dispersed RAG histories. Existing non-prefix routes trade selective recomputation, contamination-aware repair, or training cost for higher reuse validity.

## Existing Approach

RAGCache/SGLang use ordered-prefix reuse; CacheBlend selectively recomputes high-deviation state; Cache-Craft estimates multi-history contamination; KVLink uses trainable link tokens after position handling.

## Fragile Assumption / Limitation

The methods rely on different assumptions: a stable ordered prefix, sparse repairable deviation, reliable contamination score, or permissible fine-tuning/knowledge-base stability. Existing results are not matched on one workload, quality threshold, and storage budget.

## Failure Scenario

Increase the number of histories, alter retrieval order, introduce retriever drift, constrain host/SSD bandwidth, or prohibit model fine-tuning.

## Observable Consequence

- A TTFT advantage disappears after enforcing equal quality or storage.
- Recompute percentage, transfer, or training cost rises sharply with history count/order dispersion.
- A method's quality drops on multi-history composition despite a high nominal cache hit rate.

## Relevant Papers

SGLang (2024), RAGCache (2024), CacheBlend (2025), Cache-Craft (2025), KVLink (2025); `research_tensions.md` T2/T12.

## Why Worth Investigating

It isolates a real systems-quality interaction rather than assuming any reuse is valid. It is potentially high impact for RAG serving, but formal novelty must be searched later.

## Cheap Validation

Construct a small replay set with 2, 5, and 10 retrieved chunks; vary ordering and history overlap. Compare exact-prefix, CacheBlend-like repair, and one compositional baseline at a fixed quality tolerance. Report TTFT, throughput, recompute %, cache hit, storage, and F1/ROUGE.

## Research Potential

High.

## Confidence

Medium-High.

# RQ-8 — Do retained-token count, KV bit-width, and storage tier interact non-additively?

## Observation

The corpus contains hybrid reductions already: H2O combines retention with low-bit KV storage, GEAR combines quantization with low-rank/sparse compensation, and ShadowKV combines low-rank state with offload. It therefore does **not** support the claim that every paper optimizes one independent axis. What remains unmeasured in the supplied corpus is whether retained-token count, bit-width, and tier placement compose additively under the same workload and SLO.

## Existing Approach

H2O, KIVI, GEAR, PyramidKV, ShadowKV, FlexGen, and InfiniGen each cover subsets of the axes with different models, metrics, and hardware. Their speedups cannot be multiplied across papers.

## Fragile Assumption / Limitation

Stacked optimizations are treated as approximately orthogonal: reducing bytes should help placement, and reducing tokens should reduce transfer/attention cost. Kernel granularity, residual buffers, approximation error, and fetch latency can invalidate that assumption.

## Failure Scenario

Aggressive token retention combined with low-bit KV and a cold tier, especially for a reasoning/RAG workload where quality repair or small transfers dominate.

## Observable Consequence

- A factorial interaction term is non-zero for TTFT, TPOT, memory, or quality.
- Pareto ordering reverses: the best single-axis setting is not part of the best combined configuration.
- The combined gain is sub-additive or causes a quality/tail regression.

## Relevant Papers

H2O (2023), KIVI (2024), GEAR (2024), PyramidKV (2024), ShadowKV (2025), FlexGen (2023), InfiniGen (2024); `research_tensions.md` T1/T5/T12 and `limitation_map.md` L2.

## Why Worth Investigating

This is a compact way to reject or support Stage 2A's broad joint-optimization claim before investing in a universal controller. A null result is also valuable: it would reclassify the idea as an engineering composition.

## Cheap Validation

Use one 7B model, three retained-token budgets, two precision settings, and two tiers (GPU/host or a calibrated tier emulator). Evaluate two workload types at a fixed quality threshold. Report a 3×2×2 factorial analysis of memory, TTFT, TPOT, throughput, and quality; do not design a new controller.

## Research Potential

High as a Stage 3 validation question; Medium-Low for a novel mechanism until non-additivity is observed.

## Confidence

Medium-High.

# RQ-4 — Can cache affinity improve reuse without violating per-class tail latency or fairness under skewed demand?

## Observation

Cache-aware routing and longest-prefix policies avoid prefill but can create hot nodes or starve shorter/cold requests. The corpus measures cache hit and SLOs more often than it measures their joint fairness frontier.

## Existing Approach

SGLang uses radix/longest-prefix behavior; Mooncake balances queue, prefill, transfer, and replication with a threshold; Sarathi/FastServe address scheduling and tail through different mechanisms.

## Fragile Assumption / Limitation

Cache affinity is assumed to be worth its queue cost, and a global/average SLO is assumed to capture user-level harm. Direct fairness evidence is limited in the local corpus.

## Failure Scenario

Heavy popularity skew, simultaneous requests to a hot prefix, heterogeneous request priorities, or a strict P99 objective.

## Observable Consequence

- Higher cache hit / average throughput but worse P99 or per-class slowdown.
- Replication reduces a hotspot but increases memory/bandwidth enough to hurt another class.

## Relevant Papers

SGLang (2024), Mooncake (2025), Sarathi-Serve (2024), FastServe (2023), KVFlow (2025); `research_tensions.md` T3.

## Why Worth Investigating

It may expose an important control-plane constraint that raw throughput and cache-hit metrics conceal.

## Cheap Validation

Use a skewed prefix trace with two request classes. Compare affinity-only, load-only, and a simple threshold/replication policy. Report hit rate, aggregate throughput, P95/P99, queue delay, and a declared per-class slowdown metric.

## Research Potential

Medium.

## Confidence

Medium; the trade-off is supported, while the unaddressed fairness mechanism is not yet established.

# RQ-5 — Does workflow-aware KV lifecycle control remain beneficial under branching, retries, and uncertain tool delays?

## Observation

Agent workflows produce reuse opportunities across tool gaps. KVFlow and Continuum show gains for graph/lifetime-aware policies, but their supplied evidence is bounded and does not establish a broad branching/looping model.

## Existing Approach

KVFlow uses an agent step graph and prefetching; Continuum optimizes TTL across tool gaps; generic systems use LRU/TTL/cache affinity without workflow semantics.

## Fragile Assumption / Limitation

The future path, time-to-reuse, and output length are sufficiently predictable for retained/prefetched state to repay its memory and bandwidth cost.

## Failure Scenario

Dynamic fan-out, loop/retry behavior, cancelled branches, unanticipated tool delay, or high workflow concurrency.

## Observable Consequence

- Prefetch bandwidth/memory rises without a job-completion-time improvement.
- Mis-predicted state crowds out a later useful cache entry or worsens P95/P99.

## Relevant Papers

KVFlow (2025), Continuum (2025), AMPD (2026), LoongServe (2024); `assumption_map.md` A4/A6/A7 and `research_tensions.md` T9.

## Why Worth Investigating

Agent workloads are an important and understandardized target. The question must first be grounded with trace semantics before a systems mechanism is justified.

## Cheap Validation

Use synthetic but explicitly parameterized branching/retry traces plus a small real tool-delay slice. Compare LRU/TTL to a simple oracle and an existing graph/lifetime proxy at equal HBM/bandwidth budgets; report JCT, hit, wasted prefetch, and P95.

## Research Potential

Medium.

## Confidence

Medium-Low.

# RQ-6 — Under which transfer-size, concurrency, and topology conditions is fetch cheaper than recomputation for KV state?

## Observation

Tiered systems report capacity and throughput gains, but different work uses PCIe, DRAM, SSD, CXL, RDMA, and different chunk/control paths. The actual fetch/recompute crossover is thus conditioned by more than bytes alone.

## Existing Approach

FlexGen uses overlap and placement search; InfiniGen fetches selectively; ShadowKV reduces fetched state; LMCache coalesces; DynamicPlacement models placement; Beluga changes the memory path.

## Fragile Assumption / Limitation

A linear, static data-movement cost model predicts the best tier or that remote transfer can be hidden. Granularity, contention, and coherence can invalidate it.

## Failure Scenario

Small fragmented transfers, concurrent consumers, an oversubscribed root complex, a low-bandwidth remote path, or short contexts where prefill is cheap.

## Observable Consequence

- Fetch time exceeds recompute time or misses a P99 SLO.
- Nominal bandwidth and realized throughput diverge; controller decisions invert across load.

## Relevant Papers

FlexGen (2023), InfiniGen (2024), ShadowKV (2025), LMCache (2025), DynamicPlacement (2025), Beluga (2025); `research_tensions.md` T5.

## Why Worth Investigating

A measurable boundary would make later placement/routing claims auditable across hardware rather than tied to one prototype.

## Cheap Validation

Measure representative KV transfer sizes under controlled concurrency on available local topology or a calibrated emulator. Compare measured fetch against re-prefill for two model sizes and report TTFT/TPOT, realized bandwidth, and the crossover point.

## Research Potential

Medium.

## Confidence

Medium-High.

# RQ-7 — How do scale and failure semantics change the benefit of a shared/disaggregated KV pool?

## Observation

Shared KV pools and distributed serving improve locality/capacity under the happy path. Their reliability and controller boundaries are incompletely comparable; a generic “no fault tolerance” claim is invalid because DéjàVu provides a direct pipeline-replication design.

## Existing Approach

Mooncake, LMCache, FlowKV, Dynamo, and Beluga focus on placement/data-path control; DéjàVu streams and replicates KV for a particular pipeline-parallel failure model.

## Fragile Assumption / Limitation

Controller/cache metadata, replication, recovery, and topology remain cheap enough that the pool continues to meet tail SLOs during scale/failure events.

## Failure Scenario

Hot-block incast, controller contention, node/cache loss, replication lag, or a failure/recovery event during high load.

## Observable Consequence

- Goodput/P99 deteriorates despite the steady-state cache hit rate.
- Recovery consumes enough memory/bandwidth to negate shared-pool benefits.

## Relevant Papers

Mooncake (2025), LMCache (2025), FlowKV (2025), Beluga (2025), DéjàVu (2024); `research_tensions.md` T10.

## Why Worth Investigating

It supplies a needed operational envelope, but likely begins as a systems evaluation/engineering study rather than a mechanism-only paper.

## Cheap Validation

Inject one bounded fail-stop/recovery or controller-delay condition in a small shared-cache setup. Compare steady state versus disturbance; report recovery time, P99 TTFT/TPOT, goodput, cache availability, and replication traffic.

## Research Potential

Medium-Low.

## Confidence

Medium.

## Tier decision

### Tier 1

- **RQ-1:** Strong cross-paper evidence and a compact experimental matrix; it directly connects P/D, network, workload, and tail SLOs.
- **RQ-2:** Fast to falsify and foundational for several compression/retention claims; it can separate genuine operating-boundary evidence from a generic “adaptive controller” story.
- **RQ-8:** Tests whether Stage 2A's joint-optimization narrative has a measurable interaction at all; it is intentionally a factorial measurement before any compound algorithm.
- **RQ-4:** Directly links cache reuse, scheduling, P99, and an explicit fairness definition; the local corpus has a concrete starvation limitation but no matched solution.

### Tier 2

- **RQ-3:** A high-impact non-prefix reuse robustness question, but existing Cache-Craft/CacheBlend/KVLink mechanisms make it unsuitable as a broad “nobody solves RAG reuse” claim.
- **RQ-5:** Valuable emerging-workload question, but requires trace realism before generalization.
- **RQ-6:** Technically important and evidence-backed, but hardware access/calibration raises validation cost.
- **RQ-7:** Important operationally; scope it tightly to one scale/failure model rather than claiming an unstudied universal reliability gap.

### Drop / do not advance as currently phrased

| Original direction | Audit classification | Reason for dropping or reframing |
|---|---|---|
| “A universal joint token × bit × placement optimizer” | Incremental / overly broad | The corpus supports a bounded joint **evaluation** in RQ-8, not a pre-justified controller or a promised multiplicative gain. |
| “Fault tolerance is missing from KV systems” | Probably already partially solved / engineering | DéjàVu directly implements asynchronous replication and recovery. A new question must name a modern cache-pool topology and fault model. |
| “Multi-tenant KV reuse is insecure” | Insufficient evidence | The local notes mainly infer this risk; no supplied direct attack/isolation evaluation establishes its magnitude. |
| “Perf/$ and perf/W is a new serving mechanism gap” | Benchmark / evaluation | Cost/energy reporting is useful, but it is a measurement extension unless paired with a concrete systems decision. |
| “Poisson traces prove production systems fail” | Benchmark / evaluation | Synthetic traces are a representativeness limitation, not direct proof of system failure. Use it as an evaluation dimension in RQ-1/RQ-4/RQ-5. |

## Traceability

Source synthesis: `research_tensions.md`, `opportunity_matrix.md`, `candidate_gaps.md`, `assumption_map.md`, `evaluation_map.md`, and the local paper notes cited in each RQ. No external novelty search has been performed in Stage 2B.
