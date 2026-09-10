# Stage 2B Final Review — Literature Synthesis Audit & Research Space Refinement

> **Role:** Senior Research Reviewer audit of the supplied Stage 2A corpus.  
> **Research scope:** LLM inference/serving optimization, with emphasis on KV cache and related systems.  
> **Evidence boundary:** 43-paper local manifest; `evaluation_map.md` contains detailed evaluation extraction for 24 papers. No external novelty search was performed in this stage.

## 1. Executive decision

Stage 2A is useful and substantially correct as a **broad field map**. It covers the central technical routes, records many paper-level caveats, and generally keeps candidate gaps marked “not yet verified as novel.” It should **not**, however, be used unchanged as evidence that a method family is globally absent or that cross-paper speedups are comparable.

The audit makes four structural corrections:

1. Replace the flat 14-way taxonomy with orthogonal **problem**, **intervention**, and **context/evaluation** axes.
2. Merge causally duplicated bottlenecks and move tail latency/quality from “root bottlenecks” to mandatory outcome metrics.
3. Downgrade candidate gaps that are already partially addressed, are chiefly benchmarks/engineering, or rest on inferred rather than direct evidence.
4. Retain **8 bounded Stage 3 questions**, with **4 Tier 1** questions selected for falsifiable, small-scale validation before any solution design.

## 2. What Stage 2A got right

- **Coverage breadth:** the corpus covers allocation/layout, retention, quantization, reuse, tiering, transfer, scheduling, P/D disaggregation, and distributed serving. This is adequate to formulate Stage 3 hypotheses, although it is not a complete field search.
- **Key systems interactions:** it correctly identifies capacity versus bandwidth, cache locality versus load, prefill/decode interference, and approximation versus quality as recurring themes.
- **Conditional reading of apparent contradictions:** `contradictions.md` is strongest when it conditions claims on workload, topology, scale, implementation, and metric rather than treating paper results as logical incompatibilities.
- **Evaluation awareness:** `evaluation_map.md` appropriately records TTFT, TPOT/TBT, throughput/goodput, memory, quality, tail metrics, workloads, and hardware as distinct axes.
- **Novelty restraint:** the original candidate gaps were marked conservatively. That restraint must remain: Stage 2B still does not certify novelty.

## 3. Material corrections incorporated into the final artifacts

| Stage 2A issue | Audit correction | Evidence boundary |
|---|---|---|
| `coverage_report.md` still lists five “missing” papers. | It is a **pre-supplement historical snapshot**. CacheGen, CachedAttention, Dynamo, Llumnix, and LoongServe were later added to the 43-item manifest and matrix; retain the report only as provenance for that supplementation decision. | `coverage_report.md`; `papers.md` entries 39–43; `literature_matrix.md` rows 39–43. |
| Flat “method” taxonomy mixes mechanisms, architectures, workloads, and outcomes. | `taxonomy_final.md` separates problems, interventions, and context/evaluation axes. RAG/agents/long-context/heterogeneity are not peer method families. | `taxonomy_draft.md` Categories 1–14; paper notes across the corpus. |
| CacheGen is listed as a representative low-bit KV quantization method. | Reclassify CacheGen as a KV **transport/storage codec** under transfer/data-path optimization; KIVI and GEAR remain the primary numerical KV quantization examples. | `literature_matrix.md`; `paper_notes/CacheGen.md`, KIVI, GEAR. |
| FlowKV `23,469 → 1` is described as “24×.” | It is a **call-count reduction**. The roughly 24× result is a separate latency/speedup measurement and must not be substituted for the count. | `paper_notes/FlowKV.md` lines 106–109. |
| Beluga’s 89.6% TTFT reduction and 7.35× QPS are attributed to Mooncake/Dynamo behavior. | These are Beluga cache-hit results against its stated baseline; they do not show Mooncake Conductor minimizing the reported P99 values. | `paper_notes/Beluga.md` lines 131–133; Mooncake note limitations. |
| “No fault tolerance” in distributed KV serving. | False as a blanket statement: DéjàVu implements asynchronous token-KV replication and fail-stop recovery. The defensible conclusion is uneven coverage for modern shared/cache-pool topologies. | `paper_notes/DejaVu.md` lines 34–40, 105–116. |
| “No joint optimization exists; gains should multiply.” | Too strong: H2O already combines retention with low-bit storage; GEAR combines quantization, low-rank, and sparse compensation; ShadowKV combines approximation with offload. Retain only the testable question of **non-additive three-axis interaction**. | H2O, GEAR, ShadowKV notes; `candidate_gaps.md` Gap 1. |
| “All strong systems rely on static tuning; no online estimator exists.” | Too strong: fixed parameters are common, but Continuum has dynamic TTL decisions and other systems have adaptive/predictive components. Narrow the question to a specified control knob under distribution shift. | Mooncake, Continuum, SCOPE, KVFlow notes. |
| “Mostly Poisson evaluation proves production failure.” | Unsupported: Mooncake replays a timestamp trace, Llumnix includes burstiness, SGLang reports production hit rates. The defensible gap is no matched real/burst/Poisson matrix with tail/fairness reporting. | Mooncake, Llumnix, SGLang notes. |
| Privacy/isolation performance loss is presented as fact. | Reuse opportunity is directly observed, but leakage, attack feasibility, and isolation-cost magnitude are mainly inferred in this corpus. Keep as a security-validation gate, not a Stage 3 core claim. | `limitation_map.md` L7 and local notes marked inference. |
| FlexGen is treated as a two-tier HBM+DRAM system. | FlexGen explicitly spans GPU, CPU, and disk; the fragile assumption is the adequacy/predictability of the selected cold-tier path, not two-tier sufficiency. | `paper_notes/FlexGen.md`; `paper_notes/CachedAttention.md`. |
| Heterogeneous performance figures are compared as a common multiplier. | Results differ in model, trace, GPU/topology, SLO, quality constraint, and baseline. A shared-harness comparison is required before ranking systems. | `evaluation_map.md`, `contradictions.md`, `literature_matrix.md`. |
| TTFT coverage aggregates total latency, transfer/E2E proxies, and first-response metrics. | Do not re-use the Stage 2A aggregate TTFT count. Future tables must distinguish **TTFT**, first-response proxy, transfer latency, and total E2E, then compare only matched definitions/cache states/SLOs. | SGLang, HotPrefix, FlowKV notes; `evaluation_map.md` metric table. |
| “Long-context KV memory pressure is quadratic.” | Separate resources: KV-cache capacity grows approximately **linearly** with context length; prefill attention compute is typically superlinear/quadratic. The two should be measured separately. | `bottleneck_map.md` B12; long-context paper notes. |
| Dynamo is treated as equivalent peer-reviewed evidence. | Use Dynamo only as industrial/vendor implementation evidence; do not use projected/vendor speedups in paper counts, maturity scores, or cross-paper multipliers. | `paper_notes/Dynamo.md`. |

Other retained cautions: local allocation fragmentation differs from small-I/O/metadata overhead; a workload-specific Mooncake cache plateau is not a universal reuse ceiling; and reported LMCache small/large transfer figures use **GB/s**, not Gbps.

## 4. Audited bottleneck map

This table reports causal bottlenecks only. Tail latency, fairness, quality, cost, and energy are required outcomes or validation dimensions, not peer root causes.

| Canonical bottleneck | Stage 2A rows consolidated | Evidence strength | Scope | Existing coverage | Audit decision |
|---|---|---|---|---|---|
| **B1. KV footprint / capacity pressure** | B1 + B12 | Strong | General capacity issue; severity depends on context, model KV architecture, and hardware | Partially solved | Long context is a trigger of capacity pressure, not a separate root cause. |
| **B2. Allocation/layout inefficiency** | B2 | Strong | Contiguous allocators and paged-transfer layouts | Largely solved for local allocation; active for movement layout | Paged blocks address allocator fragmentation; small blocks may create metadata/kernel/small-I/O cost rather than allocator external fragmentation. |
| **B3. Reuse miss and redundant prefill** | B3 + B4 | Strong | Prefix-heavy chat and RAG; arbitrary-position composition is narrower | Exact reuse mature; compositional reuse active | Keep redundant recompute as the consequence of invalid/missed reuse. |
| **B4. Tier placement and data-movement overhead** | B5 + B6 + B13 | Strong | PCIe/RDMA/CXL/SSD path, transfer size, congestion, topology | Actively studied | Placement and movement are coupled but distinct: there is no topology-independent winner. |
| **B5. Prefill/decode resource interference** | B7 | Strong | Mixed colocated batches, prompt/output mix, SLO-constrained serving | Continuous batching mature; P/D/chunked variants active | P/D is an architecture that trades this bottleneck against B4. |
| **B6. Cache/routing/admission prediction and load balance** | B8 + B9 + B10 | Moderate | Workload skew, burst, cluster size, agent structure | Active | Imperfect prediction alone is not a research gap; measure its operating cost. |
| **B7. Lossy-KV fidelity and positional correctness** | B14 + positional cases | Strong, method-specific | Compression, pruning, quantization, non-prefix reuse, model positional encoding | Active | Quality is an outcome, but fidelity/correctness is a real mechanism-specific constraint. |
| **B8. Workflow-state lifetime and tool-delay management** | Spread across L12 / workflow notes | Moderate | Multi-turn agents, tool gaps, branching/async behavior | Emerging / weakly explored | Do not claim all agent workflows are linear; scope the question to unobserved DAG/async cases. |
| **B9. Shared-pool scale/recovery envelope** | Distribution of B10/L11 | Moderate | Controller/cache scale, hot-block replication, fail-stop/recovery model | Partially addressed / engineering-heavy | DéjàVu removes a zero-coverage claim; modern shared-pool failure behavior remains a validation question. |

### Outcomes that every Stage 3 study must report when relevant

- TTFT and TPOT/TBT: median plus P95/P99 or another explicitly defined percentile/SLO.
- Throughput **and** goodput/SLO attainment, with a fixed quality criterion for lossy methods.
- HBM/host/tier memory and realized bandwidth, not only nominal bandwidth.
- For cache/routing studies: hit rate, queue delay, per-class/per-tenant slowdown or a declared fairness metric.
- For P/D or tiered systems: topology, transfer granularity, model-weight duplication, and deployment scale.

## 5. Fragile assumptions worth testing

| Assumption | Who relies on it | Why | Violation scenario | Expected consequence | Evidence strength | Worth testing? |
|---|---|---|---|---|---|---|
| KV capacity is the limiting resource. | vLLM, FlexGen, H2O, KIVI, ShadowKV. | Memory saving is expected to permit batch/throughput growth. | Compute-bound prefill or transfer-bound decode. | Memory drops without goodput improvement. | Strong, conditional. | Yes. |
| A prefill/decode phase split remains beneficial. | Splitwise, DistServe, Sarathi, TetriInfer, AMPD. | Different phase resource profiles justify separation. | Topology/transfer overhead, workload shift, or changing model/kernel behavior. | P/D loses to colocated/chunked execution at equal SLO. | Strong in tested settings. | Yes. |
| Offline profiles and fixed thresholds transfer. | KIVI, PyramidKV, Sarathi, InfiniGen, CacheBlend, Mooncake. | Parameters/cost models are calibrated once. | Task/model/hardware/arrival distribution changes. | Quality, tail latency, or utilization regresses versus a per-condition oracle. | Strong for fixed parameters; moderate for universal failure. | Yes. |
| Arrival, output length, and reuse distributions are stable enough. | DistServe, Mooncake, RAGCache, Continuum. | Enables provisioning, queue, and reuse estimates. | Burst, long-tail tool delay, or workload mix shift. | Misprediction, cache pollution, queue/SLO failure. | Moderate. | Yes. |
| Available metadata predicts future reuse value. | SGLang, RAGCache, HotPrefix, Mooncake, KVFlow, Continuum. | Admission/eviction/prefetch/routing need an expected-value signal. | Reuse distance or workflow path is wrong. | Missed hit, wasted prefetch, hotspot, or memory pressure. | Moderate, workload-specific. | Yes. |
| Exact-prefix reuse captures the economically useful reuse. | vLLM, SGLang, RAGCache, Mooncake. | Exact reuse is cheap and correct. | Multi-history or reordered RAG context. | Redundant prefill persists or invalid reuse harms quality. | Strong for RAG-specific counterevidence. | Yes. |
| A small repaired subset makes non-prefix reuse safe. | CacheBlend, Cache-Craft, KVLink. | Selective recompute/link tokens should restore validity cheaply. | Dense cross-chunk dependency, retrieval/order drift, or model change. | Quality loss or repair cost removes latency gain. | Moderate. | Yes. |
| Past/local attention predicts future token importance. | H2O, SnapKV, PyramidKV, Scissorhands, SCOPE. | Enables cheap retention without future decoding. | Whole-context/reasoning/phase-shifted task. | Important state evicted; quality declines. | Strong within methods; conditional generally. | Yes. |
| One lossy KV configuration preserves quality. | KIVI, GEAR, ShadowKV, retention methods. | Fixed bit/budget settings are reported as good operating points. | MQA/GQA, long CoT, different task/model. | Silent quality degradation or a larger residual/bit budget is needed. | Strong, method-specific. | Yes. |
| Positional transforms preserve semantics. | StreamingLLM, CachedAttention, CacheBlend, Cache-Craft, KVLink. | Reuse/eviction must maintain positional correctness. | Incorrect RoPE/position repair or unsupported model behavior. | Large PPL/F1/accuracy degradation. | Strong within evaluated scopes. | Yes. |
| Transfer/prefetch is predictable and hideable. | FlexGen, InfiniGen, LMCache, Mooncake, FlowKV, CachedAttention. | Placement and scheduling assume a bounded data path. | Small I/O, congestion, high concurrency, root-complex/coherence effects. | TTFT/TPOT tails rise; fetch loses to recompute. | Strong, topology-specific. | Yes. |
| Workflow lifecycle metadata is predictable. | KVFlow, Continuum, AMPD. | State is retained/prefetched across tool gaps. | Branch, retry, cancellation, asynchronous tools, cross-workflow contention. | Wasted memory/bandwidth; no JCT benefit. | Moderate; few systems. | Yes. |

## 6. Candidate-gap audit

Class definitions follow the user-specified A–F scheme. “E” means only a plausible research question in this corpus; it is not a novelty conclusion.

| Stage 2A gap | Class | Audit verdict | Stage 3 disposition |
|---|---|---|---|
| G1. Joint token × bit × placement | **C — Benchmark/evaluation gap** | Existing hybrid papers mean “every axis is isolated” is false. The remaining question is whether three axes interact non-additively. | Retain as RQ-8 factorial validation, not a universal controller proposal. |
| G2. Online adaptation of static parameters | **B — Incremental gap** | Fixed knobs are common, but dynamic TTL/adaptive components already exist. | Fold into RQ-2/RQ-5 as one specific knob under drift. |
| G3. Imperfect-prediction heterogeneous placement | **E — Potential research gap** | DynamicPlacement and Beluga leave a bounded, topology-specific prediction/placement question. | Retain as RQ-6; do not promise 100s-node universal heterogeneity. |
| G4. Fault tolerance / consistency / availability | **D — Engineering gap** | DéjàVu directly supplies replication/recovery for one failure model; KV is derived/recomputable state. | Retain only as a scoped recovery-SLO validation in RQ-7. |
| G5. SLO-aware fairness with prefix affinity | **E — Potential research gap** | SGLang documents starvation; FastServe/Llumnix cover adjacent scheduler concerns without the prefix-affinity combination. | Retain as RQ-4 with an explicit fairness metric. |
| G6. Multi-tenant isolation/security | **F — Insufficient evidence** | Reuse opportunity is documented; attack/leakage magnitude and isolation cost are not established by direct evidence here. | Do not advance before threat model and security-prior-art search. |
| G7. Non-prefix RAG contamination/drift | **B — Incremental gap** | Cache-Craft already handles multi-history/order contamination. The narrower untested boundary is threshold robustness under retrieval/order/chunk drift. | Retain as RQ-3 robustness evaluation, not an absence claim. |
| G8. Branching/looping agent state lifetime | **F — Insufficient evidence as broadly phrased** | KVFlow already represents statically declared branches/barriers; the local corpus does not directly establish a general gap for all branches/loops. | Keep RQ-5 only as an exploratory, tightly scoped dynamic-spawn/async validation after trace grounding. |
| G9. TCO, power, and energy | **C — Benchmark/evaluation gap** | Some cost/energy measurements already exist; a common SLO-aware TCO/perf-W comparison is absent. | Make it a reporting layer, not a primary KV mechanism RQ. |
| G10. Realistic burst traces and tail SLO | **C — Benchmark/evaluation gap** | The corpus is not Poisson-only; no matched real/burst/Poisson + P99/fairness matrix is available. | Make it a common evaluation layer for RQ-1/RQ-4/RQ-5. |

## 7. Research space after compression

The retained research space is organized around **conditional failures**, not “no one has done this” statements:

1. P/D architecture versus transfer/queue/topology crossover.
2. Phase-/task-sensitive KV retention and precision bounds.
3. Non-additivity of token reduction, numerical precision, and tier choice.
4. Cache locality versus per-class tail latency and fairness.
5. Robustness of existing non-prefix RAG reuse under retrieval/order/chunk drift.
6. Prediction error versus heterogeneous tier placement benefit.
7. Workflow state lifetime under non-linear/async agent execution.
8. Small shared-pool scale/recovery envelope under a stated fault model.

`research_tensions.md` expands the evidence for these items. The most useful tensions are P/D versus transfer, cache affinity versus tail/fairness, exact versus repaired reuse, capacity versus data movement, static tuning versus drift, and local importance assumptions versus task/phase variation.

## 8. Final Stage 3 candidate ranking

Detailed templates (observation, existing approach, fragile assumption, failure scenario, observable consequence, relevant papers, cheap validation, potential, confidence) are in `stage3_candidates.md`.

### Tier 1 — validate first

1. **RQ-1: P/D hardware/workload crossover.** Strong evidence, direct metrics, and a small trace/topology sweep can falsify it.
2. **RQ-2: Static retention/precision budget failure boundaries.** Cheap matrix across task type and generation phase; prevents overgeneralized compression claims.
3. **RQ-8: Token × bit-width × tier non-additivity.** A 3×2×2 factorial test determines whether a compound mechanism is warranted at all.
4. **RQ-4: Cache affinity versus P99/fairness under skew.** Direct starvation evidence exists; define fairness before proposing a policy.

### Tier 2 — valuable with an added boundary or higher setup cost

5. **RQ-3:** Robustness of existing non-prefix RAG reuse under retrieval/order/chunk drift.
6. **RQ-5:** Workflow-state lifetime under branching, retries, and asynchronous tool delays.
7. **RQ-6:** Fetch/recompute crossover under imperfect placement prediction and heterogeneous tiers.
8. **RQ-7:** Shared-pool scale/recovery behavior under a declared fault model.

### Do not promote now

- A generic fault-tolerance, multi-tenant security, TCO, or realistic-trace **mechanism** claim.
- A universal online controller or universal three-axis optimizer.
- Any statement that performance regression alone establishes a research gap.

## 9. Largest remaining uncertainty

The largest uncertainty is not the availability of possible algorithms. It is whether the observed interactions survive a **matched workload × hardware/topology × metric** experiment. The local corpus varies in model, GPU, memory tier, network, scale, trace, SLO definition, quality target, and baseline. In addition:

- only part of the 43-paper corpus received full evaluation extraction;
- several Stage 2A claims were cross-paper inferences rather than directly stated paper results;
- no external 2024–2026 novelty search has been run, so the audit cannot establish “not already solved”;
- security, cost, and large-scale failure claims have particularly uneven direct evidence;
- industrial/vendor evidence (notably Dynamo) and bibliographic version/venue labels must not be treated as peer-reviewed, directly comparable measurements.

Stage 3 should therefore begin with bounded falsification experiments and retain negative results. A null result on a Tier 1 question is useful: it shrinks the research space without inventing an unnecessary solution.

## 10. Deliverable index

- `research/knowledge/taxonomy_final.md` — corrected hierarchical taxonomy.
- `research/knowledge/research_tensions.md` — evidence-bounded trade-offs and breakdowns.
- `research/knowledge/opportunity_matrix.md` — non-mechanical opportunity matrix.
- `research/knowledge/stage3_candidates.md` — eight ranked Stage 3 RQs and cheap validation directions.
- `research/knowledge/stage2_final_review.md` — this audit, bottleneck/assumption/gap decisions, and priority rationale.

## Traceability

Primary local sources: `taxonomy_draft.md`, `bottleneck_map.md`, `assumption_map.md`, `limitation_map.md`, `candidate_gaps.md`, `contradictions.md`, `evaluation_map.md`, `literature_matrix.md`, `research_landscape.md`, and paper notes for vLLM, SGLang, H2O, KIVI, GEAR, CacheBlend, Cache-Craft, KVLink, Sarathi-Serve, DistServe, Mooncake, FlowKV, Beluga, DynamicPlacement, KVFlow, Continuum, Llumnix, and DéjàVu. All claims above are scoped to that local evidence base.
