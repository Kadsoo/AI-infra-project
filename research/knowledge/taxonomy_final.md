# Final Taxonomy — LLM Inference / Serving Optimization with KV Cache

> **Stage 2B audit output** | Corpus boundary: the 43-paper manifest and local `paper_notes/` supplied to Stage 2A.  
> **Important boundary:** this is a corrected organizing scheme for the supplied corpus, not a claim that it exhausts the field or establishes novelty.

## Audit verdict

Stage 2A correctly covered the main technical routes: paged allocation, retention/eviction, compression, quantization, reuse, tiering, transfer, scheduling, prefill/decode (P/D) disaggregation, and distributed serving. Its main defect was a **flat list that mixed three different logical levels**:

1. **Problem / bottleneck** — e.g., KV capacity pressure, transfer delay, phase interference.
2. **Intervention / solution family** — e.g., quantization, prefix reuse, request routing.
3. **Operating context or evaluation axis** — e.g., RAG, agents, long context, CXL, P99, quality.

The final taxonomy keeps these axes separate. A paper can therefore have one primary intervention family while being evaluated under multiple contexts and addressing more than one bottleneck.

## Terms that must not be conflated

| Term | Meaning in this taxonomy | Examples in the corpus |
|---|---|---|
| **Capacity pressure** | A problem: KV state no longer fits a desired active set. | Long context, large batch, fragmentation; vLLM, ShadowKV, Beluga. |
| **Fragmentation** | A specific capacity-efficiency problem caused by allocation/layout. | vLLM PagedAttention; LMCache coalescing. |
| **Eviction / retention** | A state-selection intervention: discard or retain a subset of existing KV. | H2O, StreamingLLM, SnapKV, PyramidKV, SCOPE. |
| **Compression** | Change the representation of retained state while retaining its logical role. | ChunkKV semantic chunks; ShadowKV low-rank/landmarks. |
| **Quantization** | A numerical compression subfamily; bit-width and scaling are the main intervention. | KIVI, GEAR. |
| **Reuse** | Avoid re-prefill by using valid existing state. Its validity scope must be explicit. | Prefix reuse, copy-on-write (CoW), non-prefix reuse. |
| **Placement** | Decide where KV resides. | HBM/DRAM/SSD/CXL/RDMA placement. |
| **Transfer** | Move or expose KV between locations; a data-path problem distinct from placement. | FlowKV, Mooncake, CacheGen, Beluga. |
| **Tail latency / goodput / quality** | Outcomes and evaluation metrics, **not** bottleneck families. | P99 TBT, TTFT, quality/F1/LongBench. |

## A. Problem taxonomy (what is failing)

| Problem family | Concrete failure mode | Typical scope | Corpus evidence |
|---|---|---|---|
| P1. State capacity and layout inefficiency | HBM exhaustion, reservation waste, fragmentation, long-context state growth | General capacity problem; severity is model-, context-, and hardware-dependent | vLLM, FlexGen, ShadowKV, Beluga; `bottleneck_map.md` B1/B2/B12. |
| P2. Reuse opportunity and reuse validity | No useful cached state, exact-prefix miss, invalid non-prefix composition, redundant prefill | Strongly workload- and history-dependent | SGLang, RAGCache, CacheBlend, Cache-Craft, KVLink. |
| P3. Data-movement mismatch | PCIe/DRAM/SSD/RDMA/CXL path cannot supply demanded KV state at the required latency | Hardware- and implementation-dependent | InfiniGen, LMCache, FlowKV, Mooncake, Beluga. |
| P4. Execution interference and queueing | Prefill disrupts decode, head-of-line blocking, load skew, poor routing | Workload-, SLO-, and scale-dependent | Sarathi-Serve, DistServe, Mooncake, Llumnix, FastServe. |
| P5. Approximation correctness risk | Retention, compression, or quantization changes model quality or stability | Task-, model-, and generation-length-dependent | H2O, KIVI, GEAR, PyramidKV, SCOPE. |
| P6. Operational robustness | Cache lifecycle, elasticity, tenant boundary, failure recovery, consistency | Primarily deployment-/implementation-dependent; corpus coverage is sparse | DéjàVu directly covers pipeline failure recovery; later KV-pool evidence is incomplete. |

P1–P6 are **problems**. They should not appear beside a solution such as “KV compression” in the same taxonomy level.

## B. Intervention taxonomy (how a system changes the operating point)

### I. State representation and capacity efficiency

| Subfamily | Primary intervention | Do not merge with | Representative corpus evidence |
|---|---|---|---|
| I.1 Allocation and layout | Allocate variable-length KV state efficiently; page/block/segment it. | Reuse policy or remote placement. | vLLM PagedAttention, SGLang storage substrate, LMCache coalescing, FlowKV segments. |
| I.2 Retention / eviction | Select which logical tokens/blocks remain available. | Structural compression; quantization. | StreamingLLM, H2O, Scissorhands, SnapKV, PyramidKV, **ChunkKV**, SCOPE. |
| I.3 Structural approximation | Approximate retained state through low-rank or landmark structure. | Simple token eviction and numeric low-bit storage. | ShadowKV; parts of GEAR are a hybrid. |
| I.4 Numerical compression | Lower bit-width while preserving the state representation. | Token selection or a transport codec. | KIVI, GEAR; FlexGen's low-bit storage is a tiering companion. |

**Boundary:** I.2 chooses *which state* to preserve; I.3/I.4 change *how selected state is represented*. A hybrid paper may belong to more than one subfamily, but its contribution must say which component is primary.

### II. Reuse, validity, and lifecycle

| Subfamily | Primary intervention | Scope / validity rule | Representative corpus evidence |
|---|---|---|---|
| II.1 Within-request sharing | Share identical prefix/beam state by reference or CoW. | Same request or explicitly coordinated branches; not generic multi-tenant reuse. | vLLM CoW. |
| II.2 Exact cross-request prefix reuse | Reuse a token-identical prefix through a radix/tree/hash index. | Token order and positional semantics must match. | SGLang, RAGCache, Mooncake, LMCache. |
| II.3 Non-prefix or compositional reuse | Reuse parts of context only after restoring validity or accepting bounded loss. | Requires an explicit correctness mechanism or a quality budget. | CacheBlend selective recompute, Cache-Craft contamination scoring, KVLink trainable links. |
| II.4 Lifecycle policy | Admit, retain, evict, prefetch, replicate, or expire reusable state. | Depends on future reuse estimates and tenant/workflow context. | PGDSF/RAGCache, HotPrefix, KVFlow, Continuum, Mooncake. |

**Correction to Stage 2A:** “prefix caching,” “KV sharing,” and “recomputation” are not three peer families. Prefix caching and CoW differ by reuse scope; non-prefix reuse uses recomputation as a *repair/fallback mechanism*; lifecycle policy controls all of them.

### III. Placement and data path

| Subfamily | Primary intervention | Separate question it does not answer | Representative corpus evidence |
|---|---|---|---|
| III.1 Tier placement and offload | Choose HBM, DRAM, SSD, CXL, or remote-memory residency. | How efficiently the selected tier is accessed. | FlexGen, InfiniGen, CachedAttention, DynamicPlacement, Beluga. |
| III.2 Local movement and prefetch | Hide or reduce host/device movement. | Whether remote placement is worthwhile. | InfiniGen, ShadowKV, LMCache. |
| III.3 Distributed transfer and cache-pool access | Coalesce, encode, overlap, or directly access remote KV. | Placement/routing policy and cache validity. | FlowKV, Mooncake, **CacheGen**, Dynamo, Beluga. |

**Boundary:** placement answers “where should state live?”; transfer answers “how does a consumer obtain it?” A CXL or RDMA paper can contribute to both, but the two cost models must remain distinct.

### IV. Service execution and control plane

| Subfamily | Primary intervention | Representative corpus evidence |
|---|---|---|
| IV.1 Iteration-level batching and preemption | Interleave work or control queue position within a serving replica. | Orca, Sarathi-Serve, FastServe. |
| IV.2 Cache-/SLO-aware scheduling and routing | Choose admission, queue order, cache affinity, and request destination. | SGLang, HotPrefix, Mooncake, Online Scheduling; Dynamo is industrial implementation evidence only. |
| IV.3 P/D architecture | Separate prefill and decode resources and select phase-specific parallelism. | Splitwise, DistServe, TetriInfer, DéjàVu, AMPD. |
| IV.4 Elasticity and distributed coordination | Migrate state, change roles/parallelism, or scale replicas. | Llumnix, LoongServe, Mooncake, Dynamo, FlowKV. |

P/D disaggregation and distributed serving are **architectural/control-plane families**, not “bottlenecks.” Their benefits and costs must be measured through P3/P4 outcomes.

## C. Context and evaluation axes (where claims do or do not generalize)

These are mandatory tags on a result, not top-level solution families:

- **Workload:** chat/prefix-heavy, RAG, multi-turn, agent/workflow, long-context, coding/reasoning, bursty/multi-tenant.
- **Hardware/topology:** GPU generation and memory, PCIe/NVLink/CXL/RDMA/SSD path, node count, heterogeneous versus homogeneous workers.
- **Service objective:** TTFT, TPOT/TBT, throughput/goodput, P95/P99, memory, bandwidth, utilization, scale-out, quality.
- **Correctness/operations:** exactness versus bounded degradation, training required versus training-free, tenant isolation, crash/failure model.

“Workload-aware optimization” from Stage 2A is therefore moved from a peer method category to this axis. It describes a condition under which any family may be evaluated, not a single mechanism.

## Mapping from the Stage 2A draft

| Stage 2A category | Final location | Audit decision |
|---|---|---|
| 1. KV allocation & paged memory | I.1 | Keep, but remove FlexGen's placement policy from the core definition. |
| 2. Eviction | I.2 | Keep as logical-state selection. |
| 3. Compression | I.3 | Keep, but call it structural/semantic compression; do not treat ShadowKV as ordinary token dropping. |
| 4. Quantization | I.4 | Keep as a distinct numerical subfamily. |
| 5. Prefix caching & reuse | II.2 | Narrow to exact cross-request prefix reuse. |
| 6. KV sharing | II.1 and II.3 | Split CoW (within request) from non-prefix compositional reuse. |
| 7. Offloading & hierarchical memory | III.1 | Keep placement/tiering here. |
| 8. Recomputation | II.3 | Reclassify as repair/fallback for reuse, not a standalone universal family. |
| 9. KV transfer & communication | III.3 | Keep as data-path optimization. |
| 10. Scheduling & batching | IV.1 and IV.2 | Split local execution from global control. |
| 11. Routing & load balancing | IV.2 | Keep as control-plane policy. |
| 12. P/D disaggregation | IV.3 | Keep as architecture. |
| 13. Distributed serving & elasticity | IV.4 | Keep as architecture/control plane. |
| 14. Workload-aware optimization | C (context axis) | Remove as a peer solution family. |

## Coverage and evidence cautions

1. **A solution stack is not a single method.** Paging, reuse, offload, transfer, and scheduling often compose. Claims of “orthogonality” require a joint experiment, not only separate paper results.
2. **Reuse requires a validity statement.** “Shared,” “prefix,” and “non-prefix” reuse are materially different regarding position, causal attention, tenant boundary, and quality. CacheGen is a transport/storage codec, not a representative low-bit KV quantization method.
3. **Tail latency, fairness, quality, and energy are outcome axes.** They should be reported alongside a proposed intervention rather than named as independent solution routes.
4. **Reliability/security are sparse in this corpus.** They are operational constraints, not proof that a new KV method is required. DéjàVu supplies direct fault-recovery evidence; privacy/tenant claims in other notes are predominantly inferred.
5. **Maturity is conditional.** Paged allocation and exact prefix caching are mature in the surveyed systems; 2-bit KV quantization, non-prefix reuse, CXL-tiering, and workflow-aware lifecycle policies remain workload/hardware dependent.
6. **Vendor evidence is not peer-reviewed performance evidence.** Dynamo can illustrate implementation direction, but projected/vendor metrics are excluded from cross-paper multipliers, paper-frequency counts, and maturity conclusions.

## Traceability

This revision was derived from `taxonomy_draft.md` Categories 1–14, the causal distinctions in `bottleneck_map.md` B1–B14, the assumption boundaries in `assumption_map.md`, and paper-note evidence for vLLM, SGLang, KIVI, GEAR, CacheBlend, Cache-Craft, KVLink, Mooncake, FlowKV, DistServe, Sarathi-Serve, Beluga, DynamicPlacement, KVFlow, Continuum, and DéjàVu. Claims outside that local corpus are intentionally not made.
