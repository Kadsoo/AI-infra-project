# Stage 3R-C — Anomaly Review, Mechanism Framing, and Research-Question Gate

> Date: 2026-08-28  
> Reviewer role: Senior Empirical Systems Researcher  
> Scope: Stage 3R-A/B measurement artifacts, selected raw repeat files, Stage 2 maps/tensions, and local paper notes. This is an evidence gate, not a novelty search or an optimization design.

## Executive Decision

The reset produced one strong, system-specific serving phenomenon worth a narrowly scoped return to Stage 3A: a concurrency knee with a large TTFT/tail cost after request throughput has plateaued. Two further questions are retained only as Tier 2: one tests whether output length changes that knee, and one tests which runtime signals actually reveal it. No observation supports a Stage 4 mechanism or a novelty claim.

| Item | Count / decision |
|---|---|
| Reviewed anomaly records | 5 |
| TRUSTED | 2: A_01, A_02 |
| PROVISIONAL | 1: A_03 as an instrumentation observation only |
| WEAK | 1: A_05 |
| ARTIFACT | 1: A_04 for its latency/throughput claim |
| Candidate RQs retained | 3 |
| Tier 1 | 1 |
| Tier 2 | 2 |
| Recommendation | Re-enter Stage 3A narrowly for Tier 1 only; do not enter Stage 3B or Stage 4 yet. |

All results are scoped to hf_transformers_naive, tiny-gpt2, FP32 CPU inference, Windows, and one host. They are not evidence about vLLM, SGLang, PagedAttention, GPU memory pressure, or A100-scale serving.

## 1. Measurement Trust Gate

### Audit method

- Read the Stage 3R-A baseline, reliability review, instrumentation validation, sweep matrix, anomaly records, negative observations, and Stage 3R-B summary.
- Recomputed the central three-repeat aggregates from the processed raw files for A_01, A_02, and A_04.
- Checked the reported mixed-workload reproduction artifact names against the workspace file inventory.
- Treated raw/summary disagreement as a measurement-reporting risk, rather than silently choosing the more favorable summary value.

The harness-level checks remain useful: warmups are excluded, SSE TTFT/TPOT are defined consistently, sampler overhead was measured below roughly 2 percent in the normal setting, and the central A_01/A_02 repeats have zero failures. The system-level inference is nevertheless narrower than the harness reliability claim because server queue state, executor backlog, active requests, batch size, KV state, per-core CPU, and Torch thread state are unavailable.

### Classification table

| ID | Audit classification | Recomputed or directly checked evidence | Main confounder / artifact risk | Disposition |
|---|---|---|---|---|
| A_01 — concurrency knee and tail explosion | **TRUSTED** for the relative phenomenon | Fresh 3x repeats, synthetic 512/64 closed workload: c1 = 0.673 plus-or-minus 0.003 rps and p50 0.619 plus-or-minus 0.004 s; c4 = 0.885 plus-or-minus 0.004 and 3.071 plus-or-minus 0.169 s; c8 = 0.873 plus-or-minus 0.015 and 5.817 plus-or-minus 0.160 s. Every repeat completed 38/38 requests. | Absolute performance drifts across sessions; the defensible knee is around c3-c4, not an exact universal c3. One CPU-only, no-continuous-batching engine. | Tier 1 RQ root. |
| A_02 — output-length tradeoff | **TRUSTED**, but expected | At c2 with 512 input, 3x repeats give 16 to 256 output: 1.071 plus-or-minus 0.050 to 0.421 plus-or-minus 0.001 rps, p50 1.049 plus-or-minus 0.012 to 3.910 plus-or-minus 0.033 s, while TTFT stays 0.388 to 0.430 s. | This is ordinary autoregressive decode and request-versus-token accounting; it is not itself a new bottleneck. Intermediate points are single-run. | Control/root for Tier 2 interaction test only. |
| A_03 — telemetry does not track tail magnitude | **PROVISIONAL** as a telemetry observation; not a root-cause claim | The sweeps consistently show a much larger tail change than the aggregate CPU/RAM signals, and server-side state is unavailable. | The report contains CPU/RAM summary inconsistencies. For example, baseline_summary reports about 13-18 percent CPU for baseline rows, but a sampled baseline raw file has a 74.74 percent mean. A c64 processed file reports 19.75 GB RAM, not the 17.6 GB summarized in A_01/A_03. | Tier 2 diagnostic RQ only; no resource-bottleneck claim. |
| A_04 — bursty arrivals improve latency | **ARTIFACT** for latency/throughput; at most a weak client-queue signal | Fresh 3x: closed 0.897 plus-or-minus 0.006 rps and p50 5.677 plus-or-minus 0.188 s; bursty-r2 0.887 plus-or-minus 0.018 and 5.632 plus-or-minus 0.334 s. The intended effect is below repeat variance. | The remaining queue difference is client semaphore wait, not server queue, and the bursty queue CV is about 57 percent. | Drop. |
| A_05 — mixed workload helps long requests | **WEAK** as an anomaly/RQ root | Six single coarse runs show the stated direction at c4 and c8. | Claimed reproMixed files are absent from the workspace. More importantly, mixed requests have substantially less total input/output work than homogeneous-long requests, so workload amount is inseparable from scheduling. | Drop; do not generate an RQ from it. |

### Raw-trace checks and record inconsistencies

The central A_01 and A_02 numbers above match the following processed files: sweeps/raw/reproA1_conc{1,4,8}_rep{0,1,2}-*_processed.json and sweeps/raw/reproA2_out{16,64,256}_rep{0,1,2}-*_processed.json. A_04 also matches sweeps/raw/reproA5_{closed,bursty}_rep{0,1,2}-*_processed.json.

The A_05 document states that reproMixed 3x artifacts exist, but an inventory of research/measurement finds only the six mixed_{homog_short,homog_long,interleaved}_{c4,c8} raw runs. It does not find the claimed reproMixed files. Additionally, the measured mixed workload contains about 575.4 input and 144 output tokens per measured request, versus about 1022.7 and 256 for homogeneous-long. At the same concurrency, this is a lower-work comparison, not an isolation control.

These findings do not erase the A_01/A_02 latency and throughput curves. They do mean that exact resource percentages, RAM deltas, and the A_05 reproduction claim must not be repeated as settled evidence.

## 2. Observation to Mechanism

The labels below deliberately separate observed facts from candidate mechanisms. A mechanism is not promoted merely because it is plausible.

### A_01 — concurrency knee and tail explosion

**OBSERVED FACT.** On one fixed 512/64 synthetic closed-loop workload, request throughput rises from c1 to c4 but is essentially flat from c4 to c8, while p50 rises from 3.071 to 5.817 s and TTFT rises from about 0.995 to 2.831 s. Fine and coarse sweeps show the same qualitative plateau; the fresh c1-to-c8 p50 effect is about 9.4x.

**MECHANISM HYPOTHESES — NOT CAUSALLY VERIFIED.**

1. Runtime execution serialization or contention: the server uses asyncio.to_thread around incremental CPU generation. Queueing or lock/CPU-cache contention may increase before host-wide CPU looks saturated.
2. Absence of continuous batching: additional request concurrency may create waiting rather than useful batch amortization.
3. Client-side HTTP/runtime contention: a client connection and task are created per request; some delay can occur outside model execution.
4. Cross-session thermal/background drift: this explains absolute throughput changes across hours, but does not explain the low within-session CV or the stable relative shape.

**Existing instrumentation.** Client queue_time, end-to-end latency, TTFT, TPOT, host-wide CPU/RAM, and best-effort NVML are available. Client queue_time is pre-dispatch and cannot explain the measured post-dispatch TTFT/p50 inflation. The remaining measurements support a waiting-dominant interpretation, but cannot distinguish post-dispatch client/HTTP time, executor wait, model serialization, and server queueing. Server queue depth, executor backlog, active workers, per-core CPU/frequency, and batch state are unavailable.

### A_02 — output length tradeoff

**OBSERVED FACT.** With input 512 and c2 fixed, 16x more allowed output lowers request throughput by 60.7 percent and raises p50 by 3.73x, while TTFT and median TPOT are approximately flat and token throughput rises.

**MECHANISM HYPOTHESES — NOT CAUSALLY VERIFIED.**

1. Each extra generated token executes another autoregressive decode step; stable TPOT and output-independent TTFT strongly support this basic model.
2. Fixed prefill/request overhead is amortized over more output tokens, so higher token throughput is not higher request-serving capacity.
3. At c2, longer jobs may add runtime waiting beyond the simple TTFT plus output-times-TPOT expression, but the present instrumentation cannot locate that waiting.

The first two mechanisms fully explain the standalone curve at the present granularity. A_02 is therefore retained as a control dimension, not promoted as a standalone systems problem.

### A_03 — aggregate telemetry versus tail

**OBSERVED FACT.** Across the recorded sweeps, large tail changes are not mirrored proportionally in host-wide CPU/RAM/GPU summaries.

**MECHANISM HYPOTHESES — NOT CAUSALLY VERIFIED.**

1. Host-wide psutil CPU can hide a saturated core, executor, or serialized critical section.
2. Queueing/waiting can enlarge TTFT without consuming CPU time.
3. GPU values are not meaningful on this CPU-inference workload.
4. Aggregation or report extraction errors contribute to the apparent mismatch.

The raw-summary inconsistencies make this a provisional instrumentation finding. It constrains the A_01 causal test; it does not establish an unsaturated-resource bottleneck.

### A_04 and A_05

A_04 does not enter mechanism analysis because its latency/throughput effect failed the repeat gate. A_05 does not enter because it lacks the claimed reproduction artifacts and changes offered work at the same time as workload composition. Neither condition should be converted into a research problem before its trust failure is repaired.

## 3. Literature Positioning

This section uses only the supplied local corpus. It does not claim a complete, current novelty search.

| Observation | Closest local literature | Classification | What existing work explains | What remains unmeasured here |
|---|---|---|---|---|
| A_01 | Orca, FastServe, Sarathi-Serve; Stage 2 Bottlenecks 8/11 and T7 | **Known Phenomenon, Different Regime** | Request/batch granularity, head-of-line waiting, queueing, and throughput-versus-tail tradeoffs are established serving phenomena. | Which boundary is responsible on this non-continuous-batched CPU runtime: post-dispatch client/HTTP behavior, executor/runtime serialization, or model execution; whether the boundary survives a continuous-batching engine. |
| A_02 | Orca, vLLM, FastServe, DistServe; Assumption 2 and T7 | **Fully Explained** as a standalone curve | Autoregressive generation accumulates per-token decode cost; request/s and token/s can move in opposite directions; output length is uncertain at scheduling time. | Only the interaction with the A_01 concurrency knee, which this dataset has not yet measured jointly. |
| A_03 | Mooncake, Llumnix, DistServe; Stage 2 Bottleneck 10 | **Partially Explained** | Modern systems use queue, service, memory, and transfer state instead of treating aggregate CPU as the whole load model. | Whether minimal runtime signals can expose the observed queue-bound transition on this engine, and whether the signal generalizes beyond it. |
| A_04 | Arrival-model assumptions in DistServe/Llumnix and Stage 2 Assumption 6 | Not eligible | Arrival distributions and burst effects are already studied dimensions. | Nothing causal is established by this failed latency repeat. |
| A_05 | Orca/FastServe mixed-length scheduling, SGLang fairness caveat, T7 | Not eligible | Heterogeneous request lengths can affect waiting and tail; cache-affinity starvation requires a cache-aware scheduler absent here. | A load-matched composition effect, which current data cannot isolate. |

No row is classified as Conflicting Literature: the local literature and this tiny-CPU observation are conditioned on different engines, resource models, workloads, and metrics. No row is classified as Apparently Underexplained or novel.

## 4. Engineering Issue versus Research Problem

| Observation | Primary interpretation | Gate decision |
|---|---|---|
| A_01 | System policy / runtime-scale breakdown. The no-continuous-batching design may be the proximate engineering limitation, but the causal queue boundary is not yet measured. | Worth a bounded causal research question. |
| A_02 | Fundamental but expected autoregressive tradeoff. | Keep only as a control variable. |
| A_03 | Instrumentation weakness and reporting-integrity issue, not a discovered hardware bottleneck. | Use as a mandatory measurement constraint. |
| A_04 | Failed latency effect plus client-side queue variation. | Drop. |
| A_05 | Confounded workload comparison plus missing reproduction artifacts. | Drop. |

## 5. Research-Value Gate

The following is a qualitative gate, not a mechanical average.

| Candidate | Evidence | Effect / realism | Generality | Mechanism uncertainty | Prior-work status | Cost to validate | Decision |
|---|---|---|---|---|---|---|---|
| RQ-1: runtime concurrency knee | Strong within one engine | Very large at c4-c8 and realistic enough for a single-node load test | Low until a second engine is tested | High but sharply testable | Known phenomenon, different regime | Low: server/runtime timestamps and c1/c4/c8 controls | Tier 1 |
| RQ-2: output length changes the knee | Strong roots, but no joint interaction measurement | Output and concurrency effects are both large | Low-medium | High; simple null is plausible | A_02 component: Fully Explained; proposed interaction is unmeasured locally, not an underexplained gap | Low-medium: small factorial sweep | Tier 2 |
| RQ-3: signal needed to identify queue-bound tail | Provisional because resource summaries disagree | Important operational consequence if real | Low until cross-engine replication | High | Partially explained | Low: add per-stage tracing before any policy study | Tier 2 |

The A_02 standalone curve is intentionally absent: its large effect is already fully explained. A_04 and A_05 fail the empirical-root requirement and are not candidates.

## 6. Tiering and Next-Stage Decision

### Tier 1

1. **RQ-1 — What runtime boundary produces the concurrency-induced throughput knee and TTFT tail in non-continuous-batched serving?**

This is the only candidate whose root has both a material repeated effect and an unresolved causal locus. Its first Stage 3A contract must add trace points from arrival through semaphore, HTTP/runtime dispatch, executor start, first model step, first token, and completion. It must compare c1/c4/c8 with a fixed workload and record per-core/process CPU and executor state. It must not design a new scheduler or optimization.

### Tier 2

1. **RQ-2 — Does output length change the throughput/tail concurrency knee beyond its expected linear decode cost?**
2. **RQ-3 — Which execution/queue signal distinguishes a queue-bound tail transition from apparently low aggregate utilization?**

Both are worthwhile only after their stated control tests. RQ-2 may collapse to the already-known decode model; RQ-3 may collapse to a measurement hygiene issue on one Python runtime.

### Drop

- A_02 as a standalone RQ: fully explained autoregressive decode baseline.
- A_04: latency/throughput effect is an artifact under the repeat gate.
- A_05: missing claimed replications and offered-work confounding.
- Any revived KV-cache, prefix-cache, P/D, GPU-memory, or cache-affinity question: no such empirical root exists in this engine, and Stage 3C previously rejected its hypothesis-driven evidence.

## Recommendation

**Yes, re-enter Stage 3A, but only as a narrow, falsification-first plan for RQ-1.** RQ-2 and RQ-3 remain Tier 2 until their cheap controls are written and pass. Do not start an optimization, claim a new bottleneck, or proceed to Stage 4. A vLLM/Linux/GPU replication is required before elevating any result from the current runtime to a general LLM-serving conclusion.

## Traceability

- Measurement roots: baseline_summary.md, reliability_review.md, instrumentation_validation.md, sweep_matrix.md, anomaly_index.md, anomalies/A_01.md through A_05.md, negative_observations.md, and stage3rb_summary.md.
- Raw spot checks: sweeps/raw/reproA1_conc{1,4,8}_rep{0,1,2}-*_processed.json; sweeps/raw/reproA2_out{16,64,256}_rep{0,1,2}-*_processed.json; sweeps/raw/reproA5_{closed,bursty}_rep{0,1,2}-*_processed.json; sweeps/raw/sweepA_conc_c064-1787903884_processed.json; raw/baseline_low_concurrency-rep0-1787899654_raw.json.
- Stage 2 framing: taxonomy_final.md, bottleneck_map.md, assumption_map.md, literature_matrix.md, research_tensions.md, stage2_final_review.md, and local notes for Orca, FastServe, Sarathi-Serve, vLLM, DistServe, Mooncake, Llumnix, and SGLang.
