# New Stage 3 Candidates — Observation-Rooted Only

> Date: 2026-08-28  
> Evidence boundary: all empirical roots below come from the CPU-only hf_transformers_naive tiny-gpt2 serving system. The labels rank validation priority, not novelty. No candidate is a proposed optimization, and none is a claim about vLLM, SGLang, PagedAttention, or GPU serving.

## Tier 1

# RQ-1 — What Runtime Boundary Produces the Concurrency-Induced Throughput Knee and TTFT Tail in Non-Continuous-Batched Serving?

## Empirical Observation

On synthetic 512-input/64-output requests with closed-loop concurrency and 40 requests (38 measured after warmup), fresh 3x repeats show:

- c1: 0.673 plus-or-minus 0.003 rps, p50 0.619 plus-or-minus 0.004 s, p95 0.669 plus-or-minus 0.010 s, TTFT about 0.060 s.
- c4: 0.885 plus-or-minus 0.004 rps, p50 3.071 plus-or-minus 0.169 s.
- c8: 0.873 plus-or-minus 0.015 rps, p50 5.817 plus-or-minus 0.160 s, p95 8.829 plus-or-minus 0.37 s, TTFT about 2.831 s.

Thus c1 to c8 buys only 29.7 percent request-throughput growth, while p50 increases about 9.4x and TTFT about 47x. Fine and coarse sweeps show the same plateau/tail shape; the cautious boundary is around c3-c4, not an exact universal c3.

Source: anomalies/A_01.md; sweep_matrix.md; sweeps/raw/reproA1_conc{1,4,8}_rep{0,1,2}-*_processed.json.

## Reproducibility

The three fresh repeats per c1/c4/c8 point all complete 38/38 requests. Throughput CV is 0.4-1.7 percent and p50 CV is 0.6-5.5 percent, far below the c1-to-c8 p50 effect. The relative shape also appears in baseline, coarse, and fine sweeps.

Absolute throughput is not stable across sessions: c1 is reported as 0.673, 0.864, and 1.512 rps in sessions separated by hours. All causal claims must therefore use same-session comparisons and record the session/provenance.

## System / Workload Scope

- System: hf_transformers_naive with transformers 5.16.1 and torch 2.13.0+cpu; no continuous batching, no prefix cache, no PagedAttention.
- Hardware: i7-14650HX CPU, Windows 11, one host; tiny-gpt2, FP32.
- Workload: deterministic synthetic filler, 512 input/64 output, closed arrivals, c1-c64. The c4-c8 portion is the relevant single-node operating range; c32-c64 is stress characterization.
- Not established for another model, GPU, scheduler, open-loop workload, input/output pair, or production trace.

## Candidate Mechanism

**NOT YET CAUSALLY VERIFIED.** A runtime queue or serialized execution path begins to dominate around c3-c4: additional request concurrency adds wait before useful service rather than useful parallel work. The likely contributors are incremental asyncio.to_thread execution, Torch/Python thread contention, and the absence of continuous batching.

## Competing Explanations

1. The dominant delay is post-dispatch client-side HTTP/task contention, not a server/runtime queue.
2. Host-wide CPU aggregates hide one saturated core, a lock, or thread oversubscription; the mechanism is local CPU contention rather than a generic serving scheduler boundary.
3. Session-level thermal/background drift changes the apparent knee, even if within-session repetition is stable.
4. No-continuous-batching itself explains the plateau, but does not identify the stage responsible for the large TTFT tail.

## Relevant Prior Work

- Orca: request-granularity batches make new arrivals wait; iteration-level scheduling reduces this queueing behavior.
- FastServe: FCFS/head-of-line blocking, heterogeneous service times, and tail-goodput tradeoffs; its scheduler uses partial length information.
- Sarathi-Serve: stall-free chunked/hybrid scheduling and prefill/decode interference under SLOs.
- Stage 2: Bottleneck 8 (scheduler inefficiency), Bottleneck 11 (tail latency), and T7 (throughput/utilization versus interactive tail SLO).

## What Existing Work Explains

The general queueing, head-of-line, and throughput-versus-tail mechanisms are already known. Existing work also shows that scheduling granularity and batching can move the tradeoff. Therefore this question is not an “unexplained tail-latency” or novelty claim.

## What Remains Unexplained

The current system has no server queue, executor, worker, per-core, or model-stage tracing. It cannot yet say whether the observed c3-c4 boundary comes from client admission, asyncio/thread execution, Torch/model serialization, or absent batching. It also cannot say whether the boundary survives a continuous-batching engine.

## Research Question

For a fixed workload on a non-continuous-batched serving runtime, which execution-stage queue or serialized state defines the request-concurrency knee at which request throughput stops improving but TTFT and tail latency rise sharply, and does the same boundary persist after changing the execution model?

## Why It Matters

If the boundary is real, request-count admission can create severe user-facing TTFT/P95/P99 regressions without a commensurate request-throughput gain. Locating the actual queue is a prerequisite for any reliable SLO, admission, or autoscaling evaluation; using only aggregate utilization could select the wrong intervention.

## Falsifiable Hypothesis

At c4-c8, one instrumented runtime stage will show a sharply increasing wait/backlog contribution that explains the TTFT growth better than host-wide CPU/RAM averages. Changing only that stage's controlled capacity or execution mode will move the knee. If the additional delay is entirely in a post-dispatch client/HTTP path, or no stage-specific state changes with the tail transition, the stated runtime-boundary hypothesis is falsified or narrowed.

## Cheapest Causal Test

Add timestamped trace points for client arrival, semaphore acquisition, server receipt, executor submission/start/end, first model forward step, first SSE token, and completion. At c1/c4/c8, run three same-session repetitions with the existing 512/64 workload; collect per-core/process CPU, Torch thread settings, and executor work-queue/active-worker state. A small controlled execution-capacity or batching-mode ablation can then distinguish pre-dispatch admission and post-dispatch client/HTTP time from executor/model serialization without introducing a new scheduler.

## Risk

- Already-known queueing phenomenon rather than a research contribution.
- Python/tiny-CPU runtime specific.
- Cross-session performance drift and absent code hashes weaken provenance.
- Instrumentation may expose a simple engineering limitation rather than a general systems boundary.

## Priority

**High** for causal validation, not for novelty.

## Tier 2

# RQ-2 — Does Output Length Change the Throughput/Tail Concurrency Knee Beyond Its Expected Linear Decode Cost?

## Empirical Observation

Two reliable but separate sweeps establish the roots:

- With input 512 and c2 fixed, output 16 to 256 decreases request throughput from 1.071 plus-or-minus 0.050 to 0.421 plus-or-minus 0.001 rps and raises p50 from 1.049 plus-or-minus 0.012 to 3.910 plus-or-minus 0.033 s. TTFT is approximately 0.388 to 0.430 s and median TPOT remains near 0.010-0.012 s.
- With 512/64 fixed, the c1-to-c8 sweep produces the RQ-1 throughput knee and p50 tail increase.

The output-length curve and the concurrency curve were not measured as a joint factorial interaction. This is a candidate derived from two observed facts, not an observed interaction.

Source: anomalies/A_01.md and A_02.md; sweeps/raw/reproA1_* and reproA2_* processed files.

## Reproducibility

For output 16/64/256, the three-repeat throughput CVs are 4.7, 1.1, and 0.2 percent, while p50 CVs are 1.1, 0.5, and 0.9 percent. Actual generated output lengths in the raw records equal their configured values, so early EOS does not explain the endpoints. The c1/c4/c8 concurrency roots are independently reproduced as stated in RQ-1.

## System / Workload Scope

- Same CPU-only no-continuous-batching engine as RQ-1.
- Synthetic 512-token input, output 16-256, closed-loop c2 for the output sweep; synthetic 512/64 at c1-c64 for the concurrency sweep.
- Not yet measured for output-length heterogeneity, input/output combinations, real traces, continuous batching, or cache-enabled scheduling.

## Candidate Mechanism

**NOT YET CAUSALLY VERIFIED.** A fixed request-count limit may reach its throughput/tail knee earlier for longer output jobs because the longer service lifetime raises queueing and TTFT before request throughput reveals the loss. The standalone decode component is expected; the candidate mechanism is an output-length-by-concurrency interaction beyond that linear component.

## Competing Explanations

1. There is no interaction: total latency is simply TTFT plus output times stable TPOT, and the concurrency knee remains unchanged after normalizing service work.
2. The apparent interaction would arise only from this runtime's incremental CPU execution or connection behavior.
3. Token throughput rises because fixed request overhead is amortized, not because long requests improve capacity; request/s and SLO metrics may already capture all relevant behavior.

## Relevant Prior Work

- vLLM: output length and KV growth are unknown at scheduling time; iteration-level scheduling and memory management address this uncertainty.
- Orca: generation requires repeated token iterations rather than one request-level computation.
- FastServe: uses input-length/first-step information, treats output length as uncertain, and evaluates input/output-ratio effects under tail-goodput objectives.
- DistServe: separates TTFT and TPOT/goodput by workload and SLO.

## What Existing Work Explains

Autoregressive decode cost, unknown output length, and the difference between request throughput and token throughput are established. A_02 alone is fully explained and is deliberately not promoted as a standalone RQ.

## What Remains Unexplained

The local record does not test whether output length shifts the count-based concurrency knee, whether that shift survives a service-work-normalized comparison, or whether continuous batching changes it. It therefore has no evidence for a new policy or universal output-aware scheduler.

## Research Question

Holding model, input length, and admission mechanism fixed, does output length or output-length heterogeneity move the concurrency boundary at which request-level TTFT/P95/P99 deteriorates without meaningful request-throughput gain beyond the expected linear decode cost?

## Why It Matters

Serving systems often report tokens/s or average throughput while interactive clients experience requests and tail SLOs. A verified interaction would identify when a simple request-count limit becomes misleading under a realistic range of response lengths; a null result would prevent overclaiming an output-aware mechanism.

## Falsifiable Hypothesis

At fixed input 512, longer outputs will shift the throughput/tail concurrency knee to a lower request count or amplify the tail above the additive TTFT-plus-TPOT prediction. If tail curves collapse after accounting for service work and no knee shift appears over the factorial sweep, the interaction hypothesis is rejected.

## Cheapest Causal Test

Run a 3-by-3 matrix: output 16/64/256 by c1/c4/c8, with three same-session repeats, identical input and warmup, and the RQ-1 stage timestamps. Report request/s, tokens/s, TTFT/TPOT P50/P95/P99, stage-specific wait, and a declared SLO-goodput measure. Add one token-work-normalized control before interpreting a long-output tail as a scheduling effect.

## Risk

- The entire result may reduce to an already-known linear decode model.
- Current evidence has no joint interaction and only a single runtime.
- Output length is partly a configured maximum, not a production response-length distribution.
- Novelty potential is low without a cross-engine, mixed-workload confirmation.

## Priority

**Medium.** Promote only if the cheap factorial control rejects the additive baseline.

# RQ-3 — Which Execution Signals Distinguish a Queue-Bound Tail Transition from Apparently Low Aggregate Utilization?

## Empirical Observation

The same concurrency sweeps show very large latency changes while the available aggregate resource metrics do not change proportionally. In sampled fresh processed repeats, c1 has p50 0.619 s and host-wide CPU mean about 44.9 percent, while c8 has p50 5.849 s and host-wide CPU mean about 43.3 percent. Other sessions provide different CPU summaries: baseline_summary lists roughly 13-18 percent, while a sampled baseline raw file averages 74.74 percent.

The robust observation is not an exact CPU percentage. It is that the available aggregate telemetry cannot explain or localize the tail transition; server queue, executor state, active requests, per-core CPU, and batch/KV state are absent.

Source: anomalies/A_03.md; instrumentation_validation.md; sweeps/raw/reproA1_conc1_rep0-1787906559_processed.json; sweeps/raw/reproA1_conc8_rep0-1787906877_processed.json; raw/baseline_low_concurrency-rep0-1787899654_raw.json.

## Reproducibility

The tail pattern itself is independently reproduced by A_01. The telemetry mismatch is observed across concurrency and output sweeps, but exact CPU/RAM figures are not publication-grade because summary and raw records disagree. This RQ must start by reconciling telemetry provenance and collecting missing signals.

## System / Workload Scope

- CPU-only hf_transformers_naive serving; host-wide psutil CPU/RAM and best-effort WDDM NVML.
- Synthetic 512/64 concurrency sweep and 512-input output sweep.
- No evidence yet for GPU inference, continuous batching, a distributed queue, cache pressure, or an autoscaler.

## Candidate Mechanism

**NOT YET CAUSALLY VERIFIED.** The critical execution path is queue/wait or localized serialized work, whereas host-wide aggregate utilization averages away the causal state. The useful signal could be executor backlog, active model calls, per-core saturation, or a decomposition of client versus server queue time.

## Competing Explanations

1. A few CPU cores are saturated but global CPU averaging hides them.
2. The delay is waiting outside CPU execution, such as post-dispatch client/HTTP runtime contention.
3. Sampling interval, CPU accounting semantics, or a report-extraction mistake creates the apparent mismatch.
4. The observation is unique to CPU/GIL-style execution and disappears in a GPU continuous-batching engine.

## Relevant Prior Work

- Mooncake models queue, prefill, and transfer time rather than only utilization.
- DistServe includes queue terms in its latency/goodput model.
- Llumnix uses virtual memory availability/freeness rather than a single utilization percentage.
- Stage 2 Bottleneck 10 explicitly uses queue/service terms; T7 motivates reporting tail-SLO outcomes.

## What Existing Work Explains

Existing serving systems already use richer queue/service state and do not treat aggregate CPU as a complete load model. The local literature therefore does not support a claim that queue-aware observability is absent.

## What Remains Unexplained

There is no causal mapping from the present runtime's host-wide telemetry to its tail behavior. It is unmeasured whether a minimal stage-specific signal can identify the transition early and whether that relationship holds on a different engine.

## Research Question

In non-continuous-batched serving, which minimal execution/queue signals distinguish a queue-bound TTFT/tail transition from a resource-bound transition when host-wide aggregate utilization remains ambiguous?

## Why It Matters

An incorrect diagnosis can turn a tail problem into an inappropriate CPU, RAM, GPU, or autoscaling response. A small, validated signal set would also make cross-engine comparisons more interpretable, provided it survives replication.

## Falsifiable Hypothesis

Stage-specific queue/executor signals will track and predict the c1/c4/c8 tail transition substantially better than host-wide CPU/RAM alone. If reconciled telemetry or stage tracing shows that aggregate utilization already identifies the same transition equally well, or no queue/serialized state changes, this RQ is reduced to reporting hygiene.

## Cheapest Causal Test

Use the RQ-1 trace instrumentation, but preregister a predictive comparison: aggregate CPU/RAM alone versus aggregate plus per-core/process CPU, executor backlog/active workers, server active requests, and client/server queue decomposition. Train or fit no controller; simply test which signal set separates c1/c4/c8 with held-out repeats and reconcile every summary value against its raw samples.

## Risk

- May be an instrumentation/accounting repair rather than a systems research result.
- Strongly coupled to RQ-1 and to one Python runtime.
- Existing systems may already solve the practical observability problem.
- Summary/raw inconsistencies must be fixed before claiming an effect size.

## Priority

**Medium.** It is a required measurement companion to RQ-1, not an independent optimization direction.
