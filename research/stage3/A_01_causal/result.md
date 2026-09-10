# A_01 causal-localization result

## Final classification

DESIGN INVALID

## Stopping rule applied

The locked instrumentation OFF/ON gate failed.  Per `LOCKED_CAUSAL_PLAN.md`,
no formal trace-ON c=1/4/8 repetition, pooled-client control, or internal
latency-component interpretation was run after this failure.

All 12 matched gate runs completed 38/38 requests on one unchanged real CPU
server process (PID 18968); this establishes transport/trace correlation
integrity, but it does not establish measurement validity.

## Gate evidence

| Concurrency | Matched seed | OFF -> ON p95 TTFT change | OFF -> ON p95 total-latency change |
|---:|---:|---:|---:|
| 1 | 3101 | +163.72% (0.099208 s -> 0.261632 s) | +85.99% (0.363684 s -> 0.676433 s) |
| 1 | 3102 | +4.64% (0.081959 s -> 0.085764 s) | +9.94% (0.323062 s -> 0.355166 s) |
| 4 | 3401 | +4.54% | +3.31% |
| 4 | 3402 | -1.57% | -13.63% |
| 8 | 3801 | -2.30% | -5.93% |
| 8 | 3802 | -1.30% | -4.50% |

The precommitted gate rejects this dataset because:

- c=1 p95 TTFT median paired difference was +84.18% (limit: 5%), with one
  individual pair +163.72% (limit: 10%);
- c=1 p95 total latency median paired difference was +47.97%, with one
  individual pair +85.99%;
- c=4 and c=8 p95 total-latency median paired differences also exceeded the
  5% criterion; and
- the c4/c1 p95 TTFT multiplier changed by -47.10% between tracing OFF and
  ON (limit: 10%).

The complete machine-readable decision is
`processed/overhead_gate.json`; all individual raw requests, health snapshots,
and run manifests remain under `raw/`.

## Alternative explanations and measurement boundaries

- The strongest live alternative is tracing interaction and/or intra-session
  non-stationarity/run order.  Counterbalancing was insufficient to meet the
  precommitted stability criterion, so these cannot be separated.
- Client/server ingress is deliberately not cross-process-subtracted.  A
  pooled-client control was not run because the formal phase was blocked; it
  must not be retrofitted as a causal explanation for this invalid design.
- PID, model, device, precision, workload, seeds, and request success remained
  fixed.  CPU, CPU-frequency, server RSS/threads, RAM, and GPU telemetry were
  recorded, but their presence cannot rescue a failed tracing gate.
- GPU model execution is not applicable: the locked server was CPU FP32.
  Continuous-batching state and a scheduler queue depth are not exposed by
  this runtime.

## Decision

A_01 disposition: INCONCLUSIVE.

Re-enter Stage 3B causal validation: No.  A new, independently validated
instrumentation design would be required before repeating Stage 3A; this run
does not localize a runtime, executor, or model-execution boundary.
