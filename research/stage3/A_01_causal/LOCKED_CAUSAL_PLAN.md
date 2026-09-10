# LOCKED CAUSAL PLAN — A_01 concurrency knee

> Status: LOCKED BEFORE FORMAL MEASUREMENT
>
> Scope: Focused Stage 3A causal localization only.  This plan may not be
> changed after the first formal trace run.  A material change requires a new
> LOCKED_CAUSAL_PLAN_v2.md that names the reason and preserves this file.

## 1. Question and allowed conclusions

Question: when the observed concurrency knee is crossed in the local real
serving shim, which measured execution stage accounts for the tail-latency
turn?

Permitted final classifications are exactly:

- LOCALIZED — RUNTIME
- LOCALIZED — EXECUTOR QUEUE
- LOCALIZED — MODEL EXECUTION
- MULTI-STAGE
- ORDINARY QUEUEING
- INCONCLUSIVE
- DESIGN INVALID

No scheduler change, queue optimization, model change, algorithm, or tuning is
permitted.  A result applies only to this CPU, non-continuous-batched runtime.

## 2. Fixed system and workload

| Control | Locked value |
|---|---|
| Server | single FastAPI/Uvicorn process running hf_transformers_naive |
| Model | sshleifer/tiny-gpt2 |
| Device and precision | CPU, FP32; health must report mock=false and device=cpu |
| Generation | streaming SSE, temperature 0.0, maximum 64 tokens |
| Workload | synthetic deterministic text, target 512 input tokens, no prefix reuse |
| Measured request count | 38 requests per run |
| Warmup | 2 sequential requests, excluded from statistics |
| Total request count | 40 including warmup |
| Arrival semantics | all measured tasks are scheduled at run start; client semaphore admits at most c in-flight HTTP requests; this is a finite client-gated burst, not a generic open-loop trace |
| HTTP primary mode | one HTTP/1.1 AsyncClient per request, matching the original observation |
| Sampler | 0.3 s host resource sampler in every OFF and ON run |
| Timeouts | 180 s request timeout |
| Session | one newly launched traced server process on a new unused localhost port; it is not the pre-existing port-8000 process |

Before the first run, logs must capture the start command, PID, port,
health response, environment hash, and full SHA-256 of hf_server.py,
harness.py, generator.py, the causal runner, and this lock file.

## 3. Fixed concurrency points and repetitions

The only load values are c=1, c=4, and c=8.  c=1 is below the previously
observed knee, c=4 is at/just above it, and c=8 is above it.  No sweep and no
other workload is authorized.

Formal trace-ON repetitions use three interleaved blocks in this exact order:

| Block | Run order | c=1 seed | c=4 seed | c=8 seed |
|---|---|---:|---:|---:|
| 1 | 1, 4, 8 | 1010 | 1040 | 1080 |
| 2 | 4, 8, 1 | 1011 | 1041 | 1081 |
| 3 | 8, 1, 4 | 1012 | 1042 | 1082 |

Every formal run must complete 38/38 measured requests.  A failed request,
missing correlated server trace, or negative/non-monotonic timestamp in more
than 5% of measured requests makes the formal result INCONCLUSIVE; more than
5% in any single run invalidates that run and prevents a three-repeat result.

## 4. Hypotheses and falsifiers

H1, H2, H3, H4, the multi-stage condition, and all operational support and
falsification rules are fixed in hypothesis.md.  The key categories are:

| Hypothesis | Required dominant component |
|---|---|
| H1 runtime | pre-executor runtime component |
| H2 executor queue | direct submit-to-worker-start interval |
| H3 model execution | worker-local / Torch forward intervals after worker start |
| H4 | no stable local component transition |

For TTFT component share, H1 includes handler/runtime intervals through the
first executor submission, H2 includes preparation plus first-step executor
wait, and H3 includes first worker-to-forward plus first forward wall.
Later-step waits and forwards are completion-side corroboration only; they are
not added to the server-time-to-first-content denominator.

The GPU alternative is not testable because the locked system is CPU inference.
It must never be reported as GPU model execution.

## 5. Timestamp and metric contract

The T0–T7 map, raw fields, and additive formulas are fixed in
instrumentation_plan.md.  Client and server clocks are not subtracted across
processes.  The following summary statistics are reported for every latency
component at each concurrency: count, mean, median, p50, p95, p99, sample
standard deviation, variance, minimum, and maximum.

Primary tail outcomes:

- end-to-end TTFT p50/p95/p99;
- client total latency p50/p95/p99;
- server time-to-first-content p50/p95/p99;
- each additive server-first-content component p50/p95/p99;
- first-step and later-step executor-wait and forward wall distributions;
- server decode/completion p50/p95/p99.

For causal shares, use the mean of per-request additive components:

component_share_i =
  (mean_i(c8) - mean_i(c1)) /
  (mean_server_first_content(c8) - mean_server_first_content(c1)).

If the denominator is not positive, no component can be called a causal
increase and the knee is not localized.

## 6. Knee-reproduction rule

The knee reproduces only if all of the following hold in the trace-ON formal
session:

1. all nine runs have 38/38 measured successes;
2. the median across the three run-level p95 end-to-end TTFT values has
   c4/c1 >= 2.0 and c8/c4 >= 1.25;
3. the median across the three run-level throughput values has
   c8/c4 <= 1.10; and
4. the c4 and c8 TTFT increases are in the same direction in at least two
   blocks.

If the knee does not reproduce, the final classification is INCONCLUSIVE,
not a null causal localization.

## 7. Stage-localization decision rule

For a component to be a localized cause, all apply:

1. its p95 has c4/c1 >= 1.5 and c8/c4 >= 1.25;
2. its c1-to-c8 mean increase has a non-parametric bootstrap 95% interval
   above zero, using 10,000 resamples stratified by run;
3. the direction and at least 25% component share hold in two of three
   blocks; and
4. its pooled component share is at least 50%.

The H1, H2, and H3 dominant components are then classified as LOCALIZED —
RUNTIME, LOCALIZED — EXECUTOR QUEUE, and LOCALIZED — MODEL EXECUTION
respectively.  Two or more components satisfying at least 25% each and 70%
jointly yield MULTI-STAGE.  A reproduced knee with no component satisfying
these rules yields ORDINARY QUEUEING.

If correlated tracing, clock ordering, client control, or resource controls
cannot distinguish the alternatives, yield INCONCLUSIVE.  If the overhead
gate fails, yield DESIGN INVALID without interpreting components.

## 8. Instrumentation ON/OFF gate

For each c in {1,4,8}, use these fixed matched-seed orders:

| c | seed A order | seed B order |
|---:|---|---|
| 1 | 3101: OFF then ON | 3102: ON then OFF |
| 4 | 3401: OFF then ON | 3402: ON then OFF |
| 8 | 3801: OFF then ON | 3802: ON then OFF |

The exact pass/fail thresholds are in instrumentation_plan.md.  The formal
trace-ON blocks may begin only after every overhead criterion passes.

## 9. Required alternative-explanation checks

| Alternative | Precommitted check | Consequence |
|---|---|---|
| Network/client / HTTP connection | record client-send-to-headers and run pooled-client c=1/4/8 control after formal runs | if pooled mode changes the end-to-end c8/c1 p95 TTFT increase by more than 30% while all server components stay within 10%, server conclusion is INCONCLUSIVE and client/HTTP is the strongest explanation |
| Request generator / semaphore | log scheduled arrival, slot-acquired time, client send, actual request count, and all seeds | client semaphore wait is reported separately and excluded from server components |
| Batching | log no continuous batching and unavailable batch state | no batching conclusion beyond this engine |
| Warmup | 2 sequential excluded requests before every run | a warmup failure invalidates its run |
| GPU utilization | log health device and NVML samples | because device=cpu, no GPU-cause conclusion is allowed |
| Memory pressure | report host RAM and server RSS before/after every run | material memory growth without a stage-local signal is a confounder, not localization |
| Scheduler defaults | log default asyncio executor and no explicit scheduler | do not invent queue depth or scheduler policy |
| CPU contention | report host CPU, CPU frequency, process RSS/thread count and Torch thread settings | a process/config change across runs invalidates the session |

The pooled-client control is three trace-ON runs, one at each c, with seeds
4101, 4401, and 4801 and all other controls unchanged.  It is a control, not
a new workload or concurrency sweep.

## 10. Output contract

The following artifacts are required under research/stage3/A_01_causal:

- hypothesis.md
- instrumentation_plan.md
- LOCKED_CAUSAL_PLAN.md
- raw/ per-run client records, server traces, health snapshots, and config
- processed/ overhead gate, formal statistics, bootstrap, and control summary
- logs/ launch, commands, hashes, and session events
- result.md
- causal_summary.md

The final causal_summary.md must answer, in explicit terms: reproduction,
component, effect size, stability, strongest alternative explanation, whether
a concrete runtime boundary was localized, A_01 disposition (PROCEED, NEEDS
ONE CONTROL, ORDINARY QUEUEING / DROP, or INCONCLUSIVE), and whether Stage 3B
causal validation is warranted.
