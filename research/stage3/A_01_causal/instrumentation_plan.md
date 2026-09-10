# A_01 — Minimal instrumentation plan

## Implementation boundary

The target is research/measurement/harness/hf_server.py and its existing
research/measurement/harness/harness.py client.  The implementation adds
request-local monotonic timestamps, a stable request/run correlation header,
and one post-run trace retrieval endpoint.  It does not replace the default
asyncio executor, inspect private executor queue state, change worker count,
change scheduling policy, add profiling, synchronize CUDA, or write from the
token hot path.

All server-internal timestamps use time.perf_counter_ns().  All client
timestamps use the same clock only within the client process.  Cross-process
subtraction is prohibited: client-send to server-handler is reported as
NOT RELIABLY AVAILABLE rather than as a fabricated network interval.

## Correlation and collection

For every measured and warmup request, the client sends:

- X-Stage3-Run-Id: immutable run identifier
- X-Stage3-Request-Id: immutable request identifier

The server preserves both values in its request-local trace.  When tracing is
enabled, it stores the completed request trace in memory once, after the
streaming generator has finished.  The client retrieves that run's traces
only after the workload has completed.  There is no print, synchronous file
I/O, lock acquisition on every token, or profiler in the request path.

Trace OFF retains the same headers and request shape but creates no trace
object and writes no trace records.  This makes the OFF/ON comparison isolate
the extra tracing work rather than the header shape.

## Timestamp mapping

| Required point | Measured field | Concrete location | Reliability note |
|---|---|---|---|
| T0 | t_client_send_ns | immediately before client.stream POST | reliable client monotonic time |
| T1 | t_handler_enter_ns | first executable statement of chat_completions | FastAPI handler entry, not NIC receive time |
| T2 | t_stream_runtime_enter_ns | first statement of gen_real | closest available internal-runtime entry; no separate scheduler exists |
| T3 | t_first_step_submit_ns | immediately before first model-step asyncio.to_thread | reliable executor submission time |
| T4 | t_first_step_worker_start_ns | first statement inside first _one_step worker closure | reliable actual default-executor worker entry |
| T5 | t_first_forward_begin_ns | immediately before model(...) | reliable CPU/Torch forward start |
| T6 | t_first_content_yield_ns | immediately before first non-empty SSE content yield | server first content; client separately records first content |
| T7 | t_server_done_ns / t_client_done_ns | server streaming generator finalization / client stream completion | separately clocked endpoints |

Preparation is separately traced because tokenization itself uses
asyncio.to_thread:

- t_prepare_submit_ns
- t_prepare_worker_start_ns
- t_prepare_worker_done_ns
- t_prepare_resume_ns

Every model step records submit, worker start, forward begin, forward end,
sampled-token time, and coroutine-resume/yield time.  The completed request
contains the first-step fields plus all-step count, sums, mean, p95, and max
for executor wait, worker-to-forward time, and forward wall time.  Raw JSON
retains the individual step records.

## Derived intervals

The following formulas are frozen.  All server terms are non-negative
durations in ns converted to seconds only for reporting.

| Name | Formula |
|---|---|
| client semaphore wait | t_client_slot_acquired - t_client_scheduled_arrival |
| client headers/connection interval | t_client_headers_ns - t_client_send_ns |
| end-to-end TTFT | t_client_first_content_ns - t_client_send_ns |
| client total latency | t_client_done_ns - t_client_send_ns |
| handler-to-runtime | t_stream_runtime_enter_ns - t_handler_enter_ns |
| runtime before preparation submit | t_prepare_submit_ns - t_stream_runtime_enter_ns |
| preparation executor wait | t_prepare_worker_start_ns - t_prepare_submit_ns |
| preparation execution | t_prepare_worker_done_ns - t_prepare_worker_start_ns |
| preparation resume delay | t_prepare_resume_ns - t_prepare_worker_done_ns |
| runtime between preparation and first step | t_first_step_submit_ns - t_prepare_resume_ns |
| first-step executor wait | t_first_step_worker_start_ns - t_first_step_submit_ns |
| worker-to-first-forward | t_first_forward_begin_ns - t_first_step_worker_start_ns |
| first-forward wall | t_first_forward_end_ns - t_first_forward_begin_ns |
| model-to-first-content | t_first_content_yield_ns - t_first_forward_end_ns |
| server time to first content | t_first_content_yield_ns - t_handler_enter_ns |
| decode/completion | t_server_done_ns - t_first_content_yield_ns |
| server total | t_server_done_ns - t_handler_enter_ns |

For time-to-first-content attribution, pre-executor is handler-to-runtime plus
runtime-before-preparation-submit plus preparation-resume-delay plus
runtime-between-preparation-and-first-step.  The TTFT executor component is
preparation executor wait plus first-step executor wait; the TTFT model
component is worker-to-first-forward plus first-forward wall.  All later-step
executor waits and forward walls are reported separately for decode/completion
diagnosis and are never added into a TTFT component share.

## Resource and alternative-explanation evidence

Each run records:

- server health before and after: PID, model, mock flag, device, dtype,
  torch/transformers versions, Torch intra/inter-op thread settings, source
  hashes, and trace state;
- client host CPU, RAM, CPU-frequency, and best-effort GPU samples at the
  already fixed 0.3 s interval;
- server process RSS and active Python-thread count at health capture;
- actual request success, effective input/output token counts, response
  headers time, and client semaphore wait;
- stream=true, no continuous batching, default asyncio executor, and no
  exposed scheduler/batch/KV state as explicit NOT AVAILABLE fields.

The client harness deliberately retains one AsyncClient per request in the
primary runs so that the previously observed protocol is reproduced.  A
pooled-client control uses one AsyncClient with HTTP/1.1 and limits at least
eight connections, at c=1/4/8, after the formal runs.

## Overhead gate

For each of c=1, c=4, and c=8, two matched workload seeds are run in a
counterbalanced OFF/ON order.  The same headers, warmup, workload, source
hash, process, port, and sampler are used.  Trace mode can change only
between runs.

The gate passes only when, at every concurrency:

1. the absolute median paired relative difference is at most 5% for
   throughput, p95 end-to-end TTFT, and p95 total latency;
2. neither individual paired relative difference exceeds 10% for any of the
   same three metrics; and
3. the OFF-to-ON relative change in the c4/c1 and c8/c4 p95 TTFT multipliers
   is at most 10%.

Failure of any condition is DESIGN INVALID.  Formal causal interpretation
then stops immediately.
