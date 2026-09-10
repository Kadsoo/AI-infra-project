# Instrumentation smoke (not a formal measurement)

- Run ID: a01-instrumentation-smoke
- Request ID: a01-smoke-001
- Server: PID 18968 at http://127.0.0.1:8017
- HTTP status: 200
- Response echoed both correlation headers.
- Retrieved one trace with the same run/request IDs.
- Trace contained every required T1–T7 internal field, preparation timestamps,
  and four model-step records.
- Observed ordering was monotonic from handler entry through server completion.

This smoke only validates trace wiring.  It is excluded from the overhead gate,
formal repetitions, all aggregates, and causal interpretation.
