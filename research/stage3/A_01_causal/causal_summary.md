# A_01 causal summary

1. **Concurrency knee reproduced:** not assessed under valid tracing.  The
   OFF/ON gate intentionally precedes the formal knee-reproduction matrix and
   failed, so its observations are not a valid A_01 replication.
2. **Latency component containing the knee:** not determined.  No internal
   component statistics were interpreted after the gate failure.
3. **Effect size:** no causal effect size is reportable.  The relevant measured
   effect is the invalidating overhead instability: at c=1/seed 3101, tracing
   ON increased p95 TTFT by 163.72% and p95 total latency by 85.99%.
4. **Stability:** failed.  The c4/c1 p95 TTFT knee multiplier changed -47.10%
   between OFF and ON, beyond the locked 10% limit.
5. **Strongest alternative explanation:** tracing interaction and/or
   intra-session run-order/host non-stationarity.  The data cannot distinguish
   the two safely.
6. **Concrete runtime boundary localized:** no.
7. **A_01 disposition:** INCONCLUSIVE (formal causal classification: DESIGN
   INVALID).
8. **Return to Stage 3B causal validation:** no.  First replace or isolate the
   tracing design and rerun its OFF/ON validity gate under a newly versioned
   locked plan.
