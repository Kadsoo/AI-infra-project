# RQ-4 — future solution notes (exploratory only, NOT implemented)

> Stage 3B instruction §13: record only; no implementation. Bottleneck identified: hot-node queue hotspot (hot P99 2.8–10× load-only; SLO crossing at p_hot ≥ 0.7, c ≥ 16), with the aggregate mean simultaneously degraded (1.7–2.7×) and naive k=2 replication insufficient (restores within 30% at 1/13 cells).

Very short candidate directions (in no priority order):

- Replication policy with demand-aware replica count / adaptive θ (k as a function of hot request rate), evaluated on the H3 restoration line (≤1.3× of load-only).
- Routing scoring that blends affinity with backlog (hot requests to the hot node only when its predicted wait stays under a threshold) — a "soft affinity" family between A and B.
- Per-class SLO-aware admission or dispatch priority to cap hot-class P99 without stranding cold-class attainment.
- Measure first: E4 (real system, ≥3 nodes) before any of these; within-node scheduling order (cold-class starvation) is unmodeled and may dominate. E4 step-0 length–service curve fit is a prerequisite.