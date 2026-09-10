# RQ-1 Stage 3B Experiment Log

> All experiments in this log follow `LOCKED_PLAN.md` (frozen 2026-08-27). Environment: `../environment.md` (Windows 11, i7-14650HX, 32 GB RAM, CPU-only; no A100, no git repo). Wave-1 scope only; GPU gates E3/E4/E5 blocked by hardware.

## E1 — Transfer-materiality cost model (CPU)

- **Timestamp:** 2026-08-27
- **Command:** `python code\trace_gen.py --lam 3.0 --out raw\trace_5000.csv` then `python code\e1_cost_model.py`
- **Config:** trace seed 20260827, λ=3.0; f×{0.2,0.5,1.0} × ρ×{0.3,0.5,0.7} × prefill×{0.5,1,2} × class×{long,mixed} = 72 cells + NVLink pair; corpus anchor f=0.24, ρ=0.5.
- **Purpose:** arithmetic-support gate for H1; reachable kill lines.
- **Result:** SUCCESS. H1 not killed; pre-registered "narrow band → E3 required" (fraction 4.6–72.0%, gap 4.8–256% across cells); anchored point supports (P95 frac 41.8%, P50 41.8%, NVLink 0.09%, gap 71.6%).
- **Outputs:** `raw/e1_cells.csv`, `raw/e1_anchor.json`, `raw/e1_verdict.json`, `processed/e1_report.md`
- **Script SHA-256:** e1_cost_model.py `23d916fce0aac948fbc42ecfc9a8feb6c76c567b5c39da3c4c05daa229c519ba`; trace_gen.py `9817a2387bb8a0d9db4a338aff57aff5e5565db79cbd3df56334f140c487e71a`
- **Notes:** bugs fixed during bring-up: (1) `df.quantile` collided with pandas' `.quantile()` method (IndexError) — fixed; (2) anchored f initially solved from P95 length instead of the plan's f=0.24 — corrected to the frozen anchor. Trace regenerated at the E2-chosen λ after E1 first ran; length statistics are seed-identical (verified), E1 re-run on the final trace.

## E2 — Calibrated queue simulator, H2 gate (CPU)

- **Timestamp:** 2026-08-27
- **Command:** `python code\e2_run.py` (probe → trace regen → 225 P/D runs + 45 colocated runs); then `python code\e2_diag_lowload.py` (diagnostic)
- **Config:** λ=3.0 req/s (probe-chosen, colocated attainment 100%, P95 1.40 s at CV=1); conditions A (CV1 affine), B (CV3 affine), C (CV3 loadbal), D (CV2 affine), E (CV3 affine, 2P); service×{0.5,1,2} × transfer×{0.7,1,1.4} × 5 seeds; NVLink transfer only (mean 8 ms × scale); hot prefix 6144 tok; colocated tau=1024 +25%, equivalent prefix cache ON (frozen hit-rate guard).
- **Purpose:** H2 kill gate: does cache-affine pairing form a queue hotspot at CV=3?
- **Result:** SUCCESS (runs completed; hypothesis not supported). P99 ratio vs colocated ∈ [0.07, 0.87] across all cells — never ≥ 1.0; B-cell ratio 0.63 (needs ≥1.5×); qfrac ≥ 0.93 at CV=3 at any load (kill conjunct <15% unreachable); C does not reproduce the ≥1.5× tail.
- **Outputs:** `raw/e2_pd_runs.csv`, `raw/e2_colocated_runs.csv`, `raw/e2_summary.csv`, `raw/e2_lowload_diag.csv`, `configs/e2_config.json`, `processed/e2_report.md`
- **Script SHA-256:** e2_sim.py `9e8fca4300fb2a2b14b8ab139f409ba55f877c9296145eea349a17d31c905b8c`; e2_run.py `4f21da3258df8224078e452f3847f2a6d1436562e68659a8b76fe9cf37b2c3c6`
- **Notes:** bugs fixed during bring-up (all re-run after fixes, no data reused from broken runs): (1) event tuples ordered (kind, time) instead of (time, kind) → heapq mis-ordering produced negative TTFTs; (2) deterministic service times caused event-time ties → added per-instance sequence key; (3) colocated baseline lacked the equivalent prefix cache (hit-rate guard) → P/D artificially favored; (4) colocated queue-wait instrumentation initially recorded full TTFT as queue wait — corrected. Design finding: the frozen kill conjunct "P99 queue fraction < 15%" is structurally unreachable at CV≥2 (qfrac ≥ 0.93 at λ∈{0.5,1,3}).

## Blocked gates (hardware)

- E3 (transfer microbenchmark), E4 (vLLM serving 3× A100), E5 (allocation sweep): NOT RUN — testbed absent on this machine (single RTX 4060 8 GB laptop; no 3× A100, no 25 Gbps link, no NVLink pair; no vLLM/PyTorch). See `environment.md`. E1/E2 outcomes decide whether these gates would be needed: H1 survived E1 (narrow band) → E3/E4 still informative for H1; H2 rejected at E2 per the evidence (E4 for H2 only if a v2 plan re-derives the kill condition).