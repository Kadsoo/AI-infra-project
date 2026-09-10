# RQ-1 E2 Report — Calibrated Queue Simulator: H2 Queue-Hotspot Test (Wave-1, CPU-only)

> Experiment E2 of `LOCKED_PLAN.md` (freeze date 2026-08-27). Scripts: `code/e2_sim.py` (SHA-256 `9e8fca4300fb2a2b14b8ab139f409ba55f877c9296145eea349a17d31c905b8c`), `code/e2_run.py` (`4f21da3258df8224078e452f3847f2a6d1436562e68659a8b76fe9cf37b2c3c6`), `code/e2_diag_lowload.py`. Raw: `raw/e2_pd_runs.csv` (225 runs), `raw/e2_colocated_runs.csv` (45 runs), `raw/e2_summary.csv`, `raw/e2_lowload_diag.csv`. Config: `configs/e2_config.json`. Date: 2026-08-27.

## 1. Setup

- Model: Llama-3.1-8B-class prefill 0.1026 ms/token; hot-prefix length 6144 tokens; 60% of long-class share the hot prefix (trace 5,000 requests, seed 20260827).
- 2 P/D pairs; NVLink-class transfer (lognormal mean 8 ms × scale, sigma 0.5) in ALL verdict cells; 25 Gbps transfer excluded by scope (review fix R3).
- Colocated baseline: 2 independent 1-GPU chunked-prefill queues (tau=1024, +25% chunking overhead), JSQ dispatch, **equivalent prefix cache enabled** per the frozen hit-rate guard (both instances hold the hot prefix → hot requests get the same service reduction as on the P/D hot pair).
- Arrivals: Gamma renewal, CV = 1/sqrt(a) exact (achieved 0.996/2.054/2.996); fixed mean rate across all cells.
- Pre-registered 3-point-probe expansion (6 points, scale 1.0, CV=1, colocated): λ = 3.0 req/s chosen — P95 1.40 s, attainment 100%; all six probe points meet SLO (P95 ≤ 4.0 s, attainment ≥ 95%).
- Sweep per condition: service-scale {0.5,1,2} × transfer-scale {0.7,1,1.4}, 5 seeds; 225 P/D runs + 45 colocated runs. (Plan's "12 runs/condition" was implemented as the full 3×3 grid × 5 seeds; recorded implementation-level deviation.)

## 2. Results (λ = 3.0 req/s; means over 5 seeds; transfer-scale ×1 shown; full table in `raw/e2_summary.csv`)

| Cond | CV | Policy | Svc× | P99 TTFT (s) | P99 qfrac | P99 ratio vs colocated | colocated P99 (s) | max-pair share | SLO attainment |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | affine | 0.5 | 0.65 | 0.76 | 0.79 | 0.82 | 0.74 | 1.000 |
| A | 1 | affine | 1.0 | 1.42 | 0.88 | 0.67 | 2.12 | 0.63 | 1.000 |
| A | 1 | affine | 2.0 | 6.68 | 0.96 | 0.32 | 21.1 | 0.52 | 0.937 |
| **B** | **3** | **affine** | **1.0** | **3.57** | **0.97** | **0.63** | **5.63** | **0.53** | **0.995** |
| B | 3 | affine | 0.5 | 1.24 | 0.94 | 0.69 | 1.80 | 0.54 | 1.000 |
| B | 3 | affine | 2.0 | 16.44 | 0.99 | 0.31 | 52.3* | 0.55 | 0.593 |
| C | 3 | loadbal | 1.0 | 4.89 | 0.98 | 0.87 | 5.63 | 0.57 | 0.976 |
| C | 3 | loadbal | 0.5 | 1.50 | 0.96 | 0.83 | 1.80 | 0.57 | 1.000 |
| C | 3 | loadbal | 2.0 | 38.9 | 1.00 | 0.74 | 52.3* | 0.58 | 0.322 |
| D | 2 | affine | 1.0 | 2.11 | 0.94 | 0.60 | 3.50 | 0.56 | 1.000 |
| E | 3 | affine, 2P | 1.0 | 1.45 | 0.89 | 0.26 | 5.63 | 0.52 | 1.000 |

\* colocated baseline itself diverges at service-scale 2.0 (utilization > 1 at λ=3.0); those cells are in the unstable region and reported, not used in verdicts.

## 3. Frozen-threshold evaluation (H2)

| Frozen line | Threshold | Measured (B, svc 1×) | Pass? |
|---|---|---|---|
| B success: P99 queue fraction (T_queue + transfer_wait)/TTFT ≥ 20% | ≥0.20 | 0.968 | ✓ |
| B success: P99 TTFT ≥ 1.5× colocated | ≥1.5 | **0.633** | ✗ (reversed) |
| A steady-state: P99 ≤ 1.1× colocated | ≤1.1 | 0.673 | ✓ |
| C discriminator: P99 ≤ 1.2× colocated | ≤1.2 | 0.868 | ✓ |
| C discriminator: ≥1.5× tail NOT reproduced | — | not reproduced | ✓ (no burst-load artifact) |
| Kill branch 1: B P99 ≤ 1.1× AND qfrac < 15% | both | 0.633 ✓ but qfrac 0.968 ✗ | partial |
| Kill branch 2: C reproduces ≥1.5× | ≥1.5 | 0.868 | no |
| E sanity: doubling N_P dissolves hotspot | — | ratio drops 0.63→0.26 | consistent with no concentration hotspot |

Across ALL cells (every condition × scale, including scale 0.5 and the low-load diagnostic), the P/D P99 TTFT ratio vs colocated is **0.07–0.87, never ≥ 1.0**: the cache-affine P/D configuration is faster than the colocated baseline at P99 everywhere, with the largest ratio at the load-balanced CV=3 cell (0.87).

## 4. Low-load diagnostic (`raw/e2_lowload_diag.csv`)

At λ ∈ {0.5, 1.0} (far below the matched-SLO rate), conditions A/B/C, 3 seeds:

| λ | Cond | P99 ratio vs colocated | P99 qfrac (P/D) |
|---|---|---|---|
| 0.5 | A (CV=1) | 0.80 | 0.17–0.50 |
| 0.5 | B (CV=3) | 0.71 | 0.93 |
| 0.5 | C (CV=3) | 0.79 | 0.94 |
| 1.0 | B (CV=3) | 0.67 | 0.94 |
| 1.0 | C (CV=3) | 0.82 | 0.95 |

**Design finding:** at CV=3 the P99 queue fraction is ≥93% at ANY load (burst queueing dominates the P99 of TTFT in both systems). The frozen kill conjunct "P99 queue fraction < 15%" (LOCKED_PLAN §6, H2 branch 1) is **structurally unreachable** under the calibrated service model for CV≥2 — an analogue of the pre-freeze unreachable-kill-line problem fixed in RQ-1 E1 (design_review R2) but left unresolved in the E2 kill condition. The ratio leg (P99 ≤ 1.1×) fires at every cell; the conjunct does not.

## 5. Interpretation (numbers only; analyst verdicts in `result.md`)

- The ≥1.5× tail predicted by H2 did not appear at any of the 225 runs; the direction is reversed (P/D faster).
- Concentration DOES occur (max-pair share up to 0.91 at CV=1 low load; 0.53–0.58 at CV=3 matched load), but the concentrated pair's work share is small: hot-prefix requests are only ~18% of arrivals AND their service is ~3× cheaper (cache skip), so the pair's utilization cannot approach 1 in the stable regime. The hotspot is therefore not a pairing-concentration phenomenon at this trace's hot share.
- The load-balanced discriminator (C) is *worse* than cache-affine (B) at CV=3 (0.87 vs 0.63): load-balancing sacrifices the cache benefit (hot misses → full prefill) without gaining enough queueing relief. Affinity at NVLink is not a tail liability in this model; it is a tail asset.

## 6. Sanity checks

- Event-tie bug found and fixed during bring-up (heap order by (time, kind, seq); deterministic service times created ties) — logged in `experiment_log.md`; full 270-run matrix re-run after the fix.
- Colocated baseline fairness fix: equivalent prefix cache enabled per the frozen hit-rate guard (P/D hit-rate would otherwise exceed colocated by >10 points) — logged.
- Monotonicity: higher load → higher P99 for both systems (probe + scale sweep); CV=3 > CV=2 > CV=1 tail at matched load; all consistent.
- 3-point probe expanded to 6 points; λ=3.0 is the SLO-matched rate for the colocated baseline at nominal scale.