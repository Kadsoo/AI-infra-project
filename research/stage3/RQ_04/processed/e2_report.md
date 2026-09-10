# e2_report.md - RQ-4 H2 threshold sweep + H1 grid falsification (frozen experiment_plan.md E2)

Generated 2026-08-27T16:20:50.696965+00:00 UTC. Script SHA-256: f2d831fcb08ee4c476a7b9059e998644e7a6ec0f87263cc1bedc4ed5b8eccdb1

## 1. Scope and config

- Simulator: identical to E1 (per-node FIFO, admission = dispatch, cap c, warm-up 200 hot completions, uniform-random tie-break among exact minima, LRU-radix cache, memory-accounted hit rate).
- Primary grid: p_hot {0.5,0.7,0.8,0.9} x c {8,16,32,64,128} x load {0.5,0.65,0.8} x budgets {L1,L2} x policies {A,B} x 10 reps = 2400 runs (raw/e2_cells.csv; 2600 rows present).
- Burst robustness: Gamma CV {2,4} on p_hot {0.7,0.8,0.9} x c {16,32,64} x loads x budgets x {A,B} x 10 reps = 2160 runs (raw/e2_burst.csv; 2160 rows present).
- Policy C: threshold replication k = 2, fixed theta = 1.0 hot req/s (active iff p_hot*lam > 1.0; recorded), at the surviving threshold points.
- gamma-robustness band {0.5, 2.0} at verdict-relevant points (raw/e2_gamma_robust.csv; 400 rows).
- Identification rule (recorded): crossing cell = mean(hot P99 under A) > 2.0 s AND mean(cold attainment under A) >= 0.99, means over 10 reps; identified at load 0.8, budget L1 (recorded); cross-checked at load 0.65 and budget L2. Overload region (hot-node util >= 1.0) excluded from verdicts; reported separately.
- Common SLO: per-class P99 TTFT <= 2.0 s. FI report-only. Hit leg of H1 was pre-registrationally killed at E0 (G = 0.00 pp); H1 grid falsification criterion (i) (hit gain <= 5pp) does NOT apply - the queue leg decides.

## 2. Threshold identification (load 0.8, L1, gamma 1)

Crossing map (hot P99(A) > 2.0 s AND cold attainment(A) >= 0.99):

| p_hot \ c | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0.5 | . | . | X | X | X |
| 0.7 | . | X | X | X | X |
| 0.8 | . | X | X | X | X |
| 0.9 | . | X | X | X | X |

- p_hot* (at c = 32): 0.7 ; c* (at p_hot = 0.8): 16
- Conjunction check on [p_hot*, 0.9] x [c*, 128]: holds
- Monotonicity: c*(p_hot) non-increasing in p_hot: True (function: {'0.5': 32, '0.7': 16, '0.8': 16, '0.9': 16}); p_hot*(c) non-increasing in c: True (function: {'8': None, '16': 0.7, '32': 0.7, '64': 0.7, '128': 0.7})
- Cross-checks: crossing maps at load 0.65 and at budget L2 are recorded in processed/e2_thresholds.json (same rule; used as robustness cross-checks of the load-0.8/L1 identification).

## 3. Attainment table at the threshold points (means over 10 reps)

| p_hot | c | load | budget | hot P99 A (s) | hot attn A | cold attn A | hot P99 B (s) | hot ratio A/B |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 16 | 0.50 | L1 | 2.322 +/- 0.111 | 0.9780 | 1.0000 | 0.589 +/- 0.030 | 3.95 |
| 0.7 | 16 | 0.50 | L2 | 2.314 +/- 0.139 | 0.9781 | 1.0000 | 0.589 +/- 0.030 | 3.94 |
| 0.7 | 16 | 0.65 | L1 | 3.453 +/- 0.566 | 0.9146 | 1.0000 | 0.726 +/- 0.032 | 4.77 |
| 0.7 | 16 | 0.65 | L2 | 3.487 +/- 0.575 | 0.9137 | 1.0000 | 0.726 +/- 0.032 | 4.82 |
| 0.7 | 16 | 0.80 | L1 | 6.582 +/- 1.806 | 0.7327 | 0.9965 | 0.834 +/- 0.031 | 7.94 |
| 0.7 | 16 | 0.80 | L2 | 6.730 +/- 2.011 | 0.7316 | 0.9963 | 0.834 +/- 0.031 | 8.09 |
| 0.7 | 64 | 0.50 | L1 | 2.322 +/- 0.111 | 0.9780 | 1.0000 | 0.589 +/- 0.030 | 3.95 |
| 0.7 | 64 | 0.50 | L2 | 2.314 +/- 0.139 | 0.9781 | 1.0000 | 0.589 +/- 0.030 | 3.94 |
| 0.7 | 64 | 0.65 | L1 | 3.453 +/- 0.566 | 0.9146 | 1.0000 | 0.726 +/- 0.032 | 4.77 |
| 0.7 | 64 | 0.65 | L2 | 3.487 +/- 0.575 | 0.9137 | 1.0000 | 0.726 +/- 0.032 | 4.82 |
| 0.7 | 64 | 0.80 | L1 | 6.587 +/- 1.801 | 0.7330 | 1.0000 | 0.834 +/- 0.031 | 7.94 |
| 0.7 | 64 | 0.80 | L2 | 6.666 +/- 1.814 | 0.7313 | 1.0000 | 0.834 +/- 0.031 | 8.04 |
| 0.8 | 32 | 0.50 | L1 | 2.311 +/- 0.117 | 0.9798 | 1.0000 | 0.562 +/- 0.000 | 4.11 |
| 0.8 | 32 | 0.50 | L2 | 2.308 +/- 0.120 | 0.9800 | 1.0000 | 0.562 +/- 0.000 | 4.11 |
| 0.8 | 32 | 0.65 | L1 | 3.259 +/- 0.221 | 0.9208 | 1.0000 | 0.601 +/- 0.027 | 5.42 |
| 0.8 | 32 | 0.65 | L2 | 3.256 +/- 0.210 | 0.9214 | 1.0000 | 0.601 +/- 0.027 | 5.42 |
| 0.8 | 32 | 0.80 | L1 | 6.014 +/- 1.061 | 0.7430 | 1.0000 | 0.707 +/- 0.023 | 8.49 |
| 0.8 | 32 | 0.80 | L2 | 5.994 +/- 1.060 | 0.7409 | 1.0000 | 0.707 +/- 0.023 | 8.47 |
| 0.9 | 16 | 0.50 | L1 | 2.235 +/- 0.119 | 0.9823 | 1.0000 | 0.562 +/- 0.000 | 3.98 |
| 0.9 | 16 | 0.50 | L2 | 2.243 +/- 0.116 | 0.9822 | 1.0000 | 0.562 +/- 0.000 | 3.99 |
| 0.9 | 16 | 0.65 | L1 | 3.241 +/- 0.226 | 0.9249 | 1.0000 | 0.562 +/- 0.000 | 5.76 |
| 0.9 | 16 | 0.65 | L2 | 3.249 +/- 0.226 | 0.9250 | 1.0000 | 0.562 +/- 0.000 | 5.78 |
| 0.9 | 16 | 0.80 | L1 | 5.913 +/- 0.786 | 0.7529 | 0.9988 | 0.590 +/- 0.020 | 10.01 |
| 0.9 | 16 | 0.80 | L2 | 5.889 +/- 0.796 | 0.7536 | 0.9988 | 0.590 +/- 0.020 | 9.97 |
| 0.9 | 64 | 0.50 | L1 | 2.235 +/- 0.119 | 0.9823 | 1.0000 | 0.562 +/- 0.000 | 3.98 |
| 0.9 | 64 | 0.50 | L2 | 2.243 +/- 0.116 | 0.9822 | 1.0000 | 0.562 +/- 0.000 | 3.99 |
| 0.9 | 64 | 0.65 | L1 | 3.241 +/- 0.226 | 0.9249 | 1.0000 | 0.562 +/- 0.000 | 5.76 |
| 0.9 | 64 | 0.65 | L2 | 3.249 +/- 0.226 | 0.9250 | 1.0000 | 0.562 +/- 0.000 | 5.78 |
| 0.9 | 64 | 0.80 | L1 | 5.913 +/- 0.786 | 0.7529 | 1.0000 | 0.590 +/- 0.020 | 10.02 |
| 0.9 | 64 | 0.80 | L2 | 5.889 +/- 0.796 | 0.7536 | 1.0000 | 0.590 +/- 0.020 | 9.97 |

## 4. H2 falsification-range check (hot P99 under A within SLO at >=95%% attainment anywhere in p_hot [0.5,0.9] x c [8,128])

- Strict SLO reading (per-class P99 TTFT <= 2.0 s): hot P99 under A is > 2.0 s at EVERY primary cell (minimum 2.066 s over all loads/budgets) - the hot class never stays within the SLO anywhere on the tested range.
- Loose reading (hot attainment >= 0.95): such cells exist only at load 0.5 (hot attainment 0.97-0.98, all p_hot and c); the criterion requires >= 0.95 at EVERY cell, and at loads 0.65/0.8 hot attainment is <= 0.93 - the criterion is NOT met on the full range.
- Cells with hot attainment >= 0.95 (load 0.5, budget L1):

| p_hot | c | hot P99 A (s) | hot attn A |
|---|---|---|---|
| 0.5 | 8 | 2.481 | 0.9672 |
| 0.5 | 16 | 2.481 | 0.9672 |
| 0.5 | 32 | 2.481 | 0.9672 |
| 0.5 | 64 | 2.481 | 0.9672 |
| 0.5 | 128 | 2.481 | 0.9672 |
| 0.7 | 8 | 2.318 | 0.9780 |
| 0.7 | 16 | 2.322 | 0.9780 |
| 0.7 | 32 | 2.322 | 0.9780 |
| 0.7 | 64 | 2.322 | 0.9780 |
| 0.7 | 128 | 2.322 | 0.9780 |
| 0.8 | 8 | 2.311 | 0.9798 |
| 0.8 | 16 | 2.311 | 0.9798 |
| 0.8 | 32 | 2.311 | 0.9798 |
| 0.8 | 64 | 2.311 | 0.9798 |
| 0.8 | 128 | 2.311 | 0.9798 |
| 0.9 | 8 | 2.235 | 0.9823 |
| 0.9 | 16 | 2.235 | 0.9823 |
| 0.9 | 32 | 2.235 | 0.9823 |
| 0.9 | 64 | 2.235 | 0.9823 |
| 0.9 | 128 | 2.235 | 0.9823 |

- Where the crossing fails at load 0.8 (L1), it fails on the COLD side (cold attainment < 0.99): the c = 8 column (all p_hot) and (0.5, 16), (0.7, 16). At c = 8 the cluster admission cap binds under A (mean gate waits 500-700 vs 0 under B), degrading BOTH classes - an admission-cap regime, not a hot-node-hotspot regime (hot P99 is still > 2.0 s there).

## 5. H1 grid-falsification check (queue leg; hit criterion inapplicable)

Falsified iff hot P99(A) within 20%% of hot P99(B) at EVERY grid cell (ratio in [0.8, 1.2]). Hot P99 ratio A/B (means over 10 reps, min/max across loads {0.5,0.65,0.8} and budgets {L1,L2}):

| p_hot \ c | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0.5 | 2.8-5.6 | 2.8-5.6 | 2.8-5.7 | 2.8-5.7 | 2.8-5.7 |
| 0.7 | 3.9-8.1 | 3.9-8.1 | 3.9-8.0 | 3.9-8.0 | 3.9-8.0 |
| 0.8 | 4.1-8.6 | 4.1-8.5 | 4.1-8.5 | 4.1-8.5 | 4.1-8.5 |
| 0.9 | 4.0-10.0 | 4.0-10.0 | 4.0-10.0 | 4.0-10.0 | 4.0-10.0 |

Cells with ratio in [0.8, 1.2] (within-20% cells): 0 (none expected; E1 measured 4-10x).

## 6. gamma-robustness band (verdict-relevant points, gamma {0.5, 2.0})

| p_hot | c | gamma | load | hot P99 A (s) | hot P99 B (s) | hot ratio | cold ratio | hot attn A | cold attn A |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 16 | 0.5 | 0.80 | 3.322 | 0.417 | 7.98 | 1.255 | 0.9454 | 0.9986 |
| 0.7 | 16 | 0.5 | 0.80 | 3.309 | 0.417 | 7.95 | 1.269 | 0.9457 | 0.9986 |
| 0.7 | 16 | 2.0 | 0.80 | 13.171 | 1.668 | 7.92 | 1.257 | 0.3846 | 0.4891 |
| 0.7 | 16 | 2.0 | 0.80 | 13.157 | 1.668 | 7.93 | 1.258 | 0.3859 | 0.4890 |
| 0.7 | 64 | 0.5 | 0.80 | 3.320 | 0.417 | 8.01 | 0.954 | 0.9454 | 1.0000 |
| 0.7 | 64 | 0.5 | 0.80 | 3.306 | 0.417 | 7.99 | 0.964 | 0.9460 | 1.0000 |
| 0.7 | 64 | 2.0 | 0.80 | 13.163 | 1.668 | 7.90 | 0.955 | 0.3857 | 0.4916 |
| 0.7 | 64 | 2.0 | 0.80 | 13.157 | 1.668 | 7.93 | 0.962 | 0.3857 | 0.4915 |
| 0.8 | 32 | 0.5 | 0.80 | 3.006 | 0.353 | 8.51 | 0.881 | 0.9512 | 1.0000 |
| 0.8 | 32 | 0.5 | 0.80 | 2.999 | 0.353 | 8.50 | 0.881 | 0.9512 | 1.0000 |
| 0.8 | 32 | 2.0 | 0.80 | 12.043 | 1.413 | 8.53 | 0.881 | 0.4084 | 0.5024 |
| 0.8 | 32 | 2.0 | 0.80 | 12.025 | 1.413 | 8.50 | 0.881 | 0.4090 | 0.5023 |
| 0.9 | 16 | 0.5 | 0.80 | 2.942 | 0.295 | 9.97 | 0.967 | 0.9525 | 0.9999 |
| 0.9 | 16 | 0.5 | 0.80 | 2.945 | 0.295 | 9.98 | 0.967 | 0.9524 | 0.9999 |
| 0.9 | 16 | 2.0 | 0.80 | 11.791 | 1.180 | 9.98 | 0.967 | 0.4272 | 0.5081 |
| 0.9 | 16 | 2.0 | 0.80 | 11.761 | 1.180 | 9.97 | 0.967 | 0.4277 | 0.5081 |
| 0.9 | 64 | 0.5 | 0.80 | 2.942 | 0.295 | 9.97 | 0.967 | 0.9525 | 1.0000 |
| 0.9 | 64 | 0.5 | 0.80 | 2.945 | 0.295 | 9.97 | 0.967 | 0.9524 | 1.0000 |
| 0.9 | 64 | 2.0 | 0.80 | 11.791 | 1.180 | 10.00 | 0.967 | 0.4272 | 0.5094 |
| 0.9 | 64 | 2.0 | 0.80 | 11.761 | 1.180 | 9.96 | 0.967 | 0.4277 | 0.5094 |

## 7. Burst robustness (Gamma CV {2,4}, matched mean)

| p_hot | c | CV | load | hot P99 A (s) | hot P99 B (s) | hot ratio | cold ratio | hot attn A | overload A |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 16 | 2 | 0.65 | 8.759 | 1.739 | 5.06 | 1.164 | 0.5658 | 0 |
| 0.7 | 16 | 2 | 0.80 | 14.856 | 1.999 | 7.51 | 3.411 | 0.3600 | 0 |
| 0.7 | 16 | 4 | 0.65 | 34.717 | 5.930 | 5.91 | 4.365 | 0.1950 | 0 |
| 0.7 | 16 | 4 | 0.80 | 52.964 | 7.301 | 7.35 | 5.901 | 0.1086 | 0 |
| 0.7 | 32 | 2 | 0.65 | 8.862 | 1.739 | 5.13 | 0.850 | 0.5662 | 0 |
| 0.7 | 32 | 2 | 0.80 | 14.863 | 1.999 | 7.52 | 1.126 | 0.3618 | 0 |
| 0.7 | 32 | 4 | 0.65 | 34.655 | 5.930 | 5.92 | 3.156 | 0.1959 | 0 |
| 0.7 | 32 | 4 | 0.80 | 53.015 | 7.301 | 7.35 | 4.872 | 0.1082 | 0 |
| 0.7 | 64 | 2 | 0.65 | 8.862 | 1.739 | 5.10 | 0.846 | 0.5664 | 0 |
| 0.7 | 64 | 2 | 0.80 | 14.863 | 1.999 | 7.52 | 0.784 | 0.3604 | 0 |
| 0.7 | 64 | 4 | 0.65 | 34.652 | 5.930 | 5.87 | 1.102 | 0.1951 | 0 |
| 0.7 | 64 | 4 | 0.80 | 53.095 | 7.301 | 7.36 | 2.804 | 0.1079 | 0 |
| 0.8 | 16 | 2 | 0.65 | 9.234 | 1.464 | 6.33 | 1.418 | 0.5501 | 0 |
| 0.8 | 16 | 2 | 0.80 | 15.959 | 1.648 | 9.72 | 4.425 | 0.3481 | 0 |
| 0.8 | 16 | 4 | 0.65 | 40.070 | 4.708 | 8.54 | 6.278 | 0.1829 | 0 |
| 0.8 | 16 | 4 | 0.80 | 60.515 | 5.468 | 11.11 | 8.782 | 0.1002 | 0 |
| 0.8 | 32 | 2 | 0.65 | 9.210 | 1.464 | 6.31 | 0.814 | 0.5496 | 0 |
| 0.8 | 32 | 2 | 0.80 | 16.011 | 1.648 | 9.75 | 1.279 | 0.3486 | 0 |
| 0.8 | 32 | 4 | 0.65 | 40.158 | 4.708 | 8.54 | 4.756 | 0.1816 | 0 |
| 0.8 | 32 | 4 | 0.80 | 60.610 | 5.468 | 11.10 | 7.468 | 0.0997 | 0 |
| 0.8 | 64 | 2 | 0.65 | 9.210 | 1.464 | 6.30 | 0.814 | 0.5496 | 0 |
| 0.8 | 64 | 2 | 0.80 | 15.976 | 1.648 | 9.74 | 0.792 | 0.3492 | 0 |
| 0.8 | 64 | 4 | 0.65 | 40.070 | 4.708 | 8.54 | 1.952 | 0.1814 | 0 |
| 0.8 | 64 | 4 | 0.80 | 60.515 | 5.468 | 11.08 | 4.773 | 0.1001 | 0 |
| 0.9 | 16 | 2 | 0.65 | 10.297 | 1.273 | 8.11 | 2.053 | 0.5312 | 0 |
| 0.9 | 16 | 2 | 0.80 | 17.524 | 1.390 | 12.64 | 5.427 | 0.3354 | 0 |
| 0.9 | 16 | 4 | 0.65 | 44.422 | 3.888 | 11.51 | 8.485 | 0.1699 | 0 |
| 0.9 | 16 | 4 | 0.80 | 67.124 | 4.373 | 15.48 | 12.242 | 0.0935 | 0 |
| 0.9 | 32 | 2 | 0.65 | 10.284 | 1.273 | 8.10 | 0.601 | 0.5319 | 0 |
| 0.9 | 32 | 2 | 0.80 | 17.628 | 1.390 | 12.71 | 1.515 | 0.3371 | 0 |
| 0.9 | 32 | 4 | 0.65 | 44.418 | 3.888 | 11.45 | 6.617 | 0.1693 | 0 |
| 0.9 | 32 | 4 | 0.80 | 67.124 | 4.373 | 15.42 | 10.581 | 0.0935 | 0 |
| 0.9 | 64 | 2 | 0.65 | 10.284 | 1.273 | 8.11 | 0.601 | 0.5319 | 0 |
| 0.9 | 64 | 2 | 0.80 | 17.628 | 1.390 | 12.71 | 0.574 | 0.3366 | 0 |
| 0.9 | 64 | 4 | 0.65 | 44.418 | 3.888 | 11.47 | 3.045 | 0.1695 | 0 |
| 0.9 | 64 | 4 | 0.80 | 67.124 | 4.373 | 15.48 | 7.263 | 0.0937 | 0 |

## 8. Policy C (threshold replication, k = 2, theta = 1.0 hot req/s) at the surviving threshold points

| p_hot | c | load | budget | hot P99 C (s) | hot P99 A (s) | hot P99 B (s) | C hot attn | replication active |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 16 | 0.65 | L1 | 2.037 +/- 1.099 | 3.453 | 0.726 | 0.9651 | True |
| 0.7 | 16 | 0.65 | L2 | 2.059 +/- 1.139 | 3.487 | 0.726 | 0.9640 | True |
| 0.7 | 16 | 0.80 | L1 | 2.928 +/- 2.606 | 6.582 | 0.834 | 0.9143 | True |
| 0.7 | 16 | 0.80 | L2 | 2.893 +/- 2.534 | 6.730 | 0.834 | 0.9140 | True |
| 0.7 | 64 | 0.65 | L1 | 2.037 +/- 1.099 | 3.453 | 0.726 | 0.9651 | True |
| 0.7 | 64 | 0.65 | L2 | 2.059 +/- 1.139 | 3.487 | 0.726 | 0.9640 | True |
| 0.7 | 64 | 0.80 | L1 | 2.928 +/- 2.606 | 6.587 | 0.834 | 0.9146 | True |
| 0.7 | 64 | 0.80 | L2 | 2.893 +/- 2.534 | 6.666 | 0.834 | 0.9134 | True |
| 0.8 | 32 | 0.65 | L1 | 1.710 +/- 1.001 | 3.259 | 0.601 | 0.9765 | True |
| 0.8 | 32 | 0.65 | L2 | 1.713 +/- 1.014 | 3.256 | 0.601 | 0.9769 | True |
| 0.8 | 32 | 0.80 | L1 | 2.897 +/- 2.783 | 6.014 | 0.707 | 0.9181 | True |
| 0.8 | 32 | 0.80 | L2 | 2.878 +/- 2.764 | 5.994 | 0.707 | 0.9175 | True |
| 0.9 | 16 | 0.65 | L1 | 1.657 +/- 1.057 | 3.241 | 0.562 | 0.9775 | True |
| 0.9 | 16 | 0.65 | L2 | 1.668 +/- 1.071 | 3.249 | 0.562 | 0.9776 | True |
| 0.9 | 16 | 0.80 | L1 | 2.207 +/- 2.259 | 5.913 | 0.590 | 0.9484 | True |
| 0.9 | 16 | 0.80 | L2 | 2.205 +/- 2.260 | 5.889 | 0.590 | 0.9483 | True |
| 0.9 | 64 | 0.65 | L1 | 1.657 +/- 1.057 | 3.241 | 0.562 | 0.9775 | True |
| 0.9 | 64 | 0.65 | L2 | 1.668 +/- 1.071 | 3.249 | 0.562 | 0.9776 | True |
| 0.9 | 64 | 0.80 | L1 | 2.207 +/- 2.259 | 5.913 | 0.590 | 0.9484 | True |
| 0.9 | 64 | 0.80 | L2 | 2.205 +/- 2.260 | 5.889 | 0.590 | 0.9483 | True |

## 9. Overload-region report (excluded from verdicts)

No run had any node at utilization >= 1.0 in the steady-state window; all primary and burst cells are within the stable region.

## 10. Sanity re-verification vs E0 (c-sweep anchors)

| anchor | measured util | E0 util (frozen) | util delta %% | measured mean wait (s) | E0 mean wait (s) | wait delta %% | P-K wait at realized rho (s) | P-K delta %% |
|---|---|---|---|---|---|---|---|
| A p_hot=0.8 c=8 load=0.80 | 0.8000 | 0.8000 | -0.00 | 0.9337 | 2.5111 | -62.82 | 1.1160 | -16.33 |
| B p_hot=0.8 c=8 load=0.80 | 0.2966 | 0.3000 | -1.14 | 0.0050 | 0.1433 | -96.51 | n/a | n/a |
| A p_hot=0.8 c=64 load=0.80 | 0.7988 | 0.8000 | -0.14 | 0.9073 | 2.5111 | -63.87 | 1.1080 | -18.11 |
| B p_hot=0.8 c=64 load=0.80 | 0.2966 | 0.3000 | -1.14 | 0.0050 | 0.1433 | -96.51 | n/a | n/a |
| A p_hot=0.9 c=32 load=0.80 | 0.7923 | 0.8000 | -0.97 | 0.9648 | 1.4355 | -32.79 | 1.0085 | -4.33 |
| B p_hot=0.9 c=32 load=0.80 | 0.2408 | 0.2444 | -1.47 | 0.0019 | 0.0960 | -98.02 | n/a | n/a |

E0's closed form is c-independent; the c-sweep anchors verify the cap does not distort utilization (busy-time fraction) vs the frozen formula. Mean-wait deltas follow the documented E1 findings: E0's per-node M/G/1 overshoots JSQ-routed nodes (policy B) and the cold-split assumption (policy A); the P-K wait at the realized rho validates the M/G/1 mechanism where arrivals are Poisson (hot node).

## 11. Reproducibility

- raw/e2_cells.csv (primary + C runs), raw/e2_burst.csv (Gamma CV 2/4), raw/e2_gamma_robust.csv (gamma band) - every run, every rep; resume-safe (existing run_ids are skipped, rows never dropped).
- Per-request records (raw/per_request/) and per-node traces (raw/per_node_trace/) are written for the sanity anchors and the policy-C threshold-point runs (instrumentation per plan E2).
- Seeds 1001-1010, warm-up 200 hot completions, theta = 1.0, identification rule and load recorded in processed/e2_thresholds.json.
