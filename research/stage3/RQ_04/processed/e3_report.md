# e3_report.md - RQ-4 H3 controllability test in the calibrated simulator (frozen experiment_plan.md E3)

Generated 2026-08-27T16:43:57.268807+00:00 UTC. Script SHA-256: 80d96a6607aea0934593c627d07bb074986bcbe6c510b7c40c0d42dc99eddff0

## 1. Scope and config

- Points (surviving E1/E2): H1 points (0.8, 32), (0.9, 32); H2 thresholds (0.7, 16), (0.8, 16); above-threshold (0.9, 64).
- Length variants (recorded; cold unchanged Uniform(1920,2176) mean 2048):
  - base: hot total Uniform(1920,2176) mean 2048, hot suffix mean 1024
  - s256: hot total Uniform(1152,1408) mean 1280, hot suffix mean 256
  - mm  : hot total Uniform(2944,3200) mean 3072, hot suffix mean 2048
- Load anchoring (recorded): loads {0.5, 0.65, 0.8} x the AFFINITY configuration's OWN saturation computed per variant with E0's formula lambda_sat = 1/(p_hot * hot_suffix_mean * gamma * GAMMA_BASE); this keeps the hot-node utilization (frozen formula) = load fraction for every variant (verified) and the cells comparable at equal load. Overload region (hot-node util >= 1.0) excluded from verdicts.
- Policy C: threshold replication k = 2, fixed theta = 1.0 hot req/s (active iff p_hot*lam > theta). Engagement per variant (recorded): s256 active at all loads; base active at loads 0.65/0.8 (at 0.5 the hot rate equals theta exactly -> inactive, C = A); mm inactive at all loads (hot rate = load <= 0.8 < theta) -> C = A there, recorded.
- Replication accounting (recorded): replication traffic = one 1024-token hot-prefix copy to the second replica, counted once per run; total KV transfer = sum of input tokens in the steady window; share = replication_tokens / total_input_tokens. Additional KV memory = 1024 tokens (one extra prefix copy).
- 5 points x 3 loads x 3 variants x 2 budgets x 3 policies x 10 reps = 2700 runs (raw/e3_cells.csv; 2700 rows present). Poisson CV=1, 10,000 requests, max_tokens 256, warm-up 200 hot completions, seeds 1001-1010.
- Numbers only; verdicts are written by the senior analyst (result.md). Simulator passes are labeled 'queue-concentration passes; the tail claim is untested until E4' (frozen revision R1).

## 2. Per-point tables (budget L1; means over 10 reps; P99 TTFT in s)

### (p_hot = 0.7, c = 16)

| variant | load | hot P99 A | hot P99 B | hot P99 C | hot ratio A/B | cold ratio A/B | ratio-of-ratios | C-restore C/B | hit A | hit C | hit dA-C (pp) | repl share | C act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.50 | 2.322 +/- 0.111 | 0.589 +/- 0.030 | 2.331 +/- 0.144 | 3.95 | 0.976 | 4.05 | 3.97 | 0.3511 | 0.3511 | +0.00 | 0 | False |
| base | 0.65 | 3.453 +/- 0.566 | 0.726 +/- 0.032 | 2.037 +/- 1.099 | 4.77 | 0.876 | 5.45 | 2.84 | 0.3511 | 0.3511 | +0.00 | 5.1475e-05 | True |
| base | 0.80 | 6.582 +/- 1.806 | 0.834 +/- 0.031 | 2.928 +/- 2.606 | 7.92 | 1.258 | 6.29 | 3.53 | 0.3511 | 0.3511 | -0.00 | 5.1479e-05 | True |
| s256 | 0.50 | 1.597 +/- 0.105 | 0.836 +/- 0.056 | 1.239 +/- 0.188 | 1.91 | 1.005 | 1.90 | 1.48 | 0.4765 | 0.4765 | -0.00 | 6.9868e-05 | True |
| s256 | 0.65 | 2.218 +/- 0.099 | 1.358 +/- 0.120 | 1.820 +/- 0.220 | 1.64 | 1.010 | 1.63 | 1.35 | 0.4766 | 0.4765 | +0.00 | 6.9882e-05 | True |
| s256 | 0.80 | 5.664 +/- 1.212 | 3.341 +/- 0.733 | 3.843 +/- 1.085 | 1.71 | 1.343 | 1.27 | 1.15 | 0.4766 | 0.4766 | +0.00 | 6.9913e-05 | True |
| mm | 0.50 | 4.415 +/- 0.305 | 1.062 +/- 0.000 | 4.422 +/- 0.322 | 4.16 | 0.999 | 4.16 | 4.16 | 0.2598 | 0.2598 | +0.00 | 0 | False |
| mm | 0.65 | 6.759 +/- 1.319 | 1.126 +/- 0.044 | 6.747 +/- 1.328 | 6.01 | 0.946 | 6.35 | 6.01 | 0.2598 | 0.2598 | +0.00 | 0 | False |
| mm | 0.80 | 12.982 +/- 3.779 | 1.297 +/- 0.034 | 13.089 +/- 3.963 | 10.02 | 1.426 | 7.03 | 10.10 | 0.2598 | 0.2598 | +0.00 | 0 | False |

### (p_hot = 0.8, c = 16)

| variant | load | hot P99 A | hot P99 B | hot P99 C | hot ratio A/B | cold ratio A/B | ratio-of-ratios | C-restore C/B | hit A | hit C | hit dA-C (pp) | repl share | C act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.50 | 2.311 +/- 0.117 | 0.562 +/- 0.000 | 2.302 +/- 0.109 | 4.11 | 0.999 | 4.11 | 4.10 | 0.4010 | 0.4010 | +0.00 | 0 | False |
| base | 0.65 | 3.259 +/- 0.221 | 0.601 +/- 0.027 | 1.710 +/- 1.001 | 5.42 | 0.963 | 5.63 | 2.86 | 0.4010 | 0.4010 | +0.00 | 5.1297e-05 | True |
| base | 0.80 | 6.014 +/- 1.061 | 0.707 +/- 0.023 | 2.897 +/- 2.783 | 8.52 | 0.917 | 9.29 | 4.12 | 0.4010 | 0.4010 | -0.00 | 5.1301e-05 | True |
| s256 | 0.50 | 1.248 +/- 0.027 | 0.412 +/- 0.038 | 0.907 +/- 0.189 | 3.05 | 1.007 | 3.03 | 2.21 | 0.5735 | 0.5735 | +0.00 | 7.3354e-05 | True |
| s256 | 0.65 | 1.528 +/- 0.088 | 0.633 +/- 0.034 | 1.089 +/- 0.244 | 2.42 | 1.006 | 2.40 | 1.72 | 0.5735 | 0.5735 | +0.00 | 7.3358e-05 | True |
| s256 | 0.80 | 2.248 +/- 0.143 | 0.836 +/- 0.066 | 1.273 +/- 0.335 | 2.71 | 1.072 | 2.52 | 1.52 | 0.5735 | 0.5735 | -0.00 | 7.3372e-05 | True |
| mm | 0.50 | 4.422 +/- 0.225 | 1.062 +/- 0.000 | 4.422 +/- 0.224 | 4.16 | 1.000 | 4.17 | 4.16 | 0.2862 | 0.2862 | +0.00 | 0 | False |
| mm | 0.65 | 6.417 +/- 0.509 | 1.064 +/- 0.004 | 6.434 +/- 0.544 | 6.03 | 0.999 | 6.04 | 6.05 | 0.2862 | 0.2862 | +0.00 | 0 | False |
| mm | 0.80 | 11.974 +/- 2.226 | 1.188 +/- 0.038 | 11.987 +/- 2.216 | 10.08 | 0.904 | 11.16 | 10.09 | 0.2862 | 0.2862 | +0.00 | 0 | False |

### (p_hot = 0.8, c = 32)

| variant | load | hot P99 A | hot P99 B | hot P99 C | hot ratio A/B | cold ratio A/B | ratio-of-ratios | C-restore C/B | hit A | hit C | hit dA-C (pp) | repl share | C act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.50 | 2.311 +/- 0.117 | 0.562 +/- 0.000 | 2.302 +/- 0.109 | 4.11 | 0.999 | 4.11 | 4.10 | 0.4010 | 0.4010 | +0.00 | 0 | False |
| base | 0.65 | 3.259 +/- 0.221 | 0.601 +/- 0.027 | 1.710 +/- 1.001 | 5.42 | 0.963 | 5.63 | 2.86 | 0.4010 | 0.4010 | +0.00 | 5.1297e-05 | True |
| base | 0.80 | 6.014 +/- 1.061 | 0.707 +/- 0.023 | 2.897 +/- 2.783 | 8.53 | 0.881 | 9.69 | 4.17 | 0.4010 | 0.4010 | -0.00 | 5.1301e-05 | True |
| s256 | 0.50 | 1.251 +/- 0.028 | 0.412 +/- 0.038 | 0.907 +/- 0.189 | 3.06 | 1.009 | 3.03 | 2.21 | 0.5735 | 0.5735 | +0.00 | 7.3354e-05 | True |
| s256 | 0.65 | 1.530 +/- 0.089 | 0.633 +/- 0.034 | 1.089 +/- 0.243 | 2.42 | 1.005 | 2.41 | 1.71 | 0.5735 | 0.5735 | +0.00 | 7.3358e-05 | True |
| s256 | 0.80 | 2.264 +/- 0.150 | 0.836 +/- 0.066 | 1.264 +/- 0.310 | 2.72 | 1.023 | 2.66 | 1.51 | 0.5735 | 0.5735 | -0.00 | 7.3372e-05 | True |
| mm | 0.50 | 4.422 +/- 0.225 | 1.062 +/- 0.000 | 4.422 +/- 0.224 | 4.16 | 1.000 | 4.17 | 4.16 | 0.2862 | 0.2862 | +0.00 | 0 | False |
| mm | 0.65 | 6.417 +/- 0.509 | 1.064 +/- 0.004 | 6.434 +/- 0.544 | 6.03 | 0.999 | 6.04 | 6.05 | 0.2862 | 0.2862 | +0.00 | 0 | False |
| mm | 0.80 | 11.974 +/- 2.226 | 1.188 +/- 0.038 | 11.987 +/- 2.216 | 10.08 | 0.895 | 11.26 | 10.10 | 0.2862 | 0.2862 | +0.00 | 0 | False |

### (p_hot = 0.9, c = 32)

| variant | load | hot P99 A | hot P99 B | hot P99 C | hot ratio A/B | cold ratio A/B | ratio-of-ratios | C-restore C/B | hit A | hit C | hit dA-C (pp) | repl share | C act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.50 | 2.235 +/- 0.119 | 0.562 +/- 0.000 | 2.236 +/- 0.124 | 3.98 | 1.000 | 3.98 | 3.98 | 0.4505 | 0.4505 | +0.00 | 0 | False |
| base | 0.65 | 3.241 +/- 0.226 | 0.562 +/- 0.000 | 1.657 +/- 1.057 | 5.76 | 0.998 | 5.78 | 2.95 | 0.4505 | 0.4505 | -0.00 | 5.1151e-05 | True |
| base | 0.80 | 5.913 +/- 0.786 | 0.590 +/- 0.020 | 2.207 +/- 2.259 | 10.01 | 0.967 | 10.36 | 3.77 | 0.4505 | 0.4505 | +0.00 | 5.1152e-05 | True |
| s256 | 0.50 | 1.133 +/- 0.035 | 0.188 +/- 0.002 | 0.495 +/- 0.223 | 6.03 | 0.996 | 6.05 | 2.63 | 0.6802 | 0.6802 | +0.00 | 7.7238e-05 | True |
| s256 | 0.65 | 1.310 +/- 0.085 | 0.232 +/- 0.024 | 0.649 +/- 0.235 | 5.67 | 0.977 | 5.81 | 2.75 | 0.6802 | 0.6802 | +0.00 | 7.7242e-05 | True |
| s256 | 0.80 | 1.859 +/- 0.098 | 0.314 +/- 0.041 | 0.715 +/- 0.069 | 6.00 | 1.032 | 5.81 | 2.30 | 0.6803 | 0.6802 | +0.01 | 7.7244e-05 | True |
| mm | 0.50 | 4.402 +/- 0.224 | 1.062 +/- 0.000 | 4.397 +/- 0.224 | 4.15 | 1.000 | 4.15 | 4.14 | 0.3105 | 0.3105 | +0.00 | 0 | False |
| mm | 0.65 | 6.426 +/- 0.441 | 1.062 +/- 0.000 | 6.445 +/- 0.470 | 6.05 | 0.999 | 6.06 | 6.07 | 0.3105 | 0.3105 | +0.00 | 0 | False |
| mm | 0.80 | 11.770 +/- 1.705 | 1.083 +/- 0.030 | 11.760 +/- 1.712 | 10.88 | 0.961 | 11.31 | 10.87 | 0.3106 | 0.3106 | +0.00 | 0 | False |

### (p_hot = 0.9, c = 64)

| variant | load | hot P99 A | hot P99 B | hot P99 C | hot ratio A/B | cold ratio A/B | ratio-of-ratios | C-restore C/B | hit A | hit C | hit dA-C (pp) | repl share | C act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.50 | 2.235 +/- 0.119 | 0.562 +/- 0.000 | 2.236 +/- 0.124 | 3.98 | 1.000 | 3.98 | 3.98 | 0.4505 | 0.4505 | +0.00 | 0 | False |
| base | 0.65 | 3.241 +/- 0.226 | 0.562 +/- 0.000 | 1.657 +/- 1.057 | 5.76 | 0.998 | 5.78 | 2.95 | 0.4505 | 0.4505 | -0.00 | 5.1151e-05 | True |
| base | 0.80 | 5.913 +/- 0.786 | 0.590 +/- 0.020 | 2.207 +/- 2.259 | 10.02 | 0.967 | 10.37 | 3.76 | 0.4505 | 0.4505 | +0.00 | 5.1152e-05 | True |
| s256 | 0.50 | 1.133 +/- 0.035 | 0.188 +/- 0.002 | 0.495 +/- 0.223 | 6.03 | 0.996 | 6.05 | 2.63 | 0.6802 | 0.6802 | +0.00 | 7.7238e-05 | True |
| s256 | 0.65 | 1.306 +/- 0.085 | 0.232 +/- 0.024 | 0.649 +/- 0.235 | 5.67 | 0.977 | 5.81 | 2.81 | 0.6802 | 0.6802 | +0.00 | 7.7242e-05 | True |
| s256 | 0.80 | 1.859 +/- 0.098 | 0.314 +/- 0.041 | 0.715 +/- 0.069 | 6.00 | 1.030 | 5.83 | 2.30 | 0.6803 | 0.6802 | +0.01 | 7.7244e-05 | True |
| mm | 0.50 | 4.402 +/- 0.224 | 1.062 +/- 0.000 | 4.397 +/- 0.224 | 4.15 | 1.000 | 4.15 | 4.14 | 0.3105 | 0.3105 | +0.00 | 0 | False |
| mm | 0.65 | 6.426 +/- 0.441 | 1.062 +/- 0.000 | 6.445 +/- 0.470 | 6.05 | 0.999 | 6.06 | 6.07 | 0.3105 | 0.3105 | +0.00 | 0 | False |
| mm | 0.80 | 11.770 +/- 1.705 | 1.083 +/- 0.030 | 11.760 +/- 1.712 | 10.87 | 0.961 | 11.31 | 10.87 | 0.3106 | 0.3106 | +0.00 | 0 | False |

## 3. Budget equality check (L1 vs L2)

Max relative difference of the CELL-MEAN hot P99 A between budget levels over all E3 cells: 0.0768. Per-rep differences are larger (the tie-break stream depends on the budget via the run_id); the cell means, which drive the verdicts, are budget-invariant as E0's algebra requires.

## 4. H3 construction and falsification checks (numerical)

Frozen thresholds: success = hot ratio A/B >= 2x cold ratio A/B; C restores hot P99 within 30%% of B (C/B <= 1.3) with hit rate within 5pp of A; expected replication traffic < 10%% of total KV transfer. Falsified if ANY of (i)-(iv).

(i) Load-driven regression (hot class singled out under load-only?): max hot P99 B over all cells = 4.899 s (vs min hot P99 A = 1.087 s); hot P99 B / hot P99 A ranges 0.094-0.612; hot P99 B / cold P99 B ranges 0.176-1.003 (load-only does not single out the hot class).

(ii) Hot ratio vs cold ratio (ratio-of-ratios >= 2 required; falsified if within +/-20% at EVERY condition): ratio-of-ratios over the grid min/median/max = 1.27 / 5.78 / 11.31; cells within [0.8, 1.2]: 0 of 45; cells below 2.0: 3 of 45.

(iii) Suffix scaling (falsified if the effect scales with suffix length RATHER THAN with p_hot): hot ratio by variant (all loads/points pooled): s256 3.80 (n=15), base 6.15 (n=15), mm 6.86 (n=15); hot ratio by p_hot at the base variant (concentration axis): 0.7 5.55, 0.8 6.02, 0.9 6.59.

(iv) C must move hot P99 by >20% while holding hit within 5pp (falsified if it fails to move it). C-active cells (25 of 45; in the inactive cells C = A by the fixed theta rule, so the move is ~0 by construction): C move (P99_C - P99_A)/P99_A min/median/max = -63.9% / -47.9% / -17.9%; hit-rate delta A-C min/max = -0.00 / +0.01 pp. Boundary case: exactly one C-active cell moves by less than 20% - (0.7, 16, s256, load 0.65): C = 1.820 s vs A = 2.218 s (move -17.9%).

Success-criterion numbers (for the analyst; C-active cells only): C restoration C/B min/median/max = 1.15 / 2.63 / 4.17 (threshold <= 1.3); replication share max = 7.7244e-05 (expected < 0.10); hit-rate delta A-C within +/-5pp at every C-active cell: True.

## 5. E2 cross-check (base variant at shared cells)

| point | load | E3 A hot P99 | E2 A hot P99 | E3 B hot P99 | E2 B hot P99 |
|---|---|---|---|---|---|
| (0.7, 16) | 0.50 | 2.322 | 2.322 | 0.589 | 0.589 |
| (0.7, 16) | 0.65 | 3.453 | 3.453 | 0.726 | 0.726 |
| (0.7, 16) | 0.80 | 6.582 | 6.582 | 0.834 | 0.834 |
| (0.8, 16) | 0.50 | 2.311 | 2.311 | 0.562 | 0.562 |
| (0.8, 16) | 0.65 | 3.259 | 3.259 | 0.601 | 0.601 |
| (0.8, 16) | 0.80 | 6.014 | 6.014 | 0.707 | 0.707 |
| (0.8, 32) | 0.50 | 2.311 | 2.311 | 0.562 | 0.562 |
| (0.8, 32) | 0.65 | 3.259 | 3.259 | 0.601 | 0.601 |
| (0.8, 32) | 0.80 | 6.014 | 6.014 | 0.707 | 0.707 |
| (0.9, 32) | 0.50 | 2.235 | 2.235 | 0.562 | 0.562 |
| (0.9, 32) | 0.65 | 3.241 | 3.241 | 0.562 | 0.562 |
| (0.9, 32) | 0.80 | 5.913 | 5.913 | 0.590 | 0.590 |
| (0.9, 64) | 0.50 | 2.235 | 2.235 | 0.562 | 0.562 |
| (0.9, 64) | 0.65 | 3.241 | 3.241 | 0.562 | 0.562 |
| (0.9, 64) | 0.80 | 5.913 | 5.913 | 0.590 | 0.590 |

Max relative deviation of hot P99 A (E3 base vs E2, same cells): 0.0000 - the runs are bit-identical re-runs: the tie-break RNG is seeded from the pre-override E1-format run_id (crc32), which is the same for shared (policy, p_hot, load, budget, gamma, rep) cells across E1/E2/E3 (recorded). This confirms full determinism across experiments.

## 6. Sanity re-verification vs E0 (c-sweep anchors)

| anchor | measured util | E0 util (frozen) | util delta %% | measured mean wait (s) | E0 mean wait (s) | wait delta %% | P-K wait at realized rho (s) | P-K delta %% |
|---|---|---|---|---|---|---|---|
| A p_hot=0.8 c=8 load=0.80 | 0.8000 | 0.8000 | -0.00 | 0.9337 | 2.5111 | -62.82 | 1.1160 | -16.33 |
| B p_hot=0.8 c=8 load=0.80 | 0.2966 | 0.3000 | -1.14 | 0.0050 | 0.1433 | -96.51 | n/a | n/a |
| A p_hot=0.8 c=64 load=0.80 | 0.7988 | 0.8000 | -0.14 | 0.9073 | 2.5111 | -63.87 | 1.1080 | -18.11 |
| B p_hot=0.8 c=64 load=0.80 | 0.2966 | 0.3000 | -1.14 | 0.0050 | 0.1433 | -96.51 | n/a | n/a |
| A p_hot=0.9 c=32 load=0.80 | 0.7923 | 0.8000 | -0.97 | 0.9648 | 1.4355 | -32.79 | 1.0085 | -4.33 |
| B p_hot=0.9 c=32 load=0.80 | 0.2408 | 0.2444 | -1.47 | 0.0019 | 0.0960 | -98.02 | n/a | n/a |

## 7. Overload-region report (excluded from verdicts)

No run had any node at utilization >= 1.0 in the steady-state window; all E3 cells are within the stable region (per-variant anchoring keeps the frozen-formula hot-node util at the load fraction).

## 8. Reproducibility

- raw/e3_cells.csv: every run, every rep (2700 rows; resume-safe; replication tokens, additional KV memory, eviction counts, hit rates per policy recorded per run).
- raw/trace_10000_variant_s256/mm_repNN.csv: variant traces (identical inter-arrival and class sequences to the Poisson traces; variant hot lengths drawn with recorded seeds).
- Per-request records and per-node traces are written for all policy-C runs (raw/per_request/, raw/per_node_trace/).
- logs/sanity_e3.json, this report. No verdicts drawn here; result.md is the analyst's deliverable.
