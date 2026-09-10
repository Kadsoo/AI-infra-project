# e0_report.md - RQ-4 gate-0 algebra (frozen LOCKED_PLAN.md, plan section E0)

Derivation run at 2026-08-27T14:38:46.604978+00:00 UTC on Windows (CPU only). Script SHA-256: 9b9b7783c10ee01f0930fd4c13c15ee978e50b45e3dee32831b50a6b7f3c4225

## 1. Model (frozen)

- N = 4 homogeneous nodes, each one FIFO prefill server; admission = dispatch, no global FIFO (plan 1.2).
- Service T_prefill = gamma * tokens_not_cached, gamma_base = 1/2048 s/token (2048-token prefill ~ 1.0 s).
- Hot class: shared 1024-token prefix + unique suffix, total Uniform(1920,2176) (suffix Uniform(896,1152), mean 1024). Cold class: unique prefixes, same total length distribution, always miss.
- Policies: A affinity (longest-prefix, tiebreak min token-weighted backlog); B load-only (min token-weighted backlog); B' request-count least-loaded (E1 cell).
- Load anchoring (frozen): loads are fractions of the AFFINITY configuration's own saturation; overload region (hot-node util >= 1.0, frozen definition) excluded from verdicts.
- Hot-node utilization (frozen formula (i)) = (p_hot * suffix_hot work) / (per-node total work) * load-only utilization; equivalently rho_hot = p_hot * lam * 1024 * gamma = load fraction by construction.

## 2. Affinity saturation (per-policy load anchoring base)

lambda_sat(p_hot, gamma) = 1 / (p_hot * 1024 * gamma)  [req/s]  (hot-node util = 1.0).

| p_hot | gamma=0.5 | gamma=1.0 | gamma=2.0 |
|---|---|---|---|
| 0.5 | 8.00000 | 4.00000 | 2.00000 |
| 0.7 | 5.71429 | 2.85714 | 1.42857 |
| 0.8 | 5.00000 | 2.50000 | 1.25000 |
| 0.9 | 4.44444 | 2.22222 | 1.11111 |

E1 load levels {0.5, 0.65, 0.8} x lambda_sat. In the closed form the resulting hot-node util (frozen def) = load exactly; utilizations and P99 ratios are gamma-scale-invariant under the linear service model (E1 re-checks at gamma in {0.5, 1, 2}).

## 3. Stable-region map (gamma = 1; frozen definition; utilizations are gamma-invariant)

rho_hot (frozen def) = load for every cell. rho_hot_full additionally includes the cold-class work the hot node serves under mean-field min-backlog routing (cold share 1/4 per node). Cells where rho_hot_full >= 1.0 are marked; the frozen overload definition uses rho_hot_frozen (>= 1.0 never occurs at these loads).

| p_hot | load | rho_hot (frozen) | rho_hot_full | rho_B per-node | stable(frozen) | stable(full) |
|---|---|---|---|---|---|---|
| 0.5 | 0.50 | 0.500 | 0.750 | 0.375 | True | True |
| 0.5 | 0.65 | 0.650 | 0.975 | 0.488 | True | True |
| 0.5 | 0.80 | 0.800 | 1.200 | 0.600 | True | False |
| 0.7 | 0.50 | 0.500 | 0.607 | 0.232 | True | True |
| 0.7 | 0.65 | 0.650 | 0.789 | 0.302 | True | True |
| 0.7 | 0.80 | 0.800 | 0.971 | 0.371 | True | True |
| 0.8 | 0.50 | 0.500 | 0.562 | 0.188 | True | True |
| 0.8 | 0.65 | 0.650 | 0.731 | 0.244 | True | True |
| 0.8 | 0.80 | 0.800 | 0.900 | 0.300 | True | True |
| 0.9 | 0.50 | 0.500 | 0.528 | 0.153 | True | True |
| 0.9 | 0.65 | 0.650 | 0.686 | 0.199 | True | True |
| 0.9 | 0.80 | 0.800 | 0.844 | 0.244 | True | True |

## 4. Predicted ratios at the pre-registered E1 points (c = 32)

Expected values per E1 cell (c = 32; budget L1/L2 do not enter the algebra). cold_*_primary = cold traffic served only on cold nodes; cold_*_mix = 25% of cold arrivals land on the hot node (mean-field split). E1 measures the realized split.

| p_hot | load | gamma | hot P99 TTFT ratio | cold P99 ratio (prim) | cold P99 ratio (mix) | agg mean ratio (prim) | agg mean ratio (mix) | hot P99 TTFT A (s) | cold P99 TTFT A (s) | hot P99 TTFT B (s) | hot util (full) | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.50 | 0.5 | 1.86 | 0.946 | 1.285 | 1.320 | 1.344 | 1.43 | 0.97 | 0.77 | 0.562 | True |
| 0.8 | 0.50 | 1.0 | 1.86 | 0.946 | 1.285 | 1.320 | 1.344 | 2.86 | 1.93 | 1.54 | 0.562 | True |
| 0.8 | 0.50 | 2.0 | 1.86 | 0.946 | 1.285 | 1.320 | 1.344 | 5.73 | 3.87 | 3.09 | 0.562 | True |
| 0.8 | 0.65 | 0.5 | 2.75 | 0.890 | 1.763 | 1.718 | 1.769 | 2.37 | 0.99 | 0.86 | 0.731 | True |
| 0.8 | 0.65 | 1.0 | 2.75 | 0.890 | 1.763 | 1.718 | 1.769 | 4.74 | 1.98 | 1.72 | 0.731 | True |
| 0.8 | 0.65 | 2.0 | 2.75 | 0.890 | 1.763 | 1.718 | 1.769 | 9.48 | 3.96 | 3.45 | 0.731 | True |
| 0.8 | 0.80 | 0.5 | 6.67 | 0.831 | 3.951 | 3.525 | 3.690 | 6.42 | 1.01 | 0.96 | 0.900 | True |
| 0.8 | 0.80 | 1.0 | 6.67 | 0.831 | 3.951 | 3.525 | 3.690 | 12.84 | 2.02 | 1.92 | 0.900 | True |
| 0.8 | 0.80 | 2.0 | 6.67 | 0.831 | 3.951 | 3.525 | 3.690 | 25.67 | 4.03 | 3.85 | 0.900 | True |
| 0.9 | 0.50 | 0.5 | 1.81 | 0.908 | 1.257 | 1.354 | 1.366 | 1.25 | 0.86 | 0.69 | 0.528 | True |
| 0.9 | 0.50 | 1.0 | 1.81 | 0.908 | 1.257 | 1.354 | 1.366 | 2.51 | 1.71 | 1.39 | 0.528 | True |
| 0.9 | 0.50 | 2.0 | 1.81 | 0.908 | 1.257 | 1.354 | 1.366 | 5.02 | 3.43 | 2.77 | 0.528 | True |
| 0.9 | 0.65 | 0.5 | 2.58 | 0.907 | 1.660 | 1.719 | 1.742 | 1.92 | 0.90 | 0.74 | 0.686 | True |
| 0.9 | 0.65 | 1.0 | 2.59 | 0.907 | 1.660 | 1.719 | 1.742 | 3.84 | 1.80 | 1.49 | 0.686 | True |
| 0.9 | 0.65 | 2.0 | 2.58 | 0.907 | 1.660 | 1.719 | 1.742 | 7.68 | 3.60 | 2.97 | 0.686 | True |
| 0.9 | 0.80 | 0.5 | 4.93 | 0.891 | 2.914 | 2.855 | 2.910 | 3.91 | 0.93 | 0.79 | 0.844 | True |
| 0.9 | 0.80 | 1.0 | 4.93 | 0.891 | 2.914 | 2.855 | 2.910 | 7.82 | 1.86 | 1.58 | 0.844 | True |
| 0.9 | 0.80 | 2.0 | 4.93 | 0.891 | 2.914 | 2.855 | 2.910 | 15.63 | 3.72 | 3.17 | 0.844 | True |

## 5. SLO crossing flags (common SLO: per-class P99 TTFT <= 2.0 s)

Cells whose hot node is overloaded (rho_hot_full >= 1.0) are marked 'overload' - M/G/1 predictions are undefined there and the cell is excluded from verdicts (frozen rule); reported separately.

| p_hot | load | gamma | hot P99 A > 2.0s | cold P99 A <= 2.0s (prim) | cold P99 A <= 2.0s (mix) | SLO crossed (prim) | SLO crossed (mix) |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.50 | 0.5 | True | True | False | True | False |
| 0.5 | 0.50 | 1.0 | True | False | False | False | False |
| 0.5 | 0.50 | 2.0 | True | False | False | False | False |
| 0.5 | 0.65 | 0.5 | True | True | False | True | False |
| 0.5 | 0.65 | 1.0 | True | False | False | False | False |
| 0.5 | 0.65 | 2.0 | True | False | False | False | False |
| 0.5 | 0.80 | 0.5 | overload | overload | overload | overload | overload |
| 0.5 | 0.80 | 1.0 | overload | overload | overload | overload | overload |
| 0.5 | 0.80 | 2.0 | overload | overload | overload | overload | overload |
| 0.7 | 0.50 | 0.5 | False | True | True | False | False |
| 0.7 | 0.50 | 1.0 | True | False | False | False | False |
| 0.7 | 0.50 | 2.0 | True | False | False | False | False |
| 0.7 | 0.65 | 0.5 | True | True | False | True | False |
| 0.7 | 0.65 | 1.0 | True | False | False | False | False |
| 0.7 | 0.65 | 2.0 | True | False | False | False | False |
| 0.7 | 0.80 | 0.5 | True | True | False | True | False |
| 0.7 | 0.80 | 1.0 | True | False | False | False | False |
| 0.7 | 0.80 | 2.0 | True | False | False | False | False |
| 0.8 | 0.50 | 0.5 | False | True | True | False | False |
| 0.8 | 0.50 | 1.0 | True | True | False | True | False |
| 0.8 | 0.50 | 2.0 | True | False | False | False | False |
| 0.8 | 0.65 | 0.5 | True | True | True | True | True |
| 0.8 | 0.65 | 1.0 | True | True | False | True | False |
| 0.8 | 0.65 | 2.0 | True | False | False | False | False |
| 0.8 | 0.80 | 0.5 | True | True | False | True | False |
| 0.8 | 0.80 | 1.0 | True | False | False | False | False |
| 0.8 | 0.80 | 2.0 | True | False | False | False | False |
| 0.9 | 0.50 | 0.5 | False | True | True | False | False |
| 0.9 | 0.50 | 1.0 | True | True | False | True | False |
| 0.9 | 0.50 | 2.0 | True | False | False | False | False |
| 0.9 | 0.65 | 0.5 | False | True | True | False | False |
| 0.9 | 0.65 | 1.0 | True | True | False | True | False |
| 0.9 | 0.65 | 2.0 | True | False | False | False | False |
| 0.9 | 0.80 | 0.5 | True | True | False | True | False |
| 0.9 | 0.80 | 1.0 | True | True | False | True | False |
| 0.9 | 0.80 | 2.0 | True | False | False | False | False |

## 6. Budget levels and memory-accounted hit-gain G

| level | total KV tokens | per-node | G (pp) | derivation |
|---|---|---|---|---|
| L1 | 32768 | 8192 | 0.00 | 4x2048-token-class concurrency: per node 4 concurrent 2048-token classes = 4*2048 = 8192 tokens; x4 nodes = 32768. Within Mooncake LRU envelope 30%@1K -> 50%@50K -> 51%@Inf (8K/node mid-curve) and SGLang production hit range. |
| L2 | 16384 | 4096 | 0.00 | 50% smaller than L1 (hypothesis.md IV-6). |

### G derivation

G = 0.00 pp at both budget levels, for all p_hot. Under the frozen cache model both policies cache ALL reusable content in steady state: A keeps the hot prefix on the hot node, B accumulates it on every node that serves hot traffic (all 4 nodes in steady state under min-backlog). Steady-state token hit rate = p_hot/2 under both policies. LRU leaf-first never evicts the shared prefix (suffix leaves are evicted first and regenerate), and both budget levels (L1=32768, L2=16384 total KV tokens) exceed the 4096 tokens B needs for its 4 prefix copies, so equal budgets do not create a differential. B's extra copies consume memory that buys no hits (no other reusable content exists). Hence G < 10pp => the pre-registered E0 kill of the hit leg fires (LOCKED_PLAN 6 / plan E0).

**Pre-registered E0 outcome:** G = 0.00 pp < 10 pp at both budget levels => the hit-rate leg of H1 is killed pre-registrationally (LOCKED_PLAN 6, plan E0 success criterion); H1 rests on the queue leg alone (hot P99 ratio >= 2, cold P99 ratio <= 1.2, aggregate mean TTFT <= 0.8x). The E1 falsification criterion (i) (hit gain <= 5pp kills H1) does NOT apply.

## 7. Pre-registered E1 / E2 grid points

- E1 (this wave): p_hot in {0.8, 0.9}, c = 32, loads {0.5, 0.65, 0.8} x affinity saturation, budgets {L1, L2}, gamma {0.5, 1, 2}, 10 reps, policies A, B (+ B' sensitivity at p_hot = 0.8). Expected values: see table 4.
- E2 (next wave, pre-registered): full grid p_hot in {0.5, 0.7, 0.8, 0.9} x c in {8, 16, 32, 64} x loads {0.5, 0.65, 0.8} x gamma {0.5, 1, 2}, both budget levels; expected values = the per-cell predictions in raw/e0_grid.csv (hot P99 TTFT ratio, cold P99 TTFT ratio, aggregate mean ratio, SLO flags).

## 8. Inversion validation

The M/G/1 P99 waiting times use numerical Laplace inversion (Abate-Whitt Euler) of the exact Pollaczek-Khinchin wait transform with Uniform service. Worst relative error vs the exact M/M/1 closed-form quantiles: 3.577e-09 (threshold 1%).

## 9. Verification of the frozen sanity rule

E0 predictions are checked against the E1 simulator at 3 anchor points (utilization and mean wait within a few %), per plan E0 baseline / E1 sanity; see processed/e1_report.md for the measured-vs-predicted comparison.
