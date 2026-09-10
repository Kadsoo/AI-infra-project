# LOCKED PLAN — RQ-8 (Token × Bit-width × Tier Non-Additivity)

> **Freeze date:** 2026-08-27. Frozen after adversarial review (see `design_review.md`). Stage 3B may NOT modify these items because results look bad. Scientific changes require `LOCKED_PLAN_v2.md` with explicit rationale; never overwrite this file.

## 0. Freeze scope

Frozen and versioned: Primary Hypothesis, Sub-Hypotheses, Main Metrics, Baseline, Success Criterion, Falsification Criterion, the unified statistical calculus, the pre-registered glue semantics, and the kill-gate order (E1 → E2a/E2b → E3 → E4). Implementation-level detail is refinable only with a written record.

## 1. Primary Hypothesis (frozen)

> Under a matched single-7B serving harness with fixed SLO, fixed quality tolerance, and identical kernels/workloads across all cells, stacking the three existing mechanisms — retained-token budget (100%/60%/20%), KV bit-width (16-bit/2-bit), and storage tier (GPU-only / GPU+host) — causes the two-way and three-way factorial interaction terms to be non-zero and to explain a material share of the main-effect variance: support requires a significant interaction term (α=0.05, BH-corrected) with R²_int_rel = SS_interactions/SS_main_effects > 0.50 in at least one of TTFT, TPOT, throughput, HBM usage, or quality across the workload matrix, compared with the baseline of strictly additive main-effect composition.

Revision note: statistics unified pre-freeze (design_review R2) — the old dual-denominator calculus (absolute R²_int for support, relative R²_int_rel for falsification) could make a genuine interaction simultaneously unsupported and unfalsified.

## 2. Sub-Hypotheses (frozen)

- **H1 (retention × bit-width, quality):** the 2-bit quality drop is ≥2× larger at 20% retention than at 60%, and the R×B interaction is significant (α=0.05) with effect ≥50% of the bit-width main effect, in ≥1 workload (per-workload verdicts reported; the ≥1-workload support / both-workloads kill asymmetry is a pre-registered design choice).
- **H2 ((retention, bit-width) × tier):** the host-tier TTFT/TPOT penalty at (20%, 2-bit) exceeds 1.5× the sum of the two single-axis penalties, with byte-normalized transfer latency non-constant across cells.
- **H3 (Pareto reversal):** the axis-best cell (20%, 2-bit, GPU-only) is dominated by ≥1 other cell with ≥10% improvement in ≥1 primary metric and ≤5% regression in every other, outside measurement error, replicated across ≥3 seeds and both workloads.

## 3. Main Metrics (frozen)

TTFT (median+P95), TPOT (median+P95), throughput (tokens/s) + SLO attainment at the ≤3% quality tolerance, measured peak HBM (`cudaMemGetInfo`, never inferred), transfer/fetch volume (bytes + fetch count), quality per workload (2WikiMQA F1; GSM8K 8-shot CoT accuracy). Statistics: Type III SS on cell means; R²_int_rel for both support and falsification; gray zone (significant, R²_int_rel ∈ [0.10, 0.50]) = inconclusive → E4 required; BH across metric×workload families; R coded as categorical contrasts in E2a and E4 identically.

## 4. Baseline (frozen)

(100%, 16-bit, GPU-only) cell as reference on both workloads; additive-composition model (main-effects-only regression, categorical R) as the statistical null; H2's additive bound `pen(20%,2-bit) ≤ 1.15×[pen(20%,16-bit) + pen(100%,2-bit)]` as the H2 null. Fairness: identical kernels, workloads, seeds, concurrency across all cells; actual-bytes accounting (quantized + scales + FP16 residual R=128 + GEAR n_b=20 buffer + sparse/low-rank components).

## 5. Success Criterion (frozen)

| Hypothesis | Success = |
|---|---|
| Primary | ≥1 interaction term significant (α=0.05, BH) AND R²_int_rel > 0.50 in ≥1 metric & ≥1 workload |
| H1 | 2-bit drop at 20% ≥ 2× the drop at 60% AND R×B significant with effect ≥50% of bit-width main effect (≥1 workload) |
| H2 | pen(20%,2-bit) > 1.5× [pen(20%,16-bit) + pen(100%,2-bit)] on TTFT or TPOT, with byte-normalized latency non-constant |
| H3 | axis-best cell dominated (≥10% improvement in ≥1 metric, ≤5% regression in others), ≥3 seeds, both workloads |

## 6. Falsification Criterion (frozen)

| Hypothesis | Falsified iff |
|---|---|
| Primary | every interaction term non-significant (α=0.05, BH) AND R²_int_rel ≤ 0.10 for all metrics and both workloads (full 3×2×2×2 at E4; an E3 null is only endpoint-additive, R ∈ {20%,100%}) |
| H1 | R×B non-significant AND effect <10% of bit-width main effect in BOTH workloads |
| H2 | penalty ≤1.15× the sum of single-axis penalties, OR byte-normalized transfer latency constant across cells |
| H3 | axis-best cell on the Pareto frontier within measurement error (no cell achieves ≥10% / ≤5% dominance) |

Overall null (all three falsified) = axes compose additively in this envelope; RQ-8 reclassified as engineering composition. The E1+E2a cheap-null path declares primary+H1+H2 null only — H3 is not assessed there.

## 7. Kill-gate order (frozen)

E1 (analytical screening, no GPU; conditional H2 kill) → E2a (quality 3×2×2 cells + decoupled-arm contrast, single GPU; H1 kill) → E2b (kernel/transfer microbenchmark; H2 empirical kill) → E3 (2×2×2 tier subset + fake-quant at GPU+host + 9th corner cell on W2 repeat; endpoint verdicts only; rated High with ≤10-person-day port budget and pre-registered fallback) → E4 (full 24-cell matrix + capacity sweep + Pareto; primary/H3 verdicts). Pre-registered glue semantics (frozen): H2O scores on FP16 attention; recent-window protection = 128 = KIVI R; groups over retained tokens in position order with recomputed scales; residual = last 128 retained positions.

## 8. Freeze rules

1. No threshold, baseline, metric, statistical definition, or kill condition in §1–§7 may change during Stage 3B based on experimental outcomes.
2. Any necessary change requires `LOCKED_PLAN_v2.md` with: what changed, why, what evidence motivated it, and whether it loosens or tightens the original claim.
3. Null results are valid and valuable outcomes (they reclassify RQ-8 as engineering composition); they terminate the pipeline per the gates and are not re-run into submission.
4. The claim is a measurement claim about existing mechanisms (no controller, no joint optimizer); a positive is a confidence-upgrade within the tested envelope, not a mechanism discovery.