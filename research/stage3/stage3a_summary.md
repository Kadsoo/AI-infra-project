# Stage 3A Summary — Hypothesis Formalization & Minimum Experiment Design

> Date: 2026-08-27. Scope: Tier 1 Candidate RQs from `knowledge/stage3_candidates.md` (RQ-1, RQ-2, RQ-4, RQ-8) formalized into falsifiable hypotheses with minimum-cost kill-early experiment designs, adversarially reviewed, fixed, and frozen. **No experiment was executed; no solution was designed.** Stage 3B begins implementation.

## 1. What was processed

| RQ | Title | Directories |
|---|---|---|
| RQ-1 | When does P/D disaggregation lose to colocated execution? | `stage3/RQ_01/` |
| RQ-2 | How fragile are fixed retention/precision/chunk budgets? | `stage3/RQ_02/` |
| RQ-4 | Can cache affinity improve reuse without violating per-class tail/fairness? | `stage3/RQ_04/` |
| RQ-8 | Do retained-token count, bit-width, and tier interact non-additively? | `stage3/RQ_08/` |

Each directory contains: `hypothesis.md` (Primary + Sub-Hypotheses, IV/DV/controls, confounders, expected observations, falsification conditions, ambiguous outcomes), `experiment_plan.md` (kill-gated minimum-cost experiments), `design_review.md` (adversarial verdicts + resolution table), `LOCKED_PLAN.md` (frozen contract).

## 2. Hypotheses formed

- **15 sub-hypotheses** across 4 RQs (RQ-1: H1 transfer materiality / H2 queue hotspot / H3 baseline overtake; RQ-2: H1 quality fragility / H2 TPOT-kernel overhead / H3 oracle gap; RQ-4: H1 hot-node hotspot / H2 SLO boundary / H3 attributability-controllability; RQ-8: H1 retention×bit-width / H2 (R,B)×tier / H3 Pareto reversal, plus the factorial primary).
- Every primary hypothesis is a single negatable sentence with quantitative anchors; every sub-hypothesis has an explicit, reachable kill condition; every experiment names which hypothesis it may kill.

## 3. Adversarial review outcome (all four required rework before freezing)

| RQ | Most severe finding | Fix class |
|---|---|---|
| RQ-1 | Kill lines physically unreachable (>130 Gbps needed on a 25 Gbps link); primary hypothesis self-contradictory; H2 untestable (transfer confound); prefill calibration 3–10× off | Thresholds re-derived; H2 isolated at NVLink; calibration corrected; SLO re-scoped |
| RQ-2 | "Non-native" cells were actually inside the papers' own suites (definitional gaming); phase dimension gated behind expected positives; no statistical power on 108–231-sample splits | Uniform native rule; out-of-suite cells; unconditional phase/open-value cells; bootstrap CIs + noise band |
| RQ-4 | **Hypothesis unsatisfiable under its own model** (cold ≥2× while hot ≤1.2× impossible; load point overloads the treatment by construction); fake Zipf mapping; FI misaligned | Restructure: harmed class = hot class; E0 gate-0 algebra; per-policy load anchoring; p_hot skew; common SLO |
| RQ-8 | Measured "interaction" unattributable (eviction×quantization glue semantics unspecified); support/falsification used different denominators (both could hold at once); overall-null shortcut skipped H3 | Glue semantics pre-registered + decoupled arm; unified R²_int_rel calculus with gray zone; H3 scoped out of the cheap-null path |

Each fix is recorded with its critic citation in the RQ's `design_review.md` §3 resolution table.

## 4. Most-priority experiments (see `stage3a_priority.md`)

1. **RQ-2** (impact/cost best) — E1 triage (desk) then E2 wave 1 (one A100).
2. **RQ-1** — E1/E2 (CPU-only) immediately; E3/E4 hardware only if they survive.
3. **RQ-4** — E0+E1 (hours, CPU) can kill the RQ before any GPU.
4. **RQ-8** — E1+E2a+E2b can answer much of the question at microbenchmark cost.

Wave 1 (no GPU, parallel): RQ-1 E1+E2, RQ-2 E1, RQ-8 E1, RQ-4 E0+E1. Kill gates stop the pipeline.

## 5. RQs that exposed severe design problems

All four did — this is normal for this stage and is why adversarial review precedes freezing:
- **RQ-4 was the most broken:** the pre-review hypothesis predicted a conjunction that the queueing model proves impossible. It required a claim restructure (hot-class hotspot), not just threshold edits.
- **RQ-1** had an unfalsifiable kill gate and a self-contradictory primary (would have been judged false exactly when the phenomenon was present).
- **RQ-8** had an unattributable interaction (glue vs mechanism) and a statistical calculus that could make the primary simultaneously unsupported and unfalsified.
- **RQ-2** had asymmetric native definitions and a noise floor above its own tolerance.

## 6. RQs NOT recommended to continue

**None dropped.** All four Tier 1 RQs are retainable after the pre-freeze rework. Two caveats recorded in the reviews:
- RQ-8's **positive** outcome is a confidence-upgrade within a narrow envelope (the null is the novel result); its E3/E4 tier experiment carries the largest engineering risk (KIVI-in-vLLM port, pre-registered ≤10-person-day budget + fallback).
- RQ-4's **positive** claims require the E4 real-system run (the simulator cannot confirm the tail claim — within-node scheduling order is unmodeled); if E0's algebra predicts trivial outcomes, the RQ's residual value is the per-class quantification itself.

## 7. RQs recommended for Stage 3B

**All four, in priority order RQ-2 → RQ-1 → RQ-4 → RQ-8** (justified in `stage3a_priority.md`), executed via the Wave 1/2/3 schedule. Each RQ's `LOCKED_PLAN.md` is the contract for Stage 3B: thresholds, baselines, metrics, success and falsification criteria, and kill-gate order are frozen; any change requires `LOCKED_PLAN_v2.md` with an explicit rationale.

## 8. Deliverables index

```
research/stage3/
  RQ_01/  hypothesis.md  experiment_plan.md  design_review.md  LOCKED_PLAN.md
  RQ_02/  hypothesis.md  experiment_plan.md  design_review.md  LOCKED_PLAN.md
  RQ_04/  hypothesis.md  experiment_plan.md  design_review.md  LOCKED_PLAN.md
  RQ_08/  hypothesis.md  experiment_plan.md  design_review.md  LOCKED_PLAN.md
  stage3a_priority.md
  stage3a_summary.md
```

## 9. Stopping point

Stage 3A is complete. Per the task instruction: **stop here — no experiment implementation was started.** Stage 3B begins with Wave 1 of `stage3a_priority.md`.