# LOCKED PLAN — RQ-2 (Static Budget Fragility)

> **Freeze date:** 2026-08-27. Frozen after adversarial review (see `design_review.md`). Stage 3B may NOT modify these items because results look bad. Scientific changes require `LOCKED_PLAN_v2.md` with explicit rationale; never overwrite this file.

## 0. Freeze scope

Frozen and versioned: Primary Hypothesis, Sub-Hypotheses, Main Metrics, Baseline, Success Criterion, Falsification Criterion, and the experiment gate order (E1 triage → E2 quality matrix → E3 TPOT/kernel → E4 oracle → E5 open-value/completion). Implementation-level detail (exact splits, knob values, seeds) is refinable only with a written record.

## 1. Primary Hypothesis (frozen)

> Under a fixed memory target equal to each method's published operating point (H2O 20% KV, StreamingLLM 4 sinks, SnapKV capacity 1024, PyramidKV α=8 with KV=64, KIVI 2-bit G=32 R=128, GEAR 2-bit s=2% r=4), the static configuration's embedded task- and phase-specific importance assumption causes the primary quality metric to cross a predeclared tolerance (≥10% relative degradation vs the full-KV baseline on the same task; PPL-type ≥15% rise) on at least one **out-of-suite** condition (∞Bench En.Sum long-form summarization, chain-of-thought reasoning, or the long-decode generation phase beyond the prefill-selected state), while a same-method same-budget per-condition oracle exceeds the fixed setting by >10% in quality or >15% in TPOT, compared with baseline B = full-KV cache on the identical task.

Revision note: "non-native" was narrowed to genuinely out-of-suite conditions (design_review R1) — in-suite tasks (SnapKV summarization, KIVI Falcon) are observation cells and cannot witness H1.

## 2. Sub-Hypotheses (frozen)

- **H1 (quality fragility):** a method at its own published budget crosses the tolerance on ≥1 out-of-suite condition while its native (in-suite) delta stays <10%. Falsified if no out-of-suite crossing exists across the completed matrix, OR native and non-native drop ≥10% alike (universal fragility — attribution fails; the "static setting is globally unsafe" finding is still recorded).
- **H2 (TPOT/kernel overhead):** at matched batch ≥8 and matched measured HBM, the memory-maximizing static setting shows TPOT P50 ≥15% above full-KV, or per-step kernel overhead (dequant/selection/recompression) >10% of decode-step time, or a ≥2× KV-representation HBM saving (KV bytes, weights excluded — revision R7) with <5% throughput gain, on ≥1 condition.
- **H3 (oracle gap):** a same-method, same-implementation, same-measured-budget re-tuning of the method's OWN knobs achieves ≥10% quality or ≥15% TPOT improvement on ≥1 condition. Falsified if oracle differs from fixed by <5% on both quality and latency on every condition, or gains require a different method family or a larger budget.

## 3. Main Metrics (frozen)

Per-task quality via official metrics (LongBench F1/ROUGE-L/EM, GSM8k/BBH EM, PPL, LongGenBench acc) as paired deltas vs full-KV with bootstrap CIs (±2-point non-discriminating band, pre-registered); TPOT P50/P95 (≥1000 decode steps); kernel-overhead share; measured peak HBM (bytes, KV-representation accounting for H2); TTFT.

## 4. Baseline (frozen)

Full-KV cache on the identical task (same model, split, template, gen length, seed, harness). Fairness: paired delta per task only; never cross-task absolute scores.

## 5. Success Criterion (frozen)

| Hypothesis | Success = |
|---|---|
| H1 | ≥1 out-of-suite condition (task class or generation phase) drops ≥10% (PPL ≥15% rise) with native <10%, confirmed by ≥2 datasets/conditions with CIs outside the ±2-point band |
| H2 | ≥1 condition shows ≥15% TPOT P50 degradation, OR ≥10% kernel share, OR ≥2× KV-bytes saving with <5% throughput gain (batch ≥8) |
| H3 | ≥1 condition: oracle beats fixed setting by ≥10% quality or ≥15% TPOT at equal measured HBM |

## 6. Falsification Criterion (frozen)

| Hypothesis | Falsified iff |
|---|---|
| H1 | no out-of-suite crossing with native-safe delta across the completed matrix (E2+E5), OR native and non-native drop ≥10% alike (universal fragility) |
| H2 | at matched batch ≥8 and matched HBM: all conditions within 15% TPOT, kernel share <10%, no ≥2×-saving-without-throughput config; batch-1-only differences excluded as noise |
| H3 | oracle vs fixed <5% on both quality and latency on every condition, or gains only via method-family change or larger budget |

## 7. Kill-gate order (frozen)

E1 (offline triage, no GPU; same-model rows only, screening-only) → E2 (single-GPU quality matrix; phase cell UNCONDITIONAL; kills H1) → E3 (TPOT/kernel microbenchmark; kills H2) → E4 (per-condition oracle at equal measured HBM; kills H3) → E5 (open-value cells unconditional — StreamingLLM evicted-distance, H2O long-form/CoT, GEAR per-task, PyramidKV per-task; completion parts conditional). Single-dataset crossings do not count; floor/ceiling exclusions (<5 / >95 baseline) apply; any post-hoc threshold adjustment is cheating.

## 8. Freeze rules

1. No threshold, baseline, metric, or kill condition in §1–§7 may change during Stage 3B based on experimental outcomes.
2. Any necessary change requires `LOCKED_PLAN_v2.md` with: what changed, why, what evidence motivated it, and whether it loosens or tightens the original claim.
3. Null results are valid outcomes: a negative E2 triggers E5's completion parts; universal fragility is recorded as a finding with H1 falsified, not re-run into submission.
4. Claims are scoped as matched-harness operating-boundary measurements; the oracle (H3) re-tunes the method's own published knobs only, at equal measured budget.