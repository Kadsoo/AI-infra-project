# LOCKED PLAN — RQ-4 (Cache Affinity vs Per-Class Tail/Fairness)

> **Freeze date:** 2026-08-27. Frozen after adversarial review (see `design_review.md`). The original hypothesis was mathematically unsatisfiable under its own model; the restructure below was applied pre-freeze. Stage 3B may NOT modify these items because results look bad. Scientific changes require `LOCKED_PLAN_v2.md` with explicit rationale; never overwrite this file.

## 0. Freeze scope

Frozen and versioned: Primary Hypothesis, Sub-Hypotheses, Main Metrics, Baseline, Success Criterion, Falsification Criterion, the skew parameterization, the SLO/FI definitions, the load-anchoring rule, and the kill-gate order (E0 → E1 → E2 → E3 → E4). Implementation-level detail is refinable only with a written record.

## 1. Primary Hypothesis (frozen)

> Under prefix-popularity skew (hot-prefix request fraction p_hot ≥ 0.7) with hot and cold classes of equal mean total prompt length (2048 tokens; 50% shared prefix) served at equal total offered load within the stable region (hot-node utilization < 1.0), longest-prefix greedy (affinity-only) routing achieves a ≥G memory-accounted aggregate cache hit-rate gain (G derived in E0; expected ≥10pp) and ≥20% lower aggregate mean TTFT than load-only routing, but the concentration of hot-class traffic on the hot node causes the hot class's P99 TTFT to exceed ≥2× its load-only value (hot-node queue hotspot), while the cold class's P99 TTFT stays within 1.2× of load-only.

Revision note: the harmed class is the HOT class (design_review R1) — the original "cold ≥2× while hot ≤1.2×" conjunction was unsatisfiable under the queueing model (hot ratio ≥1.33 at any load; cold ratio ≤1.25 in the stable regime).

## 2. Sub-Hypotheses (frozen)

- **H1 (hot-node queue hotspot):** at p_hot = 0.8 (and 0.9), c = 32, stable load, equal memory-accounted budget: hit-rate gain ≥ G (expected ≥10pp), aggregate mean TTFT ≤ 0.8× load-only, hot P99 ≥ 2× load-only, cold P99 ≤ 1.2× load-only.
- **H2 (SLO boundary):** exists p_hot* ∈ [0.7, 0.9] and c* ∈ [16, 64] such that for all p_hot ≥ p_hot*, c ≥ c* (other dimension fixed at threshold): hot P99 under affinity exceeds the common SLO (2.0 s) while cold attainment ≥99%; thresholds non-increasing in the fixed dimension.
- **H3 (attributability + controllability):** hot ratio P99(affinity)/P99(load-only) ≥ 2× the cold-class ratio; threshold replication (2 replicas) restores hot P99 within 30% of load-only with hit rate within 5pp of affinity-only.

## 3. Main Metrics (frozen)

Per-class and aggregate: cache hit rate (memory-accounted); TTFT mean/P50/P95/P99; queue delay (mean/P99); goodput under the common SLO; slowdown_k(p) ratios; FI (report-only, common SLO); replication traffic/memory (policy C).

## 4. Baseline (frozen)

Load-only routing (minimum token-weighted backlog) at equal total offered load, equal memory-accounted total KV budget, equal request set (trace-replay). Fairness: only the routing decision differs (same scheduler, batching, model, admission, eviction code).

## 5. Success Criterion (frozen)

| Hypothesis | Success = |
|---|---|
| H1 | hit-rate gain ≥ G AND aggregate mean TTFT ≤ 0.8× AND hot P99 ≥ 2× load-only AND cold P99 ≤ 1.2× load-only (stable region, both test points) |
| H2 | SLO crossing at p_hot* ≤ 0.8 / c* ≤ 32 with cold attainment ≥99%; monotone thresholds |
| H3 | hot ratio ≥ 2× cold ratio; replication restores hot P99 within 30% of load-only; hit rate within 5pp of affinity-only |

## 6. Falsification Criterion (frozen)

| Hypothesis | Falsified iff |
|---|---|
| H1 | hot P99 within 20% of load-only at every tested point in the stable region, OR memory-accounted hit-rate gain ≤ 5pp at all skew levels (hit leg pre-registrationally dead if E0 derives G < 10pp) |
| H2 | hot P99 within the common SLO at ≥95% attainment across p_hot ∈ [0.5, 0.9], c ∈ [8, 128], OR thresholds fail monotonicity |
| H3 | regression appears identically under load-only; OR hot ratio equals cold ratio within ±20%; OR effect scales with suffix length (not p_hot); OR replication fails to move hot P99 by >20% with hit rate within 5pp |

Outcome bands (frozen): overload region (hot-node util ≥ 1.0) excluded from all verdicts and reported separately; hit gain 6–14pp → reported, queue leg still decides H1; hot-only blowup is the H1 prediction (not ambiguous); cold-class degradation on the real system is a separate E4 observation (within-node scheduling order), never merged into H1–H3.

## 7. Kill-gate order (frozen)

E0 (gate-0 algebra: per-cell utilizations, predicted ratios, SLO crossings, hit-gain G; no GPU) → E1 (queueing replay, p_hot ∈ {0.8, 0.9}; kills H1) → E2 (event simulator grid; kills H1/H2) → E3 (simulator H3 + suffix-sensitivity control; kills H3) → E4 (real system: H1/H2 anchors on 2-node floor; H3 verdict requires ≥3 nodes — "void" on the floor). Load anchoring: loads are fractions of the affinity configuration's own saturation; simulator passes are labeled "queue-concentration passes; tail claim untested until E4".

## 8. Freeze rules

1. No threshold, baseline, metric, parameterization, or kill condition in §1–§7 may change during Stage 3B based on experimental outcomes.
2. Any necessary change requires `LOCKED_PLAN_v2.md` with: what changed, why, what evidence motivated it, and whether it loosens or tightens the original claim.
3. Null results are valid outcomes and terminate the pipeline per the kill gates; they are recorded, not re-run into submission.
4. The claim is a per-class quantification of a known trade-off under a matched harness (first per-class quantification with a declared fairness index per the corpus); no scheduler or controller is proposed.