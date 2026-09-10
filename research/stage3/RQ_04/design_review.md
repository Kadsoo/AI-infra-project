# Design Review — RQ-4 (Cache Affinity vs Per-Class Tail/Fairness)

> Stage 3A adversarial review. Critic verdict: **require rework before freezing** — the original hypothesis was mathematically unsatisfiable under its own model. All fixes were applied to `hypothesis.md` and `experiment_plan.md` BEFORE freezing; the frozen state is `LOCKED_PLAN.md`.

## 1. Critic verdict summary

| # | Question | Verdict | Severity |
|---|---|---|---|
| 1 | Does the experiment really test the hypothesis? | FIX-REQUIRED | Critical |
| 2 | Same result, different cause? | FIX-REQUIRED | Critical |
| 3 | Baseline fair? | FIX-REQUIRED | Critical |
| 4 | Workload cherry-picked? | FIX-REQUIRED | High |
| 5 | Success criteria post-hoc? | FIX-REQUIRED | High |
| 6 | Hidden variables? | WARN | Medium |
| 7 | Cheaper kill possible? | FIX-REQUIRED | High |
| 8 | Re-statement of a paper conclusion? | PASS (with caveats) | Info |
| 9 | Enough impact if true? | WARN | Medium |
| A | FI captures per-class unfairness? | FIX-REQUIRED | High |
| B | Hot slowdown ratio construction fair? | FIX-REQUIRED | Critical |
| C | H3 falsifiable on 2 nodes? | WARN | Medium |
| D | Simulator-first a detour? | WARN | Medium |
| E | Null suppression / outs | FIX-REQUIRED | High |

## 2. Key findings (abridged — the critic derived these from the plan's own model)

1. **The E1 load point overloads the treatment by construction.** At 0.8 of load-only saturation with p_hot ≈ 0.8 and hot service 256 tokens (L_prefix=1792/2048), the affinity hot node runs at util 1.067 — unstable. Every ratio measured there is an overload artifact via the shared admission FIFO (identical across policies), not an affinity property.
2. **H1's success conjunction is unsatisfiable under the stated model.** Hot ≤1.2× is impossible (M/D/1 gives hot ratio ≥1.33 at ANY load, ≈2.7 at ρ=0.6); cold ≥2× is unreachable in the stable regime (P-K bound ≤1.25) and an admission-queue artifact in overload. In no regime does "cold ≥2× while hot ≤1.2×" hold. The claimed harm actually falls on the HOT class (concentration), not the cold class.
3. **"Least-loaded" metric undefined; outcome flips with it.** Request-count queues route cold away from the hot node (cold improves); token-aware queues route cold onto the hot node (collapse). The plan never pins it.
4. **≥15pp hit-rate anchor contradicted by the plan's own cache model.** Under load-only, the pinned hot prefix becomes resident on all 4 nodes after warm-up (LRU leaf-first cannot evict a shared ancestor) → load-only hit ≈ affinity hit. A ≥15pp gap requires an eviction/churn mechanism the plan never specifies.
5. **Zipf↔p_hot mapping arithmetically false** (s=1.2 → 0.70 over a 2-prefix universe, not 0.8); it overrides the locked 1:1 class-mix control; L_prefix=1792/2048 is a caricature at the extreme edge of the corpus.
6. **Matched-cost control controls the wrong variable** — the steady-state service asymmetry (256 vs 2048) is constitutive of the treatment; the control tests suffix-length sensitivity, not prefill-cost attribution.
7. **H3's ratio construction measures the wrong thing** ("cold ≥2× hot ratio" passes in the stable regime while the cold class does not regress at all).
8. **Gate-0 algebra missing.** The queue component is closed-form (M/G/1); the plan's own claim "no GPU serving needed to establish the SLO crossing" makes the simulator partly redundant for the kill — a cheap algebra step should precede it.
9. **FI misaligned** (both-at-50% ⇒ FI=1.0 "fair"; the load-0.6 stable regime gives FI(A)=1.0 while the hypothesis expects <0.8; unfairness at 0.8 is driven by the HOT class).
10. **γ-robustness tests scale, not shape** (the linear-service assumption is load-bearing for the 8:1 hot/cold ratio; Mooncake's quadratic-attention statement contradicts it).

## 3. Resolution table (issue → resolution applied)

| # | Fix | Where applied | Status |
|---|---|---|---|
| 1 | **E0 gate-0 algebra step added** (critic #7/#8): closed-form M/G/1 concentration analysis computing per-node utilizations, predicted P99 ratios, SLO-crossing points, and the hit-gap derivation for every cell; E1/E2 grid points derived from E0; per-policy load anchoring: λ expressed in the AFFINITY configuration's own saturation, overload region (hot-node util ≥ 1.0) marked and excluded from verdicts | experiment_plan.md (new E0) | Applied |
| 2 | **H1 restructured to the model-supported mechanism** (critic #1/#2/#B): the harmed class is the HOT class (hot-node queue hotspot): hot P99 ≥2× load-only while cold P99 ≤1.2× load-only; the cold-class claim is demoted to a reported E4 observation (within-node scheduling order, SGLang's documented mechanism, only measurable on a real engine) | hypothesis.md (Primary, H1, H3), experiment_plan.md | Applied |
| 3 | **Hit-gain leg re-derived and memory-accounted** (critic #4): equal TOTAL KV budget across policies; expected gap G derived in E0 (≥10pp expected at the calibrated cache level); success threshold = G (pre-registered from E0); if E0 derives G < 10pp, the hit-rate leg is killed pre-registrationally and H1 rests on the queue leg alone | hypothesis.md (H1), experiment_plan.md (E0/E1) | Applied |
| 4 | **Skew re-parameterized** (critic #5): Zipf decoration dropped; skew = hot-prefix request fraction p_hot ∈ {0.5, 0.7, 0.8, 0.9}; L_prefix = 1024 of 2048 (50% shared, within corpus range); class definitions otherwise equal | hypothesis.md (IV-1/IV-5), experiment_plan.md (§1.3) | Applied |
| 5 | **"Least-loaded" defined** = token-weighted backlog (pre-registered); sensitivity cell with request-count metric added | experiment_plan.md (E1–E3) | Applied |
| 6 | **Matched-cost control restated** as suffix-sensitivity test (hot suffix length ∈ {256, 1024}); attribution language fixed ("effect scales with concentration, not suffix length") | experiment_plan.md (E3) | Applied |
| 7 | **H3 restated on the hot class** (critic #7): hot ratio ≥2× cold ratio; replication (2 replicas) restores hot P99 within 30% of load-only with hit rate within 5pp of affinity; load-only-at-same-load control | hypothesis.md (H3) | Applied |
| 8 | **Common absolute SLO** (critic #A): P99 TTFT ≤ 2.0 s for BOTH classes; FI computed on the common SLO, report-only metric | hypothesis.md (Control Variables), experiment_plan.md (§1.1) | Applied |
| 9 | **E1 tests p_hot ∈ {0.8, 0.9}** (critic #E: "most favorable point" premise removed); outcome space extended: hot-only blowup category, 6–14pp hit band → reported | experiment_plan.md (E1, E4) | Applied |
| 10 | **E4 hardware floor re-scoped** (critic #C): 2-node floor validates only H1/H2 anchors; H3 verdict requires ≥3 nodes; "void" verdict category on the floor; E4 step-0 fits the length–service CURVE across [256, 3072] | experiment_plan.md (E4) | Applied |
| 11 | **Simulator claims scoped** (critic #D): E1–E3 pass verdicts are labeled "queue-concentration passes; tail claim untested until E4" (SGLang's starvation is a scheduling-order mechanism the FIFO simulator cannot represent) | experiment_plan.md (§1.2, E4) | Applied |

## 4. Accepted as-is (no change)

- The CPU-first kill chain (E1–E3 no-GPU) — correct for a Medium-confidence tension; E2's full grid is retained but re-anchored per fix 1.
- Trace-replay, warm-up, steady-state window controls.
- Honest non-novelty disclaimer (critic Q8 PASS: the added claim is "first per-class quantification with a declared fairness index under a matched harness").
- Pre-registered γ-robustness band, now supplemented by the E4 length–service-curve fit (fix 10).

## 5. Critic bottom line (verbatim)

> "The design's skeleton is sound (matched harness, three existing policies, CPU-first kill chain, honest non-novelty disclaimer, trace-replay and warm-up controls, corpus citations verified accurate). But the load-bearing numbers are wrong: the E1 load point overloads the treatment by construction, the hot ≤1.2× clause is unsatisfiable under the plan's own FIFO model at every load, the cold ≥2× result is unreachable in the stable regime and an admission-queue artifact in the overload regime... Three of five expected observations are contradicted or underivable from the stated model. Fix the model specification and thresholds (issues 1–10) — especially the gate-0 algebra, the load anchoring, and the per-class clauses — before any run."

Resolved by fixes 1–11: the per-class claim was restructured to the model-supported hot-class hotspot, the model algebra was moved to the front as E0, and every threshold was re-derived or re-anchored. Freeze proceeds on the corrected versions.