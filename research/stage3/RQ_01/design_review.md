# Design Review — RQ-1 (P/D vs Colocated Crossover)

> Stage 3A adversarial review. Critic verdict: **require rework before freezing**. All fixes below were applied to `hypothesis.md` and `experiment_plan.md` BEFORE freezing; the frozen state is `LOCKED_PLAN.md`.

## 1. Critic verdict summary

| # | Question | Verdict | Severity |
|---|---|---|---|
| 1 | Does the experiment really test the hypothesis? | FIX-REQUIRED | Critical |
| 2 | Same result, different cause? | FIX-REQUIRED | Critical |
| 3 | Is the baseline fair? | WARN | High |
| 4 | Is the workload cherry-picked? | WARN | Medium |
| 5 | Are success criteria post-hoc? | FIX-REQUIRED | Critical |
| 6 | Hidden variables? | WARN | Medium |
| 7 | Cheaper kill available? | FIX-REQUIRED | High |
| 8 | Re-statement of a paper conclusion? | WARN | High |
| 9 | Enough impact if true? | WARN | Medium |
| A | Threshold internal consistency | FIX-REQUIRED | Critical |
| B | E2 calibration circularity | FIX-REQUIRED | High |
| C | Kill-gate structure | FIX-REQUIRED | High |
| D | Negative results suppressed? | WARN | Medium |

## 2. Key findings (abridged)

1. **H1 kill conditions are physically unreachable at every tier.** "P95 transfer fraction <10%" requires realized bandwidth >130 Gbps on a 25 Gbps link; the "gap <10% of NVLink" branch needs >100 GB/s. The kill gate was fiction — the pipeline always survived to E4.
2. **Primary hypothesis self-contradictory at its target condition.** ≥30% transfer fraction at 25 Gbps (short-class requests each pay ~0.17 s transfer) is incompatible with "P50 TTFT within 5% of colocated". The hypothesis predicted its own falsification precisely when the phenomenon is present.
3. **H2 untestable as stated.** The CV=1 control leg fails at 25 Gbps by transfer alone (not queueing); the load-balanced discriminator fires the kill branch spuriously at 25 Gbps (everyone pays transfer); no NVLink CV=3 cell exists to isolate queueing.
4. **E1's T_prefill calibration ~3–10× too optimistic.** The Splitwise anchor is an 8-GPU (TP8) number scaled as if single-GPU; the plan's own FLOPs formula gives ~0.84 s at 8k, not 0.08–0.3 s. This manufactured the "trivially true" verdict.
5. **H3's joint cell decided by SLO arithmetic before measurement.** P/D P95 TTFT ≈ 3.1 s at the matched rate vs 2.0 s SLO → near-certain "both fail" ambiguity.
6. **IV1(b) 100 Gbps declared but never run** — the crossover location, the one threshold the RQ could establish, is unmeasured.
7. **E3 cannot kill and is redundant with E4** (same testbed, same payloads); the "minimum-cost kill sequence" always reached the expensive tier.
8. **Under-specification**: colocated dispatch policy unstated; "small scheduler hook" is a Mooncake-class conductor; tc throttling may not shape RoCE/IB verbs traffic; node-identity confound (D on node B, baselines on node A).

## 3. Resolution table (issue → resolution applied)

| # | Fix | Where applied | Status |
|---|---|---|---|
| 1 | H1 kill lines re-derived to be reachable: measured P95 transfer fraction **<30%** at 25 Gbps 8k–13k (the claim's own threshold, with corrected calibration) OR 25 Gbps-vs-NVLink P95 TTFT gap **<25%** of NVLink value → H1 dead | hypothesis.md (H1, Falsification), experiment_plan.md (E1/E3/E4) | Applied |
| 2 | Primary hypothesis: **≤5% P50-TTFT conjunct removed**; replaced by "aggregate throughput at equal load within 5%" as a reported guard, not a hypothesis conjunct | hypothesis.md (Primary Hypothesis), experiment_plan.md (E4 success) | Applied |
| 3 | H2 re-scoped to **isolate the queueing mechanism at transfer ≈ 0 (NVLink placement)**: CV=1 cache-affine ≤1.1× colocated; CV=2–3 cache-affine ≥1.5× colocated with P99 queue fraction ≥20%; load-balanced at CV=3 ≤1.2× (discriminator). Kill: cache-affine CV=3 ≤1.1× with queue fraction <15%, OR load-balanced CV=3 ≥1.5× | hypothesis.md (H2), experiment_plan.md (E2/E4 cells) | Applied |
| 4 | E1 calibration corrected to FLOPs arithmetic: T_prefill(8k) ∈ [0.5, 0.9] s, (13k) ∈ [0.8, 1.5] s, (1k) ≈ 0.1 s; sweep re-checked — corpus anchor (f≈0.24) still crosses 30% | experiment_plan.md (E1) | Applied |
| 5 | SLO tuple: TTFT P95 ≤ **4.0 s** (P/D-feasible at the matched rate), TPOT P95 ≤ 0.15 s, attainment ≥95%; double-allocation rerun pre-registered as a first-class cell with verdict rules | hypothesis.md (Control Variables), experiment_plan.md (E4/E5) | Applied |
| 6 | 100 Gbps: one E3 transfer cell + one E4 serving cell (cross-node 100 Gbps, CV=1, cache-affine) added as **exploratory** (records crossover direction; not gated on H1 thresholds, which apply only at 25 Gbps) | experiment_plan.md (E3/E4) | Applied |
| 7 | E3 kept as a hardware gate but with reachable kill lines (fix 1); explicitly documented as E4's step-0 calibration when it survives | experiment_plan.md (E3) | Applied |
| 8 | Colocated dispatch = FCFS + join-shortest-queue (pre-registered); vLLM commit pinned; NIC class + throttle method (netem/tc with netperf verification) stated; node-identity balanced (joint cells run with D on node A and on node B) | experiment_plan.md (E4 controls), hypothesis.md (Control Variables) | Applied |
| 9 | Outcome bands pre-registered at every gate (E4: fraction ≥30% → H1 supported; <30% → H1 falsified; gap ≥25% → supported; <25% → falsified; both-fail → double-allocation rerun before any conclusion) | experiment_plan.md (E4 falsification) | Applied |
| 10 | KV-bytes cross-check arithmetic corrected ((8/66)×(8/96)×2 = 0.020, consistency factor ~3, noted) | experiment_plan.md (E1) | Applied |
| 11 | Q8 (re-statement risk): accepted — positive H1/H3 results are confirmatory of corpus-known directions; the added value is the matched-harness crossover measurement itself. Framed accordingly in LOCKED_PLAN (claims are operating-boundary measurements, not mechanism discoveries) | LOCKED_PLAN.md | Applied |
| 12 | Q9 (impact): accepted with re-scoping — the RQ's deliverable is a measured crossover boundary usable as an evaluation constraint, not a standalone mechanism claim | LOCKED_PLAN.md | Applied |

## 4. Accepted as-is (no change)

- **Transfer fraction instrumented inside the TTFT critical path** (correctly precludes overlap masking of the fraction).
- **Verbatim pre-registered threshold discipline** (the contract was quoted into every experiment).
- **Null-respecting rhetoric** (nulls at any gate stop the pipeline; now that kill lines are reachable, this is real).
- **Guard metrics** (hit-rate, scheduler CPU, netperf bandwidth, HBM utilization).

## 5. Critic bottom line (verbatim)

> "The design shows strong research hygiene — verbatim pre-registered thresholds, guard metrics, ambiguous-outcome handling — but the arithmetic does not support the gates... Fix the calibration, the unreachable kill lines, and the H2 cell matrix before any GPU time is committed."

Resolved by fixes 1–10. Freeze proceeds on the corrected versions.