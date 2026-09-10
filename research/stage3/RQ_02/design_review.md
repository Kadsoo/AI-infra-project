# Design Review — RQ-2 (Static Budget Fragility)

> Stage 3A adversarial review. Critic verdict: **require rework before freezing**. All fixes below were applied to `hypothesis.md` and `experiment_plan.md` BEFORE freezing; the frozen state is `LOCKED_PLAN.md`.

## 1. Critic verdict summary

| # | Question | Verdict | Severity |
|---|---|---|---|
| 1 | Does the experiment really test the hypothesis? | FIX-REQUIRED | phase dimension gated behind negativity |
| 2 | Same result, different cause? | WARN | paired-delta sound; native/non-native delta leaks model/truncation confounds |
| 3 | Baseline fair? | FIX-REQUIRED | "native" definitions convenient, not uniform |
| 4 | Workload cherry-picked? | FIX-REQUIRED | E2 primary cells = corpus's known-crossing cells; open cells gated |
| 5 | Success criteria post-hoc? | WARN | pre-registered, but values look fitted; boundary inside noise floor |
| 6 | Hidden variables? | WARN | truncation mismatch, seed check on wrong cell, FA2×kernel interaction |
| 7 | Cheaper kill possible? | WARN | deeper desk reading could resolve cells E1 refuses to touch |
| 8 | Restatement of papers? | WARN | flagship cells re-confirm paper-stated failures |
| 9 | Statistical sanity? | FIX-REQUIRED | 108–231 samples, 10% = 2–5 examples, no example-level variance |
| A | Native definitions vs papers | FIX-REQUIRED | both flagships' "non-native" cells are inside the papers' own suites |
| B | H3 oracle implementable? | WARN | implementable; weights-dominated HBM makes equal-budget near-vacuous |
| C | Budget equalization for all methods? | WARN | H2's "≥2× saving" leg structurally unreachable at E3 geometry |
| D | Null suppressed? | FIX-REQUIRED | universal fragility = H1-kill in one place, "non-discriminating" in another |
| E | 10% measurable? | FIX-REQUIRED | no error bars; boundary inside example-level noise |

## 2. Key findings (abridged)

1. **Native/non-native asymmetry.** SnapKV's summarization tasks (GovReport/QMSum/MultiNews) are inside its own published LongBench-16 table; Falcon-7B is inside KIVI's own published suite. Under the hypothesis's own rule ("native = the task suite the paper actually validated"), both flagship "non-native" cells are actually in-suite, and a drop there attacks "published-point validity" — which the hypothesis itself excludes from H1 support.
2. **Phase dimension gated behind negativity.** The generation-phase cell runs only if wave 1 is negative; pre-registered predictions say wave 1 is positive → the RQ title's "task type OR generation phase" claim is never tested.
3. **Evidence starvation.** Every novel/open cell (StreamingLLM retrieval-at-evicted-distance — never tested in-paper; H2O long-form/CoT; GEAR per-task; PyramidKV per-task native) is downstream of predicted-surviving confirmatory cells.
4. **Anchor mismatch.** SnapKV predictions use LWMChat-column numbers for Mistral-7B cells (actual Mistral anchor: NrtvQA −4.8%, not −0.9%); KIVI Falcon predictions come from max_seq=4096-truncated paper runs while E2 planned untruncated 13K splits on a 2048-window model.
5. **Noise floor.** 10% relative drop = 2–5 examples on official splits; greedy+seed measures zero variance; classification boundaries sit inside example-level noise.
6. **Contradiction on the most likely outcome.** Universal fragility (native also ≥10%) is listed as an H1 kill branch in one place and "non-discriminating" in the ambiguous-outcome block.

## 3. Resolution table (issue → resolution applied)

| # | Fix | Where applied | Status |
|---|---|---|---|
| 1 | **Uniform native rule** (critic #1/#3/A): native = exactly the paper's reported task/model/layout set, including summarization for SnapKV and Falcon-7B for KIVI. Non-native H1 cells become genuinely out-of-suite: SnapKV → ∞Bench En.Sum + GSM8K 8-shot CoT; KIVI → ∞Bench En.Sum + long-decode phase cell (output ≥1024). In-suite crossings (GovReport, Falcon) are demoted to "published-point validity" observation cells, never H1 evidence | hypothesis.md (H1 native definitions), experiment_plan.md (E1 triage rows, E2 cells) | Applied |
| 2 | **Phase cell unconditional** in wave 1 (A3 = SnapKV long-decode, 2 runs) | experiment_plan.md (E2) | Applied |
| 3 | **E5 open-value cells (a–d) run unconditionally** (PyramidKV per-task native, StreamingLLM evicted-distance, GEAR per-task LongBench, H2O long-form/CoT); only the budget-scan (e) stays conditional on the ambiguous-outcome rule | experiment_plan.md (E5) | Applied |
| 4 | **Same-model rows only** in E1 triage (SnapKV Mistral anchor = NrtvQA −4.8%; GovReport-on-Mistral flagged "measure"); **truncation protocol** pre-registered: KIVI cells match the paper's max_seq 4096 + one untruncated sensitivity cell | experiment_plan.md (E1, E2) | Applied |
| 5 | **Statistics**: per-cell bootstrap CIs (≥1000 resamples) + pre-registered non-discriminating band (±2 metric points); determinism check moved to the custom-kernel cell (KIVI Falcon) | hypothesis.md (Confounders), experiment_plan.md (E2 instrumentation) | Applied |
| 6 | **Universal-fragility mapping unified**: native also ≥10% on every tested condition → H1 falsified (attribution fails); stated identically in hypothesis, E2, E5 | hypothesis.md (Falsification, Ambiguous Outcome), experiment_plan.md (E2/E5) | Applied |
| 7 | **E5-b needle placement** beyond the window: positions 3000/5000/10000 (window ~2048) | experiment_plan.md (E5) | Applied |
| 8 | H2's "≥2× memory saving with <5% throughput gain" leg: restated as measured-at-achievable-batch (H2 tested at batch 8 per control; the 2× leg applies to the measured HBM delta of the KV representation, not peak-HBM including weights) | hypothesis.md (H2 note) | Applied |
| 9 | GEAR's in-corpus position clarified in expected observations (GEAR native reasoning is near-lossless −0.8%; its H1-relevant cells are per-task LongBench, now in E5-c) | hypothesis.md (Expected Observation) | Applied |

## 4. Accepted as-is (no change)

- Paired-delta discipline (never cross-task absolute scores).
- Pre-registration mechanics ("any post-hoc adjustment is cheating").
- HBM-by-bytes accounting, floor/ceiling exclusions with the new noise band (fix 5).
- E1 triage structure as candidate pruning (its role is now explicitly screening-only and cheap).
- H3 oracle exclusion rules (same measured budget ±1%, method's own knobs only).

## 5. Critic bottom line (verbatim)

> "The harness, controls, paired-delta discipline, HBM-by-bytes accounting, and pre-registration mechanics are genuinely strong... But the hypothesis's native/non-native definitions are asymmetrically convenient for both flagship methods, the phase dimension and every novel cell are gated behind expected-positive confirmations... the universal-fragility classification contradicts the falsification condition on the most likely outcome, and the statistical layer cannot support a 10% verdict on 108–231 examples."

Resolved by fixes 1–9. Freeze proceeds on the corrected versions.