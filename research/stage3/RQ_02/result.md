# Research Question

**RQ-2:** How fragile are fixed retention/precision/chunk budgets? (At each method's own published operating point, does the embedded task- and phase-specific importance assumption cause quality/latency to cross predeclared tolerances on out-of-suite conditions, or beatable by same-method same-budget re-tuning?)

# Hypotheses

- **H1 (quality fragility):** a method at its own published budget crosses the ≥10% tolerance (PPL ≥15%) on ≥1 out-of-suite condition (∞Bench En.Sum long-form, CoT reasoning, long-decode phase) while its native in-suite delta stays <10%.
- **H2 (TPOT/kernel overhead):** at matched batch ≥8 and matched measured HBM, TPOT P50 ≥15% above full-KV, or per-step kernel overhead >10% of decode-step time, or ≥2× KV-representation HBM saving with <5% throughput gain, on ≥1 condition.
- **H3 (oracle gap):** same-method, same-implementation, same-measured-budget re-tuning of the method's OWN knobs achieves ≥10% quality or ≥15% TPOT improvement on ≥1 condition.

# Experimental Setup

Wave-1 executed only the **E1 offline triage** (desk analysis of the corpus paper notes — the only CPU-only gate; E1 never issues hypothesis verdicts by design). E2–E5 (quality matrix, TPOT/kernel microbenchmark, oracle re-tuning, open-value cells) require a single A100-80GB (frozen control hardware) — **blocked on this machine** (8 GB laptop GPU, no PyTorch; see `environment.md`).

E1 inputs: `paper_notes/{SnapKV,KIVI,GEAR,PyramidKV,SCOPE,StreamingLLM,H2O}.md`; output: `processed/E1_triage_matrix.csv` (52 cells) + `processed/E1_report.md`. Every plan-transcribed number was re-extracted from the notes: 28 verified exactly, 2 minor mismatches (SnapKV LWM GovReport −29.25% vs −29.3% rounding; KIVI truncation note section §9 vs §6/§8), 10 cells correctly pre-flagged "measure" (no in-paper anchor).

# Baseline

Within-paper full-KV rows of the same tables (same model, same harness) — matched-harness baselines are produced only in E2+.

# Results (E1 triage, screening-only)

| Method | Native (in-suite) delta | Out-of-suite H1 cells | Triage |
|---|---|---|---|
| SnapKV (cap 1024) | −4.8% NrtvQA (Mistral) — native-safe | GovReport-Mistral, QMSum, ∞Bench En.Sum, GSM8K — **no in-paper value** | measure-flagged; open |
| KIVI (2-bit G=32 R=128) | −1.3% CoQA, −5.6% GSM8K (Llama-2-7B) — native-safe | ∞Bench En.Sum, long-decode — **no in-paper value** | measure-flagged; open |
| GEAR (2-bit s=2% r=4) | CoT −0.8%; LongBench-21 −5.0% (borderline) | per-task unresolved | E5-c unconditional |
| PyramidKV (α=8, KV=64) | **−24.6% Mistral / −16.2% LLaMA-3-8B — native-UNSAFE** | per-task unrecorded | **K1 fires** → witness-ineligible, ambiguous/universal-fragility track, E5-a |
| StreamingLLM (4 sinks+2048) | PPL stable to 4M tokens — native-safe | retrieval-at-evicted-distance never tested | E5-b unconditional |
| SCOPE (λ1+λ2≈60%) | LongGenBench −6.0/−7.9% — native-safe | (strongest anchors: TPOT +100% Slide vs Full; phase-split +88%) | top E3/E4 priority |
| H2O (20% KV) | ≤1.2% — native-safe | long-form/CoT **no in-paper data** | E5-d unconditional |

H3-knob screening (in-paper same-budget sweeps): KIVI G 32→64 = +1.1%, G→128 = −16.8%, R sweep ±0.9% → **predicted oracle gain <5% for G∈{32,64}** (pruned); StreamingLLM sink count flat (pruned); SCOPE phase-split +88% and selection-frequency +42% → **predicted ≥10%/≥15%** (survives).

Kill flags: **K1 fires** (PyramidKV native ≥10% → cannot witness H1); **K2 prunes** KIVI G∈{32,64} and StreamingLLM sink sweeps from E4; **K3 fires** — no (method, dimension) cell is simultaneously native-safe AND ≥10%-crossing → E2 shrinks to a minimal verification set and the posture flips to **falsification-first** (E5 open cells unconditional).

# Main Observations

1. All H1-eligible out-of-suite cells for SnapKV/KIVI are "measure" cells — the published corpus contains NO numbers for exactly the conditions H1 is about; the triage could not predict a crossing and could not prune one.
2. PyramidKV at its published KV=64 is the only method with a published native drop ≥10% — it is disqualified as an H1 witness and sits on the universal-fragility track (the "static setting is globally unsafe" finding, which falsifies H1's attribution but is a finding of its own).
3. SCOPE carries the only in-paper anchors that meet H2/H3 thresholds verbatim (TPOT +100%; phase-split +88% at equal budget).
4. K3's fire means the falsification-first posture: the E2 matrix is a verification set, and the open-value cells (E5) are mandatory before any H1 verdict.

# Confounders

- Cross-paper numbers are screening-weak (only within-paper pairs used for classification).
- Falcon-7B (MQA) cells are in-suite observations (never H1 evidence per revision R1); Falcon GSM8K baseline 4.55 < 5 (floor-excluded); PyramidKV aggregates hide per-task spread.
- No matched-harness data exists on this machine — no GPU, no HF Transformers, no method repos.

# Alternative Explanations

- The triage's "no crossing candidate" could mean (a) fragility is genuinely absent in-suite, (b) fragility lives only in out-of-suite conditions the corpus never measured, or (c) the corpus is selection-biased (papers report favorable tasks). (b)/(c) are exactly why E5's open cells are unconditional — they are the first place the corpus has never looked.

# Control Experiments

- K1/K2/K3 flags are the pre-registered pruning controls; the E5-a/b/c/d open-value cells (PyramidKV per-task, StreamingLLM evicted-distance, GEAR per-task, H2O long-form/CoT) are the required verification controls before any verdict.
- Budget scan slots (SnapKV 1024/2048/4096; KIVI 2/4-bit) classify universal-vs-task-specific fragility (unified classification, revision R6).

# Hypothesis Verdict

**H1: INCONCLUSIVE** — untestable at this gate (E2/E5 need A100). The triage's falsification-first posture means H1's verdict requires the matched-harness matrix; no screening-level verdict is permitted by design.

**H2: INCONCLUSIVE** — E3 (TPOT/kernel microbenchmark) needs A100. In-paper anchor (SCOPE Slide vs Full, TPOT +100%) is the strongest surviving candidate; KIVI G∈{32,64} TPOT sweeps were pruned.

**H3: INCONCLUSIVE** — E4 (oracle re-tuning) needs A100. SCOPE phase-split (+88%) and selection-frequency (+42%) anchors predict ≥10%/≥15% oracle gains; KIVI G and StreamingLLM sink sweeps pruned as flat.

# Effect Size

n/a at this gate (no matched-harness measurement). Screening-level predictions: H2/H3 candidate effect sizes are large for SCOPE (+88–100%), flat (<5%) for KIVI G and StreamingLLM sinks.

# Practical Importance

**Low** at this stage (the triage is a desk deliverable). Its value: the E2 matrix shrinks to a verification set; the E5 open cells become the first-priority GPU work; SCOPE becomes the highest-value H2/H3 cell — saving ~1–2 GPU-days of mis-triage.

# Reproducibility

Desk analysis only; every cell is cited to a note file + section in `processed/E1_triage_matrix.csv` and `processed/E1_report.md` (verification ledger included). Re-extraction command: none (no code; data from `paper_notes/*.md`).

# Recommended Next Step

- **Needs more evidence (GPU):** run E2 wave 1 (SnapKV A1+A2+A3, KIVI B1+B2, ≈20 runs) on an A100-80GB; E5 open-value cells run unconditionally in the same wave per the K3-fires posture. If E2's wave-1 cells all stay within tolerance and E5 finds no crossing, H1 is falsified (no out-of-suite crossing, native-safe); if a crossing appears with native <10%, H1 is supported.
- Triage already prunes KIVI G∈{32,64} and StreamingLLM sink sweeps from E4 (~half a day of A100 saved).
- Proceed to Stage 3C for adjudication; hardware acquisition is the blocker.