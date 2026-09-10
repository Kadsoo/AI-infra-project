# E1 Report — Offline Desk Triage of Published Tables (RQ-2, Static Budget Fragility)

- **Experiment:** E1 (cost tier 1: trace/existing-data analysis; 0 GPU, no code execution beyond arithmetic on corpus paper notes)
- **Date:** 2026-08-27
- **Agent:** data-analysis agent (screening only; NO hypothesis verdicts in this report)
- **Sources (read-only):** `research/paper_notes/{SnapKV,KIVI,GEAR,PyramidKV,SCOPE,StreamingLLM,H2O}.md`; contract `RQ_02/LOCKED_PLAN.md` §1–§7; spec `RQ_02/experiment_plan.md` E1 §"Experimental Conditions"
- **Outputs:** `processed/E1_triage_matrix.csv` (per-method×condition matrix, ~54 cells), this report, `experiment_log.md` entry
- **Operating points (hypothesis.md, frozen):** SnapKV cap 1024/window 32/kernel 7; KIVI 2-bit G=32 R=128; GEAR 2-bit s=2% r=4 n_b=20; PyramidKV α=8 β=20 KV=64; StreamingLLM 4 sinks+rolling 2048; SCOPE λ1+λ2=2048/4096 ≈60%, λ2=256; H2O 20% KV.
- **Tolerance mechanics applied:** relative delta = (compressed − full)/full ×100; quality crossing ≥10% (PPL-type ≥15% rise); TPOT crossing ≥15%; floor exclusion baseline <5; ceiling exclusion baseline >95; single-dataset crossings do not count; native-safe = |native delta| <10%. (±2-point non-discriminating band is an E2 bootstrap-CI construct — locked plan §3/§5; not applied to E1 arithmetic.)

---

## 1. Transcription verification ledger (plan → notes)

Every number in `experiment_plan.md` E1 §"Experimental Conditions" was re-extracted from the notes. Verdict legend: **VERIFIED** (matches within rounding), **MISMATCH** (differs), **MISSING** (not found in notes — flagged "measure" as the plan itself requires).

| # | Plan-transcribed number | Note source | Note value | Verdict |
|---|---|---|---|---|
| 1 | SnapKV Mistral NrtvQA 26.82→25.54 = −4.8% | SnapKV.md §10 Table 1 (l.147) | 26.82→25.54, −4.77% | **VERIFIED** (rounding) |
| 2 | SnapKV LWM GovReport −29.3% (not used) | SnapKV.md §10 Table 1 (l.144) | 27.97→19.79 = −29.25% | **MISMATCH** (~0.1pp rounding: −29.2% vs −29.3%; substance identical, both LWM column) |
| 3 | SnapKV GovReport-on-Mistral → "measure" | SnapKV.md | no Mistral GovReport value anywhere | **VERIFIED** (correctly flagged missing) |
| 4 | SnapKV QMSum → "measure" | SnapKV.md | no QMSum value | **VERIFIED** (missing) |
| 5 | SnapKV ∞Bench En.Sum, GSM8K 8-shot CoT → "measure" | SnapKV.md | paper never evaluates these | **VERIFIED** (missing) |
| 6 | KIVI Llama-2-7B CoQA 63.88→63.05 = −1.3% | KIVI.md §10 Table 3 (l.147–150) | −0.83 pts, −1.30% | **VERIFIED** |
| 7 | KIVI Llama-2-7B GSM8K 13.50→12.74 = −5.6% | KIVI.md §10 Table 3 | −0.76 pts, −5.63% | **VERIFIED** |
| 8 | KIVI Falcon CoQA −3.9% | KIVI.md §10 Table 3 (l.156–158) | 59.83→57.48, −3.93% | **VERIFIED** |
| 9 | KIVI Falcon GSM8K −25.1% (floor-excluded) | KIVI.md §10 Table 3 | 4.55→3.41, −25.05%; baseline 4.55<5 → floor | **VERIFIED** |
| 10 | KIVI Falcon MultiNews −38.9% | KIVI.md §10 Table 4 (l.169) | 11.09→6.78, −38.86% | **VERIFIED** |
| 11 | KIVI Falcon TREC −23.1% | KIVI.md §10 Table 4 | 13→10, −23.08% | **VERIFIED** |
| 12 | KIVI Falcon truncation at max_seq 4096 "KIVI.md §9" | KIVI.md §6 (l.74), §8 (l.110) | "Max sequence length … 4096 for others" recorded in §6/§8, **not §9** | **VERIFIED substance / MISMATCH citation** |
| 13 | KIVI ∞Bench En.Sum + long-decode phase → "measure" | KIVI.md | paper has no ∞Bench, no forced-long-decode cell | **VERIFIED** (missing) |
| 14 | GEAR CoT avg 40.20 vs FP16 40.52 = −0.8% | GEAR.md §10 Table 1 (l.99) | −0.32 pts, −0.79% | **VERIFIED** |
| 15 | GEAR LongBench-21 2-bit 25.48 vs 26.82 = −5.0% | GEAR.md §10 Table 2 (l.105) | −1.34 pts, −5.00% | **VERIFIED** |
| 16 | GEAR per-task unresolved → E5-c | GEAR.md §10 Table 2 | only QMSum/SAMSum/GovReport GEAR-vs-KIVI pairs; no GEAR-2bit-per-task vs FP16 | **VERIFIED** (missing) |
| 17 | PyramidKV Mistral avg 32.19 vs FKV 42.71 = −24.6% | PyramidKV.md §10 Table 1 (l.139) | −10.52 pts, −24.63% | **VERIFIED** |
| 18 | PyramidKV per-task unrecorded → ambiguous/E5-a | PyramidKV.md §10 Table 1 | only TREC 54.00, Qasper 20.21 (vs SKV/H2O, no FKV pair) | **VERIFIED** (missing) |
| 19 | StreamingLLM PG19 PPL stable to 4M | StreamingLLM.md §10 Fig.5 (l.142) | stable across 100 books/4M tokens | **VERIFIED** |
| 20 | StreamingLLM no retrieval-at-evicted-distance test | StreamingLLM.md §13 (l.193) | StreamEval answers 20 lines prior (within window) | **VERIFIED** (missing → E5-b) |
| 21 | SCOPE H2 anchor Full 36.57 vs Slide 18.28 tok/s = +100% TPOT | SCOPE.md §10 Table 3 (l.106); §9 (batch 8, RTX 3090) | 36.57/18.28 = 2.0005 → TPOT +100% | **VERIFIED** |
| 22 | SCOPE Discontinuous 25.92 = +41% | SCOPE.md §10 Table 3 | 36.57/25.92 = 1.4109 → TPOT +41.1% | **VERIFIED** |
| 23 | SCOPE H3 anchor GSM8K+ 27.75 → 52.17 = +88% (35% budget) | SCOPE.md §10 Table 2 (l.104); §6 (l.56) | 24.42/27.75 = +88.0%; "35% 总预算" present | **VERIFIED** |
| 24 | H2O OpenBookQA −0.5% | H2O.md §10 Table 4 (l.139) | 43.20→43.00 (−0.46%), 44.40→44.20 (−0.45%) | **VERIFIED** |
| 25 | H2O RTE +0.73% | H2O.md §10 (l.132) | OPT-66B RTE +0.73% | **VERIFIED** |
| 26 | H2O no non-native data → E5-d | H2O.md | no ∞Bench, no CoT; XSUM/CNN-DM in-suite short-form | **VERIFIED** (missing) |
| 27 | H3: KIVI G 32→64 GSM8K 20.77→21.00 (+1.1%) | KIVI.md §10 Table 5 (l.176–178) | +1.11% | **VERIFIED** |
| 28 | H3: KIVI G→128: 17.29 (−16.8%) | KIVI.md §10 Table 5 | −16.75% | **VERIFIED** |
| 29 | H3: KIVI R sweep 32/64/96/128 = 20.62/19.86/20.55/20.77 | KIVI.md §10 Table 5 | identical | **VERIFIED** |
| 30 | H3: StreamingLLM sinks 1 vs 4 on Falcon flat | StreamingLLM.md §10 Table 2 (l.135) | 12.12 = 12.12 (0-sink 17.90) | **VERIFIED** |
| 31 | H3: SCOPE phase-split +88%, frequency +42% | SCOPE.md §10 Table 2/3 | +88.0%; +41.8% tok/s | **VERIFIED** |

**Summary:** 28/31 VERIFIED; 1 number-level MISMATCH (#2, 0.1pp rounding on a not-used cell); 1 citation MISMATCH (#12, §9→§6/§8, substance unchanged); all "measure"/"missing" flags (#3–5, #13, #16, #18, #20, #26) confirmed against notes. No invented numbers anywhere in this report.

---

## 2. Per-method triage rows (plan's cell list, verified)

Each row follows `experiment_plan.md` E1 §"Experimental Conditions" verbatim; numbers as re-extracted.

### SnapKV (capacity 1024, window 32, kernel 7)
Native = LongBench 16 incl. summarization (R1). E2 target model Mistral-7B-Instruct-v0.2.
- NrtvQA Mistral: 26.82→25.54 = **−4.8%** [SnapKV.md §10 Table 1] → **native-safe** (<10%), H1 witnessable.
- **GovReport-on-Mistral: no in-paper value → "measure"** (E2 A4 observation cell; the −29.2% [plan: −29.3%] figure is the LWM-Text-Chat column [SnapKV.md §10 Table 1 l.144] and is NOT used for a Mistral prediction — single-dataset + model-mismatch).
- QMSum-on-Mistral: no in-paper value → **"measure"** (E4 lists SnapKV on QMSum).
- Non-native H1 cells: ∞Bench En.Sum, GSM8K 8-shot CoT → **"measure"**, no anchor (E2 A2; open).
- Long-decode phase cell (E2 A3, unconditional): no in-paper anchor → open.
- **Triage verdict: native-safe; crossing unpredicted (no anchor); 4 open measure cells. Retains H1 witness eligibility.**

### KIVI (2-bit, G=32, R=128)
Native track = Llama-2-7B/13B, Mistral-7B (CoQA/TruthfulQA/GSM8K + LongBench-8); Falcon cells are in-suite observations (R1), never H1 evidence.
- Llama-2-7B native: CoQA 63.88→63.05 = **−1.3%**; GSM8K 13.50→12.74 = **−5.6%** [KIVI.md §10 Table 3] → **native-safe**.
- Supplementary: Llama-2-13B GSM8K −8.4%, Mistral GSM8K −6.1%, Mistral LongBench avg −1.6%, Llama-2-7B LongBench avg −0.6% — all native-safe (KIVI.md §10 Tables 3–4).
- Falcon observations: CoQA **−3.9%** (safe); GSM8K **−25.1%** (FLOOR-EXCLUDED, baseline 4.55<5); MultiNews **−38.9%** (11.09→6.78, not floor-excluded); TREC **−23.1%** (13→10) [KIVI.md §10 Tables 3–4] — published-point validity observations; **2 non-floor LongBench tasks cross ≥10% on the MQA layout** → layout-specific fragility warning on the in-suite observation track (feeds E2 B3 + R6 universal-fragility classification; cannot witness H1).
- Non-native H1 cells: ∞Bench En.Sum, long-decode (output ≥1024) → **"measure"**, no anchor (E2 B2; open).
- **Triage verdict: native-safe on H1 track; crossing unpredicted (no anchor); Falcon observation track shows 2–3 crossing cells (in-suite). Retains H1 witness eligibility; Falcon obs cells are the strongest in-paper ≥10% pair and must be recorded (not counted) in E2 B3.**

### GEAR (2-bit, s=2%, r=4, n_b=20)
- Native CoT trio avg (GSM8k/AQuA/BBH 8-shot, 9 model×dataset combos): 40.20 vs FP16 40.52 = **−0.8%** [GEAR.md §10 Table 1] → **native-safe**. Per-model: LLaMA3-8B −2.05%, LLaMA2-13B +0.84%, Mistral −0.52% (all safe).
- LongBench-21 2-bit (LLaMA-2-7B): 25.48 vs 26.82 = **−5.0%** → **borderline, per-task unresolved** (per-task GEAR-2bit vs FP16 pairs not in notes) → **E5-c unconditional** [GEAR.md §10 Table 2].
- Supplementary: GEAR 4-bit LongBench avg +3.7% (above FP16); 4-bit CoT +0.7% — native-safety robust.
- **Triage verdict: native-safe; no ≥10% crossing in-paper; one borderline aggregate (−5.0%) → E5-c resolves per-task.**

### PyramidKV (α=8, β=20, KV=64)
- Mistral-7B LongBench avg: 32.19 vs FKV 42.71 = **−24.6%** [PyramidKV.md §10 Table 1] → **native delta ≥10% → K1 fires (H1 witness-ineligible) → ambiguous/universal-fragility track, excluded from E2 kill matrix.**
- Supplementary: LLaMA-3-8B KV=64 −16.2% (also ≥10%), LLaMA-3-70B KV=64 −9.75% (borderline); KV=2048 safe on all three (8B +0.07%, Mistral −2.53%, 70B tied).
- Per-task native values not recorded in notes → **ambiguous until E5-a (unconditional)** [PyramidKV.md §10].
- **Triage verdict: native-UNSAFE at the frozen operating point → K1; per-task resolution deferred to E5-a; KV=2048 safe (relevant to budget-scan classification in E5-e).**

### StreamingLLM (4 sinks + rolling, total 2048)
- PG19 PPL stable to 4M tokens; streaming QA matches one-shot (7B 71.34 vs 71.25; 13B 80.89 vs 78.16; 70B 91.37 vs 91.29) [StreamingLLM.md §10 Fig.5, Table 5] → **native-safe (PPL-type; no rise)**.
- Sink ablation (Table 2): Falcon 1 vs 4 sinks identical (12.12); MPT 14.99=14.99; Pythia 11.95→12.09; Llama-2-7B 4 vs 8: 9.59→9.54.
- **No in-paper test of retrieval at evicted distance** (StreamEval answers lie within the window; note §13 flags the gap explicitly) → **new measurement required, E5-b unconditional** (needles beyond the window at 3000/5000/10000).
- **Triage verdict: native-safe; crossing unpredicted (no in-paper anchor); E5-b is the only H1-adjacent cell and is open-value.**

### SCOPE (λ1+λ2=2048/4096 ≈60%, λ2=256)
- **H2 anchor:** Full 36.57 vs Slide 18.28 tok/s (batch 8, RTX 3090, eager) = **TPOT +100%**; Discontinuous 25.92 = **TPOT +41.1%** [SCOPE.md §10 Table 3, §9] → in-paper TPOT anchors ≥15% tolerance.
- **H3 anchor:** GSM8K+ prefill-only 27.75 (SnapKV/PyramidKV) → prefill+decode 52.17 = **+88%** at same 35% total budget [SCOPE.md §10 Table 2] — phase-split re-tuning anchor ≥10% quality.
- **H3 knob:** selection frequency Discontinuous vs Slide = **+41.8% tok/s** (TPOT −29.5%) — own-knob change only.
- Supplementary native quality cells: LongGenBench-4K Slide −6.0%, 8K −7.9%, GSM8K+ Slide −2.1% [SCOPE.md §10 Tables 1–2] → **native-safe** on quality.
- **Triage verdict: native-safe; no H1-quality crossing in-paper; but the STRONGEST H2 and H3 anchors in the corpus (TPOT +100%; +88% phase-split; +42% frequency). Top priority for E3 (cell D) and E4 (knob re-tuning).**

### H2O (20% KV)
- Native lm-eval/HELM all <10%: OpenBookQA −0.5%/−0.5% (OPT-30B/66B), RTE +0.73% (66B) [H2O.md §10 Table 4, main results] → **native-safe**. Supplementary: COPA −1.2%/−1.2%, MathQA +2.4%/−0.7%, XSUM +0.18.
- **No in-paper non-native (long-form/CoT) data** → no signal, excluded from E2-primary; long-form (∞Bench En.Sum) and CoT (GSM8K) cells in **E5-d unconditional** [H2O.md §10; §13 open questions].
- **Triage verdict: native-safe; crossing unpredicted (no anchor); E5-d open-value cells.**

### H3-knob screening (in-paper same-budget sweeps)
| Method | Knob (E4 range) | In-paper sweep (note §/Table) | Computed max gain | Prediction | E4 action |
|---|---|---|---|---|---|
| KIVI | G∈{32,64} (GSM8K, R=128) | 32→20.77, 64→21.00 [KIVI.md §10 Table 5] | +1.1% | **<5% (flat)** | **K2 PRUNE** |
| KIVI | G=128 (same) | 20.77→17.29 | −16.8% (degrades) | — | do not sweep G≥128 (budget-equivalent exclusion is G∈{32,64} per plan) |
| KIVI | R∈{32,64,96,128} (G=32) | 20.62/19.86/20.55/20.77 [KIVI.md §10 Table 5] | +0.7% (R=64 worst, non-monotonic) | **<5% (flat)** | **K2-extension candidate** (same criterion; plan pruned only G — flag for analyst) |
| StreamingLLM | sink count 1–8 (Falcon) | 1+2047 12.12 = 4+2044 12.12 [StreamingLLM.md §10 Table 2] | 0.0% | **<5% (flat)** | **K2 PRUNE** |
| StreamingLLM | sink count (other models) | Llama-2 9.59→9.54 (4→8); MPT 14.99=14.99; Pythia 11.95→12.09 | <0.6% | <5% (flat) | K2-extension candidate |
| SCOPE | phase-split (λ1/λ2, decode budget) | 27.75→52.17 = +88% [SCOPE.md §10 Table 2] | +88% | **≥10% quality** | **KEEP — top H3 candidate** |
| SCOPE | selection frequency | 18.28→25.92 = +41.8% tok/s [SCOPE.md §10 Table 3] | +42% tok/s (TPOT −29.5%) | **≥15% TPOT** | **KEEP** |
| GEAR | n_b, r (GSM8k-CoT, 2-bit) | Fig.4a: s=2%/r=4 adequate; "further increase not significant" [GEAR.md §10] | n/a (ablation, single model/task) | <5% (weak evidence) | screening-weak prune candidate — flag for analyst |

### H2 mechanism anchors (for E3 ordering)
| E3 cell | In-paper anchor | Strength |
|---|---|---|
| KIVI 2-bit Llama-2-7B (MHA, cell A) | no per-step anchor (throughput 2.35–3.47× is memory/batch-driven, KIVI.md §10) | none |
| KIVI 2-bit Falcon-7B (MQA, cell B) | mechanism-only: "MQA layout dequant fixed-cost share rises" (hypothesis); no in-paper TPOT number | weak (no anchor) |
| GEAR 2-bit (n_b=20 recompression, cell C) | **counter-evidence**: time breakdown shows low-rank+sparse negligible, quant fused [GEAR.md §10 Fig.3a] | negative in-paper |
| SCOPE Slide (cell D) | **+100% TPOT** [SCOPE.md §10 Table 3] | **strong positive anchor** |

E3 runs A→B→C→D per plan; the triage notes that the strongest in-paper H2 anchor belongs to the LAST cell (D). If A–C all survive, D is the most likely kill; the plan's fixed order is preserved (no re-ordering authority at E1).

---

## 3. K1 / K2 / K3 kill flags

### K1 — H1 witness eligibility (native-safety precondition)
- **PyramidKV KV=64: FIRES.** Native LongBench delta −24.6% (Mistral), −16.2% (LLaMA-3-8B) ≥10% [PyramidKV.md §10 Table 1]. Moved to the **ambiguous outcome / universal-fragility track**, excluded from the E2 kill matrix; per-task resolution → E5-a (unconditional).
- **KIVI: does NOT fire** — Falcon ≥10% cells (MultiNews −38.9%, TREC −23.1%) are in-suite observation cells (revision R1, plan §E1 KIVI bullet) and cannot strip witness eligibility; the H1-eligible native track (Llama-2-7B/13B, Mistral) is <10% everywhere. Falcon crossings are recorded as B3 observations and feed the R6 universal-fragility classification only.
- All other methods: native-safe (SnapKV −4.8%; GEAR −0.8%/−5.0%; StreamingLLM PPL stable; SCOPE −6.0%/−7.9%; H2O ≤1.2%) → **no other K1 firing**.

### K2 — E4 oracle sweep pruning (in-paper flat <5%)
- **KIVI G∈{32,64} on GSM8K: PRUNED** (+1.1% max in-paper; G=128 degrades −16.8%) [KIVI.md §10 Table 5].
- **StreamingLLM sink count on Falcon: PRUNED** (1 vs 4 sinks identical) [StreamingLLM.md §10 Table 2].
- Extension candidates (same <5% criterion, analyst decision): KIVI R∈{32,64,96,128} (+0.7% spread); StreamingLLM sinks on MPT/Pythia/Llama-2; GEAR n_b/r (screening-weak, single-model ablation).
- **NOT pruned:** SCOPE λ1/λ2 phase-split (+88%) and selection frequency (+42%) — the only H3 gain anchors in the corpus.

### K3 — (method, dimension) cell simultaneously native-safe AND ≥10%-crossing
- **FIRES.** Enumerating every in-paper ≥10% delta cell: SnapKV LWM GovReport −29.2% (model-mismatch, single-dataset, not E2 model); KIVI Falcon GSM8K −25.1% (floor-excluded); KIVI Falcon MultiNews −38.9% / TREC −23.1% (in-suite, 2 datasets but observation track — never H1 evidence per R1); PyramidKV Mistral/8B native (native itself ≥10% → K1); KIVI G=128 knob −16.8% (knob sweep, not a budget cell); SCOPE TPOT +100%/+41% (H2 latency, not H1 quality). **No cell is simultaneously native-safe and an out-of-suite ≥10% crossing.**
- Consequence per plan §K3: **E2 shrinks to the minimal verification set** (SnapKV A1+A2+A3; KIVI B1+B2 — all out-of-suite cells are open/measure), **posture flips to falsification-first**, and **E5 open-value cells run unconditionally** (they already do: E5-a/b/c/d). E5 completion parts (e) arm on E2 negative or the universal-fragility rule.
- Pre-registered E2 predictions (unchanged by E1): A1 safe (−4.8% anchor), B1 safe (−1.3%, −5.6% anchors), A2/B2 open, B3 reported-not-evidence. E1 adds no ≥10% crossing prediction anywhere.

---

## 4. Triage verdict summary (per method)

| Method | Native-safe? | ≥10% crossing predicted in-paper? | Flags | E2–E5 consequence |
|---|---|---|---|---|
| SnapKV | YES (−4.8%) | no (4 open cells, no anchor) | measure×4 (GovReport-Mistral, QMSum, ∞Bench En.Sum, GSM8K CoT) | E2 A1+A2+A3 first wave (minimal verification set) |
| KIVI | YES on H1 track | no on H1 track; YES×2 in-suite Falcon obs (MultiNews, TREC) + 1 floor-excluded (GSM8K) | floor×1; obs-track warning | E2 B1+B2 first wave; B3 Falcon observations must be recorded (not counted) |
| GEAR | YES (−0.8%, −5.0% borderline) | no | per-task unresolved | E5-c unconditional |
| PyramidKV | **NO** (native −24.6%) | n/a (native itself crosses) | **K1 fires**; ambiguous; per-task missing | E5-a unconditional; ambiguous track |
| StreamingLLM | YES (PPL stable) | no (no anchor) | evicted-distance untested | E5-b unconditional |
| SCOPE | YES (−6.0%/−7.9%) | no H1-quality crossing | H2 anchor TPOT +100%; H3 anchors +88%/+42% | E3 cell D (strongest H2 anchor); E4 top H3 candidate |
| H2O | YES (≤1.2%) | no (no non-native data) | no anchor | excluded from E2-primary; E5-d unconditional |

---

## 5. Data-integrity ledger

- **Number-level mismatches:** 1 — plan's "−29.3%" (SnapKV LWM GovReport) re-extracts to −29.24% (rounding; cell not used for prediction).
- **Citation-level mismatches:** 1 — KIVI truncation protocol is documented in KIVI.md §6/§8, not §9 as the plan cites; substance identical.
- **Missing in notes (all correctly pre-flagged "measure" by the plan):** SnapKV GovReport-Mistral, QMSum-Mistral, ∞Bench En.Sum, GSM8K CoT; KIVI ∞Bench En.Sum, long-decode phase; GEAR LongBench-21 per-task; PyramidKV per-task; StreamingLLM retrieval-at-evicted-distance; H2O ∞Bench En.Sum / GSM8K CoT; H2O RTE and XSUM absolute baselines (delta-only rows).
- **No numbers were invented; every row in `E1_triage_matrix.csv` carries its note source.**
