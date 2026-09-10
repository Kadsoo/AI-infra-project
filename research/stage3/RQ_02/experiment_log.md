# RQ-2 Experiment Log (Stage 3B)

Chronological entries for all RQ-2 experiments. Append-only; each entry records timestamp, purpose, sources, success/failure, and output paths. Frozen contract: `LOCKED_PLAN.md` (do not modify).

---

## 2026-08-27 — E1: Offline desk triage of published tables

- **Purpose:** Execute E1 per `experiment_plan.md` §E1 — cost-tier-1 screening of published tables to produce the per-method×condition triage matrix deciding E2–E5 cell priority and the K1/K2/K3 kill flags. Screening only; no hypothesis verdicts.
- **Sources used (read-only):** `research/paper_notes/SnapKV.md`, `KIVI.md`, `GEAR.md`, `PyramidKV.md`, `SCOPE.md`, `StreamingLLM.md`, `H2O.md`; `research/stage3/RQ_02/LOCKED_PLAN.md` (§1–§7 criteria: ±2-point band, ≥2-dataset crossing rule, floor/ceiling exclusions, K1/K2/K3); `research/stage3/RQ_02/experiment_plan.md` (E1 §"Experimental Conditions" cell list, authoritative).
- **Method:** Re-extracted every plan-transcribed number from the notes; computed paired relative deltas = (compressed−full)/full×100 mechanically; flagged eligibility per the frozen tolerance (quality ≥10%, TPOT ≥15%, floor <5, ceiling >95). 31 plan-transcribed items checked: 28 verified, 1 number-level rounding mismatch (−29.3% → −29.24% on a not-used cell), 1 citation mismatch (KIVI truncation: plan cites §9, notes record it in §6/§8), 1 rounding mismatch; all 10 "measure"/missing flags confirmed absent from notes.
- **Key determinations:** PyramidKV KV=64 native −24.6% ≥10% → K1 fires (ambiguous track, E5-a); K2 prunes KIVI G∈{32,64} on GSM8K (+1.1% flat) and StreamingLLM sink sweep on Falcon (0.0% flat); K3 fires (no cell simultaneously native-safe and ≥10%-crossing → E2 shrinks to minimal verification set, falsification-first posture, E5 open cells unconditional). SCOPE carries the only strong H2 anchor (TPOT +100%, Table 3) and the only H3 gain anchors (+88% phase-split, +42% frequency).
- **Success/Failure:** SUCCESS — triage matrix, transcription-verification ledger, H3-knob screening table, and K1/K2/K3 flags produced; no GPU, no code execution beyond arithmetic.
- **Output paths:** `processed/E1_triage_matrix.csv` (54 cells), `processed/E1_report.md` (full analysis + verification ledger). No `result.md` written (senior-analyst verdict file), no existing `.md` modified.
