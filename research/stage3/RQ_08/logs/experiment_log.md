# Experiment Log — RQ-8 Stage 3B, Wave-1 (CPU-only)

| | |
|---|---|
| Experiment | E1 — Analytical/numerical screening of the tier axis (corpus cost models) |
| RQ | RQ-8 (retained-token x bit-width x tier non-additivity) |
| Kill-gate role | E1 can KILL H2 (conditional, screening kill). CANNOT kill H1, primary, or H3. |
| Frozen contract | `LOCKED_PLAN.md` (2026-08-27); spec `experiment_plan.md` §E1; env `environment.md` |
| Date | 2026-08-27 22:29 (CST) |
| Machine | LAPTOP-1PB54QSI, Windows 11, Python 3.13.5 (stdlib only; no numpy dependency) |
| GPU | None used (E1 is offline analytical; environment.md GPU-gated cells not run) |

## Runs

| Timestamp | Command | Purpose | Success | Artifacts |
|---|---|---|---|---|
| 2026-08-27 22:29 | `python code/e1_screening.py` (cwd: research/stage3/RQ_08) | Compute 12 tier cells x 2 workloads x 2 F values: actual retained bytes (per component), transfer bytes/step, fetch count, tier latency, GPU compute estimate, byte-normalized latency, S index, regime check, peak-interaction cell; write raw tables + report | SUCCESS — exit 0; 48 rows written; verdict flag produced | `code/e1_screening.py`, `raw/e1_cells.csv`, `raw/e1_regime.csv`, `processed/e1_report.md`, this log |

## Script integrity

- `code/e1_screening.py` SHA-256: `0f0c03a09181805022519a0b90cd43046179dd75322dd530fd30b7992d0e3070`
- Matches the hash recorded in `processed/e1_report.md` (both artifacts from the same final run).

## Result (summary)

- **S (super-additivity index, corner 20%/2-bit):** 0.000 at all 4 workload x F combinations (corner tier penalty = 0 under the FlexGen `T = max(I/O, compute)` rule; 0/0-in-form treated as 0 because the additive bound holds trivially). Raw-transfer robustness reading: S = 0.155–0.163 (W1), 0.306–0.312 (W2) — all far below the frozen 1.15 bar.
- **Byte-normalized tier latency spread (host cells):** F=10.55 µs → 28.8–29.3%; F=2.0 µs → 6.6–6.8%. Not constant within ±5% at any combination; >10% variation only at F=10.55.
- **Regime check:** W1 testable (not all cells compute-dominant; (100%/60%, 16-bit) host cells I/O-dominant, tier fraction up to 79%); W2 tier-untestable at every cell (max(I/O, compute) = compute everywhere; pre-registered Ambiguous Outcome).
- **Peak-interaction cell (E3 look-first):** W1 → (100%, 2-bit, GPU+host); W2 → (60%, 2-bit, GPU+host).
- **VERDICT FLAG: H2 ANALYTICALLY FALSIFIED (E1 screening kill; W2 tier-untestable per workload).** Per LOCKED_PLAN §7: E2b is warranted only if the machine-measured constants (or E2b microbenchmarks) resurrect the mechanism; H1/primary/H3 untouched (not assessed by E1).

## Files

- `code/e1_screening.py` — the script (SHA-256 above)
- `raw/e1_cells.csv` — 48 rows (12 cells x 2 workloads x 2 F), 22 columns, every computed quantity
- `raw/e1_regime.csv` — 48 rows, regime quantities per cell (T_gpu, T_transfer, max-rule winner, tier fraction, compute-dominant flag)
- `processed/e1_report.md` — per-cell table, S, constancy, regime, verdict flag, peak cell, constant-set citation log, substitution + sensitivity notes
- No `result.md` written (senior-analyst territory). No existing .md files modified.

## Notes

- Reproduce: `python code/e1_screening.py` from `research/stage3/RQ_08` (deterministic; no random seeds needed).
- Constants: published set only (ShadowKV PCIe 31.5 GB/s / HBM 2 TB/s / α=60%; Beluga F=10.55 µs; KIVI G=32 R=128; GEAR 21.7% anchor; FlexGen max rule; InfiniGen 96.9% anchor). Substitution note for measured A100 constants is pre-registered in the report (§7) — ratio conclusions are machine-invariant by construction (raw S invariant to shared B_PCIe and α).