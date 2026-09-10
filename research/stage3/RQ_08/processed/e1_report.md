# E1 Report — Analytical Screening of the Tier Axis (corpus cost models)

**RQ-8 Wave-1 CPU-only kill-gate experiment. KILL-GATE: E1 can kill H2 (conditional, screening kill) only. Cannot kill H1, primary, or H3.**

- Date: 2026-08-27 (session date, environment.md)
- Environment: 32-layer Llama-2-7B model card; CPU-only machine (environment.md: no GPU — A100-gated cells blocked; E1 needs none)
- Script: `code/e1_screening.py` (SHA-256 `0f0c03a09181805022519a0b90cd43046179dd75322dd530fd30b7992d0e3070`)
- Raw outputs: `raw/e1_cells.csv` (48 rows: 12 cells x 2 workloads x 2 F), `raw/e1_regime.csv`

## 1. Model summary (per-cell quantities)

12 tier cells (3R x 2B x 2T) x 2 workloads x F in {2.0, 10.55} µs. Actual retained bytes per sequence per layer are per-component (never nominal bits): 16-bit = N x 16 KB; KIVI-2bit (G=32, R=128) = (N-128) x 3072 B (2048 B 2-bit data + 1024 B scales/zero-points) + 128 x 16 KB FP16 residual. GEAR n_b=20 FP16 buffer: not present in this matrix (B in {16-bit, KIVI-2}); recorded as 0.0 and documented — the byte-accounting rule includes it 'where present' (hypothesis confounder 2).

| Workload | R | B | T | F (µs) | N | store/layer (MB) | %FP16 | transfer/step (MB) | fetches/step | T_transfer (ms) | T_gpu (ms) | T_step (ms) | pen (ms) | µs/MB | tier frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1 RAG 4K | 100% | 16bit | GPU-only | 2.0 | 4096 | 67.11 | 100.0 | 0.0 | 0 | 0.00 | 7.81 | 7.81 | 0.00 |  | 0.0% |
| W1 RAG 4K | 100% | 16bit | GPU-only | 10.55 | 4096 | 67.11 | 100.0 | 0.0 | 0 | 0.00 | 7.81 | 7.81 | 0.00 |  | 0.0% |
| W1 RAG 4K | 100% | 16bit | GPU+host | 2.0 | 4096 | 67.11 | 100.0 | 859.0 | 256 | 27.78 | 7.81 | 27.78 | 19.97 | 32.3 | 78.0% |
| W1 RAG 4K | 100% | 16bit | GPU+host | 10.55 | 4096 | 67.11 | 100.0 | 859.0 | 256 | 29.97 | 7.81 | 29.97 | 22.16 | 34.9 | 79.3% |
| W1 RAG 4K | 100% | 2bit | GPU-only | 2.0 | 4096 | 14.29 | 21.3 | 0.0 | 0 | 0.00 | 6.97 | 6.97 | 0.00 |  | 0.0% |
| W1 RAG 4K | 100% | 2bit | GPU-only | 10.55 | 4096 | 14.29 | 21.3 | 0.0 | 0 | 0.00 | 6.97 | 6.97 | 0.00 |  | 0.0% |
| W1 RAG 4K | 100% | 2bit | GPU+host | 2.0 | 4096 | 14.29 | 21.3 | 182.9 | 256 | 6.32 | 6.97 | 6.97 | 0.00 | 34.5 | 47.5% |
| W1 RAG 4K | 100% | 2bit | GPU+host | 10.55 | 4096 | 14.29 | 21.3 | 182.9 | 256 | 8.51 | 6.97 | 8.51 | 1.54 | 46.5 | 55.0% |
| W1 RAG 4K | 60% | 16bit | GPU-only | 2.0 | 2458 | 40.27 | 100.0 | 0.0 | 0 | 0.00 | 7.38 | 7.38 | 0.00 |  | 0.0% |
| W1 RAG 4K | 60% | 16bit | GPU-only | 10.55 | 2458 | 40.27 | 100.0 | 0.0 | 0 | 0.00 | 7.38 | 7.38 | 0.00 |  | 0.0% |
| W1 RAG 4K | 60% | 16bit | GPU+host | 2.0 | 2458 | 40.27 | 100.0 | 515.5 | 160 | 16.68 | 7.38 | 16.68 | 9.30 | 32.4 | 69.3% |
| W1 RAG 4K | 60% | 16bit | GPU+host | 10.55 | 2458 | 40.27 | 100.0 | 515.5 | 160 | 18.05 | 7.38 | 18.05 | 10.67 | 35.0 | 71.0% |
| W1 RAG 4K | 60% | 2bit | GPU-only | 2.0 | 2458 | 9.25 | 23.0 | 0.0 | 0 | 0.00 | 6.89 | 6.89 | 0.00 |  | 0.0% |
| W1 RAG 4K | 60% | 2bit | GPU-only | 10.55 | 2458 | 9.25 | 23.0 | 0.0 | 0 | 0.00 | 6.89 | 6.89 | 0.00 |  | 0.0% |
| W1 RAG 4K | 60% | 2bit | GPU+host | 2.0 | 2458 | 9.25 | 23.0 | 118.5 | 160 | 4.08 | 6.89 | 6.89 | 0.00 | 34.4 | 37.2% |
| W1 RAG 4K | 60% | 2bit | GPU+host | 10.55 | 2458 | 9.25 | 23.0 | 118.5 | 160 | 5.45 | 6.89 | 6.89 | 0.00 | 46.0 | 44.2% |
| W1 RAG 4K | 20% | 16bit | GPU-only | 2.0 | 820 | 13.43 | 100.0 | 0.0 | 0 | 0.00 | 6.95 | 6.95 | 0.00 |  | 0.0% |
| W1 RAG 4K | 20% | 16bit | GPU-only | 10.55 | 820 | 13.43 | 100.0 | 0.0 | 0 | 0.00 | 6.95 | 6.95 | 0.00 |  | 0.0% |
| W1 RAG 4K | 20% | 16bit | GPU+host | 2.0 | 820 | 13.43 | 100.0 | 172.0 | 64 | 5.59 | 6.95 | 6.95 | 0.00 | 32.5 | 44.5% |
| W1 RAG 4K | 20% | 16bit | GPU+host | 10.55 | 820 | 13.43 | 100.0 | 172.0 | 64 | 6.13 | 6.95 | 6.95 | 0.00 | 35.7 | 46.9% |
| W1 RAG 4K | 20% | 2bit | GPU-only | 2.0 | 820 | 4.22 | 31.4 | 0.0 | 0 | 0.00 | 6.81 | 6.81 | 0.00 |  | 0.0% |
| W1 RAG 4K | 20% | 2bit | GPU-only | 10.55 | 820 | 4.22 | 31.4 | 0.0 | 0 | 0.00 | 6.81 | 6.81 | 0.00 |  | 0.0% |
| W1 RAG 4K | 20% | 2bit | GPU+host | 2.0 | 820 | 4.22 | 31.4 | 54.1 | 64 | 1.84 | 6.81 | 6.81 | 0.00 | 34.1 | 21.3% |
| W1 RAG 4K | 20% | 2bit | GPU+host | 10.55 | 820 | 4.22 | 31.4 | 54.1 | 64 | 2.39 | 6.81 | 6.81 | 0.00 | 44.2 | 26.0% |
| W2 GSM8K 900 | 100% | 16bit | GPU-only | 2.0 | 900 | 14.75 | 100.0 | 0.0 | 0 | 0.00 | 6.98 | 6.98 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 100% | 16bit | GPU-only | 10.55 | 900 | 14.75 | 100.0 | 0.0 | 0 | 0.00 | 6.98 | 6.98 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 100% | 16bit | GPU+host | 2.0 | 900 | 14.75 | 100.0 | 188.7 | 64 | 6.12 | 6.98 | 6.98 | 0.00 | 32.4 | 46.7% |
| W2 GSM8K 900 | 100% | 16bit | GPU+host | 10.55 | 900 | 14.75 | 100.0 | 188.7 | 64 | 6.67 | 6.98 | 6.98 | 0.00 | 35.3 | 48.9% |
| W2 GSM8K 900 | 100% | 2bit | GPU-only | 2.0 | 900 | 4.47 | 30.3 | 0.0 | 0 | 0.00 | 6.81 | 6.81 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 100% | 2bit | GPU-only | 10.55 | 900 | 4.47 | 30.3 | 0.0 | 0 | 0.00 | 6.81 | 6.81 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 100% | 2bit | GPU+host | 2.0 | 900 | 4.47 | 30.3 | 57.2 | 64 | 1.94 | 6.81 | 6.81 | 0.00 | 34.0 | 22.2% |
| W2 GSM8K 900 | 100% | 2bit | GPU+host | 10.55 | 900 | 4.47 | 30.3 | 57.2 | 64 | 2.49 | 6.81 | 6.81 | 0.00 | 43.6 | 26.8% |
| W2 GSM8K 900 | 60% | 16bit | GPU-only | 2.0 | 540 | 8.85 | 100.0 | 0.0 | 0 | 0.00 | 6.88 | 6.88 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 60% | 16bit | GPU-only | 10.55 | 540 | 8.85 | 100.0 | 0.0 | 0 | 0.00 | 6.88 | 6.88 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 60% | 16bit | GPU+host | 2.0 | 540 | 8.85 | 100.0 | 113.2 | 64 | 3.72 | 6.88 | 6.88 | 0.00 | 32.9 | 35.1% |
| W2 GSM8K 900 | 60% | 16bit | GPU+host | 10.55 | 540 | 8.85 | 100.0 | 113.2 | 64 | 4.27 | 6.88 | 6.88 | 0.00 | 37.7 | 38.3% |
| W2 GSM8K 900 | 60% | 2bit | GPU-only | 2.0 | 540 | 3.36 | 38.0 | 0.0 | 0 | 0.00 | 6.79 | 6.79 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 60% | 2bit | GPU-only | 10.55 | 540 | 3.36 | 38.0 | 0.0 | 0 | 0.00 | 6.79 | 6.79 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 60% | 2bit | GPU+host | 2.0 | 540 | 3.36 | 38.0 | 43.0 | 64 | 1.49 | 6.79 | 6.79 | 0.00 | 34.7 | 18.0% |
| W2 GSM8K 900 | 60% | 2bit | GPU+host | 10.55 | 540 | 3.36 | 38.0 | 43.0 | 64 | 2.04 | 6.79 | 6.79 | 0.00 | 47.4 | 23.1% |
| W2 GSM8K 900 | 20% | 16bit | GPU-only | 2.0 | 180 | 2.95 | 100.0 | 0.0 | 0 | 0.00 | 6.79 | 6.79 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 20% | 16bit | GPU-only | 10.55 | 180 | 2.95 | 100.0 | 0.0 | 0 | 0.00 | 6.79 | 6.79 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 20% | 16bit | GPU+host | 2.0 | 180 | 2.95 | 100.0 | 37.7 | 32 | 1.26 | 6.79 | 6.79 | 0.00 | 33.4 | 15.7% |
| W2 GSM8K 900 | 20% | 16bit | GPU+host | 10.55 | 180 | 2.95 | 100.0 | 37.7 | 32 | 1.54 | 6.79 | 6.79 | 0.00 | 40.7 | 18.5% |
| W2 GSM8K 900 | 20% | 2bit | GPU-only | 2.0 | 180 | 2.26 | 76.5 | 0.0 | 0 | 0.00 | 6.78 | 6.78 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 20% | 2bit | GPU-only | 10.55 | 180 | 2.26 | 76.5 | 0.0 | 0 | 0.00 | 6.78 | 6.78 | 0.00 |  | 0.0% |
| W2 GSM8K 900 | 20% | 2bit | GPU+host | 2.0 | 180 | 2.26 | 76.5 | 28.9 | 32 | 0.98 | 6.78 | 6.78 | 0.00 | 34.0 | 12.6% |
| W2 GSM8K 900 | 20% | 2bit | GPU+host | 10.55 | 180 | 2.26 | 76.5 | 28.9 | 32 | 1.25 | 6.78 | 6.78 | 0.00 | 43.4 | 15.6% |

Residual fraction note (pre-registered): the FP16 residual R=128 is a constant absolute size (2.097 MB/layer) but a retention-dependent fraction of the retained store: 128/4096 = 3.1% at 100% vs 128/820 = 15.6% at 20% retention (W1); 128/900 = 14.2% at 100% vs 128/180 = 71.1% at 20% (W2). KIVI-2bit store fraction vs FP16 at 100% retention: W1 4096 tokens = 21.3% (published anchor: 21.7%, GEAR Table 1 — the 0.4 pp gap is the G=32/R=128 decomposition vs GEAR's measured g=64/nb=64 config; ratio conclusions are unaffected).

## 2. Super-additivity index S

`S = pen(20%,2-bit) / [pen(20%,16-bit) + pen(100%,2-bit)]`, `pen(R,B) = T_tier(R,B) - T_gpu(R,B)` (FlexGen `T = max(I/O, compute)` rule; pen = max(transfer, compute) - compute). Frozen H2 null: `pen(20%,2-bit) <= 1.15 x [sum]`; screening kill iff `S <= 1.15`.

| Workload | F (µs) | pen(20%,2-bit) ms | pen(20%,16-bit) ms | pen(100%,2-bit) ms | S | S (raw-transfer) |
|---|---|---|---|---|---|---|
| W1_RAG_4K | 2.0 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.1549 |
| W1_RAG_4K | 10.55 | 0.000 | 0.000 | 1.538 | 0.0000 | 0.1633 |
| W2_GSM8K_900 | 2.0 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.3060 |
| W2_GSM8K_900 | 10.55 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.3116 |

Interpretation: the corner penalty is ZERO on every workload x F combination — at (20%, 2-bit) the predicted transfer time (1.84-2.39 ms W1; 0.98-1.25 ms W2) sits far below the GPU compute estimate (6.8-6.9 ms), so the FlexGen max-rule absorbs the entire tier cost into compute overlap. Where the ratio is 0/0 in form, S := 0 because the additive bound `pen(20%,2-bit) <= 1.15 x [sum]` holds trivially (both sides zero). Robustness reading — ignore the max-rule overlap and take raw T_transfer as the tier penalty (worst case for H2): raw S = T_t(20%,2-bit)/[T_t(20%,16-bit)+T_t(100%,2-bit)] = 0.163 (W1, F=10.55) and 0.312 (W2, F=10.55) — transfer is near-linear in actual bytes and the corner is CHEAPER per byte than the single-axis cells, not dearer. Both readings put S far below 1.15.

## 3. Byte-normalized tier latency constancy (A8)

Defined on T=GPU+host cells (6 per workload x F): GPU-only cells transfer 0 bytes — byte-normalized transfer latency is not defined there (the constancy arm of H2's falsification is a statement about transfer).

| Workload | F (µs) | host cells (n) | min µs/MB | max µs/MB | spread % of mean |
|---|---|---|---|---|---|
| W1_RAG_4K | 2.0 | 6 | 32.34 | 34.55 | 6.6% |
| W1_RAG_4K | 10.55 | 6 | 34.89 | 46.51 | 28.8% |
| W2_GSM8K_900 | 2.0 | 6 | 32.42 | 34.72 | 6.8% |
| W2_GSM8K_900 | 10.55 | 6 | 35.32 | 47.43 | 29.3% |

Frozen tolerance: constant within ±5% (noiseless model). Survival requires >10% variation. Result: at F=10.55 µs (Beluga measured) variation is 28.8-29.3% — clearly non-constant. At F=2.0 µs (A5 floor) variation drops to 6.6-6.8% — above the ±5% falsification tolerance but below the >10% survival bar; the per-fetch fixed cost is small enough at F=2 that latency is nearly byte-linear (A8 nearly holds). The constancy arm alone does NOT falsify H2 at either F (no combination is within ±5%); the S arm carries the kill.

## 4. Regime check (A2, pre-registered Ambiguous-Outcome gate)

Compute estimate per decode step: HBM-bound time at 2 TB/s for weights (13.48 GB) + attention reads of the actual retained store (T_gpu = 6.79-7.81 ms across cells; weights dominate). Tier I/O: T_transfer = fetches x F + bytes/31.5 GB/s. FlexGen rule `T = max(I/O, compute)`; compute-dominant iff T_transfer <= T_gpu.

| Workload | F (µs) | all host cells compute-dominant? | tier fraction min-max | any cell >10%? |
|---|---|---|---|---|
| W1_RAG_4K | 2.0 | False | 21.3% - 78.0% | True |
| W1_RAG_4K | 10.55 | False | 26.0% - 79.3% | True |
| W2_GSM8K_900 | 2.0 | True | 12.6% - 46.7% | True |
| W2_GSM8K_900 | 10.55 | True | 15.6% - 48.9% | True |

Per-cell detail in `raw/e1_regime.csv` (T_gpu, T_transfer, max-rule winner, tier fraction per cell). Pre-registered untestable condition — `max(I/O, compute) = compute at EVERY cell` — holds for W2 at both F (all 6 host cells compute-dominant; the ~900-token decode-heavy reasoning workload never lets PCIe surface above compute, max transfer 6.1-6.7 ms vs T_gpu 6.98-6.79 ms). It does NOT hold for W1: the (100%,16-bit) and (60%,16-bit) host cells are I/O-dominant (T_transfer 16.7-30.0 ms > T_gpu 7.4-7.8 ms), tier fraction 44-79%. So the tier axis is testable in ≥1 W1 cell (criterion (iii) passes on W1), and W2 is reported as tier-untestable per workload.

## 5. Predicted peak-interaction cell (E3 look-first)

Cell with maximum byte-normalized tier latency (largest predicted deviation from the byte-linear additive baseline):

| Workload | F (µs) | peak cell | µs/MB | tier fraction |
|---|---|---|---|---|
| W1_RAG_4K | 2.0 | 100%/2bit/GPU+host | 34.5 | 47.5% |
| W1_RAG_4K | 10.55 | 100%/2bit/GPU+host | 46.5 | 55.0% |
| W2_GSM8K_900 | 2.0 | 60%/2bit/GPU+host | 34.7 | 18.0% |
| W2_GSM8K_900 | 10.55 | 60%/2bit/GPU+host | 47.4 | 23.1% |

The peak-interaction cells are the 2-bit cells at HIGH retention — (100%, 2-bit, GPU+host) on W1, (60%, 2-bit, GPU+host) on W2. The non-linearity H2 names (per-fetch fixed cost on chunk-granularity fetches) is real in the model, but it peaks where the byte store is LARGEST (more chunks per layer, more fixed cost per MB), not at the H2 corner (20%, 2-bit) — i.e., the mechanism anti-correlates with retention, so it cannot produce a >1.5x super-additive penalty at the joint minimum. (If E3 runs tier cells for other reasons, these are the cells where deviations from byte-linearity should appear first.)

## 6. Frozen verdict checks

| Workload | F (µs) | S | S (raw) | S > 1.15? | bn spread % | bn const ±5%? | bn var >10%? | regime all-compute? | regime pass | VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|
| W1_RAG_4K | 2.0 | 0.0 | 0.1549 | False | 6.6 | False | False | False | True | H2 ANALYTICALLY FALSIFIED |
| W1_RAG_4K | 10.55 | 0.0 | 0.1633 | False | 28.78 | False | True | False | True | H2 ANALYTICALLY FALSIFIED |
| W2_GSM8K_900 | 2.0 | 0.0 | 0.306 | False | 6.84 | False | False | True | False | TIER UNTESTABLE (Ambiguous Outcome) |
| W2_GSM8K_900 | 10.55 | 0.0 | 0.3116 | False | 29.28 | False | True | True | False | TIER UNTESTABLE (Ambiguous Outcome) |

**OVERALL VERDICT FLAG: H2 ANALYTICALLY FALSIFIED (E1 screening kill; W2 tier-untestable per workload)**

Basis: S <= 1.15 at the (20%, 2-bit) corner wherever the tier axis is testable — W1 at both F (corner pen = 0 under the FlexGen max rule; raw-transfer S = 0.16 W1 / 0.31 W2 at F=10.55). Byte-normalized latency is NOT constant (28.8-29.3% spread at F=10.55; 6.6-6.8% at F=2 — neither within the ±5% falsification tolerance), so the constancy arm does not fire; the S arm fires. W2 additionally fails the regime check (all cells compute-dominant -> tier untestable on W2, pre-registered Ambiguous Outcome — E2b/E3 tier arms canceled for that workload). Consequences per the kill-gate (LOCKED_PLAN §7): H2 is killed at screening cost under the corpus cost models — the burden of proof for a real-machine super-additive tier penalty shifts to E2b (root-complex contention, fragmentation, concurrency are NOT in these models, per T5). H1, the primary, and H3 are untouched by this verdict. E3's tier arm is warranted only if E2b resurrects the mechanism.

## 7. Constant-set citation log and substitution note

| Constant | Value | Source |
|---|---|---|
| layers / hidden | 32 / 4096 | experiment_plan.md §4 (Llama-2-7B MHA, 4K) |
| per-token per-layer FP16 KV | 16 KB (2 x 4096 x 2 B) | experiment_plan.md §4; KIVI.md §2 |
| weights bytes (GPU-side) | 13.48 GB (≈6.74e9 x 2 B) | Llama-2-7B param count, FP16 |
| KIVI G / R | 32 / 128 | experiment_plan.md §4 (frozen); KIVI.md §4.1 |
| KIVI-2bit store ≈ 21.7% of FP16 (anchor) | GEAR Table 1 | GEAR.md §10 / plan §E1 |
| GEAR n_b=20 FP16 buffer | not present in this matrix | GEAR.md §4 |
| chunk (block-aligned) | 512 tokens | experiment_plan.md §4/§E1 |
| page block | 16 tokens | experiment_plan.md §4 |
| PCIe nominal bandwidth | 31.5 GB/s | ShadowKV.md §9/§10; plan §E1 |
| HBM bandwidth | 2 TB/s | ShadowKV.md §9 |
| temporal locality α | 60% | ShadowKV.md §10 ('>60% hit rate'); plan §E1 |
| per-fetch fixed latency F | {2, 10.55} µs | Beluga.md §3 (10.55 µs per 16 KB, ~75% sync); assumption_map A5; plan §E1 |
| decode transfer fraction anchor | 96.9% of per-block time | InfiniGen.md Fig 18 (FlexGen baseline); plan §E1 |
| FlexGen latency rule | T = max(I/O, compute) | FlexGen.md §5 cost model; plan §E1 |
| W1 retained sets | 4096 / 2458 / 820 | plan §E1 (4096; 128/820 = 15.6% ⇒ 820; 2458 = ceil(0.6x4096)) |
| W2 retained sets | 900 / 540 / 180 | plan §E1 (GSM8K 8-shot CoT ~900-token prefill) |

**Substitution note (pre-registered, plan §E1 Hardware):** machine-specific constants (measured PCIe bandwidth of the actual A100 platform, real per-fetch latency) are NOT available — this machine is CPU-only (environment.md; no GPU on this host). All quantities above use the published constant set and are tagged accordingly. Ratio conclusions (S index, byte-normalized latency spread) are designed machine-invariant: raw-transfer S = 0.16-0.31 is ~4-7x below the 1.15 bar, and would need the corner's per-MB cost to be >4-6x the single-axis cells' to flip — no plausible PCIe-constant rescaling does that (raw S is exactly invariant to the shared PCIe bandwidth and to α). Before E2b/E3, measured constants replace the published values (sensitivity check; re-run logged).

## 8. Sensitivity notes

- **F sweep:** F ∈ {2, 10.55} µs changes T_transfer by ≤2.7 ms (W1 top cells) and shifts the byte-normalized spread from 6.6-6.8% (F=2) to 28.8-29.3% (F=10.55); it does not change the S verdict (corner pen = 0 at both).
- **Temporal locality α:** the plan fixes α = 0.60 (ShadowKV); transfer bytes = (1-α) x retained store. α = 1.0 (no locality, full re-fetch) would scale every T_transfer by 2.5x: W2 (100%,16-bit) becomes I/O-dominant, but the corner remains compute-dominant (T_transfer ≈ 4.9 ms vs T_gpu ≈ 6.8 ms at (20%,2-bit) W1) — raw S is exactly α-invariant (0.16 W1 / 0.31 W2). Verdict robust to α.
- **Compute estimate:** HBM-bound GPU time (weights + KV at 2 TB/s) is the plan-consistent estimate (constants list HBM 2 TB/s; FlexGen measures decode GPU utilization ~13%, i.e., bandwidth-bound). A FLOPs-only estimate (≈14 GFLOP/step → 45 µs at 312 TFLOPS) would make every cell I/O-dominant (regime always 'testable') but leaves raw S = 0.16-0.31 unchanged — the kill is not an artifact of the compute estimate.
- **Rounding:** 60% of 4096 = 2457.6 → 2458 (ceil); 20% → 820 (plan's own arithmetic). W2 = 900/540/180 exactly.

## 9. Artifacts

- `code/e1_screening.py` — SHA-256 `0f0c03a09181805022519a0b90cd43046179dd75322dd530fd30b7992d0e3070`
- `raw/e1_cells.csv` — 48 rows, every computed quantity (bytes per component, transfer, fetches, latencies, pen, µs/MB, tier fraction)
- `raw/e1_regime.csv` — regime quantities per cell
- `logs/experiment_log.md` — run record