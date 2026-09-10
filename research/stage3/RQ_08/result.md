# Research Question

**RQ-8:** Do retained-token count, KV bit-width, and storage tier interact non-additively? (Under a matched single-7B serving harness, do the two-way and three-way factorial interactions of retention × bit-width × tier explain a material share of the main-effect variance, or do the axes compose additively?)

# Hypotheses

- **H1 (retention × bit-width, quality):** the 2-bit quality drop is ≥2× larger at 20% retention than at 60%, and the R×B interaction is significant (α=0.05) with effect ≥50% of the bit-width main effect, in ≥1 workload.
- **H2 ((retention, bit-width) × tier):** the host-tier TTFT/TPOT penalty at (20%, 2-bit) exceeds 1.5× the sum of the two single-axis penalties, with byte-normalized transfer latency non-constant across cells.
- **H3 (Pareto reversal):** the axis-best cell (20%, 2-bit, GPU-only) is dominated by ≥1 other cell with ≥10% improvement in ≥1 primary metric and ≤5% regression in every other, replicated across ≥3 seeds and both workloads.
- **Primary:** ≥1 significant interaction term (α=0.05, BH) with R²_int_rel = SS_interactions/SS_main_effects > 0.50 in ≥1 metric & ≥1 workload.

# Experimental Setup

Wave-1 executed **E1 only** — analytical/numerical screening of the tier axis from the corpus cost models (CPU-only, no GPU, no serving harness). E2a (quality 3×2 matrix, H1), E2b (kernel/transfer microbenchmark, H2 empirical gate), E3/E4 (serving matrix) all require an A100-80GB (and for E2b/E3 the real host-DRAM path) — **blocked on this machine** (no GPU, no PyTorch; `environment.md`).

E1 model: Llama-2-7B (MHA, 4K); per-token per-layer FP16 KV = 16 KB; KIVI 2-bit G=32/R=128 decomposition (2048 B data + 1024 B scales/ZP per token + 2.097 MB/layer FP16 residual — reproduced the published 21.7%-of-FP16 anchor within 0.4 pp); 12 tier cells × 2 workloads (W1 RAG 4K, W2 GSM8K 900-token CoT) × F ∈ {2, 10.55} µs; chunk 512 tokens; PCIe 31.5 GB/s (ShadowKV); HBM 2 TB/s; α = 60% temporal locality; FlexGen rule T = max(I/O, compute); additive null = byte-linear penalty with zero per-fetch cost.

# Baseline

Additive-composition null: tier penalty linear in ACTUAL bytes with zero per-fetch overhead; every mechanism H2 names (per-fetch latency, chunk-granularity fetch count, residual fraction, dequant cost) is a deviation from it.

# Results (E1; raw: `raw/e1_cells.csv`, `raw/e1_regime.csv`; full 48-row table in `processed/e1_report.md`)

## Per-cell tier economics (W1 @ F=10.55 µs; select cells)

| R / B / T | store/layer (MB) | transfer/step (MB) | fetches/step | T_transfer (ms) | T_gpu (ms) | µs/MB |
|---|---|---|---|---|---|---|
| 100% / 16-bit / GPU+host | 67.1 | 859 | 256 | 29.97 | 7.81 | 34.9 |
| 100% / 2-bit / GPU+host | 14.3 | 183 | 256 | 8.51 | 6.97 | **46.5** |
| 20% / 16-bit / GPU+host | 13.4 | 172 | 64 | 6.13 | 6.95 | 35.7 |
| 20% / 2-bit / GPU+host (corner) | 4.2 | 54 | 64 | 2.39 | 6.81 | 44.2 |
| 20% / 2-bit / GPU-only (axis-best) | 4.2 | — | — | — | 6.81 | — |

Residual accounting: the FP16 residual R=128 is retention-dependent as a fraction (3.1% of store at 100% vs 15.6% at 20% on W1; up to 71.1% at 20% on W2 — at W2/20%/2-bit the store is 76.5% FP16: compression nearly eaten at low retention on short contexts).

## Frozen checks

| Check | Frozen line | Result |
|---|---|---|
| Super-additivity S = pen(20%,2-bit)/[pen(20%,16-bit)+pen(100%,2-bit)] | H2 null: ≤ 1.15; kill: S ≤ 1.15 | **S = 0.000 at all 4 workload×F combos** (corner tier penalty = 0 — transfer 1.8–2.4 ms < compute 6.8 ms → fully overlapped); raw-transfer robustness reading S = 0.155–0.312 (4–7× below the bar) |
| Byte-normalized latency constancy (A8) | falsify if within ±5% | 6.6–6.8% (F=2), 28.8–29.3% (F=10.55) — never within ±5%; constancy arm does NOT fire |
| Regime check (A2) | untestable iff compute-dominant at every cell | **W1 testable** (tier fraction up to 79% at 100%/16-bit); **W2 untestable** (all cells compute-dominant; pre-registered Ambiguous Outcome) |
| Predicted peak-interaction cell | E3 look-first | **(100%, 2-bit, GPU+host) on W1**; (60%, 2-bit, GPU+host) on W2 — the per-fetch non-linearity peaks at HIGH retention (more chunks), anti-correlated with the H2 corner |

# Main Observations

1. The tier penalty at the H2 corner (20%, 2-bit) is **zero under the corpus cost model**: at 4.2 MB/layer, even the worst-case transfer (2.39 ms at F=10.55 µs) hides under the 6.8 ms compute step. H2's super-additive mechanism (small transfers + per-fetch overhead + residual buffers) cannot reach a 1.5× penalty when the corner's absolute transfer volume is so small that PCIe never surfaces above compute.
2. The non-linearity H2 names IS present in the model (byte-normalized latency 34.9 → 46.5 µs/MB, spread 28.8% at F=10.55), but it **anti-correlates with retention**: it peaks at (100%, 2-bit) where the byte store (and hence fetch count) is largest — the opposite corner from the hypothesis's joint minimum.
3. The residual window R=128 makes 2-bit storage retention-inefficient at low retention on short contexts (W2/20%: 76.5% of the "2-bit" store is FP16 residual) — a structural byte-accounting effect, not a latency interaction.
4. W2 (GSM8K, ~900-token) is decode-compute-dominant everywhere: the tier axis is not testable on the reasoning workload under the corpus model (pre-registered Ambiguous Outcome).

# Confounders

- Machine-specific constants (real PCIe bandwidth, real per-fetch latency) are unavailable — published constants used; ratio conclusions are machine-invariant by design (raw S is exactly invariant to shared PCIe bandwidth and α; verified).
- The compute estimate choice (HBM-bound vs FLOPs-bound) changes the regime reading but not raw S (0.155–0.312).
- F-sweep: the constancy arm's outcome flips with F (6.6% vs 28.8%) — the per-fetch fixed latency is the entire non-linearity; if the real machine's F is small, A8 nearly holds.
- Root-complex contention, fragmentation, and concurrency (T5) are NOT in the corpus models — E2b is the empirical gate, still required before any hard "H2 dead" claim on hardware.

# Alternative Explanations

- **"The corner penalty is hidden by compute overlap, not absent"**: true by construction of the FlexGen max rule — but the hypothesis's own mechanism list (transfer bytes, fetch count, residual fraction) is what produces the overlap; the serving-level penalty is what H2 claims, and it is 0 in the model. The raw-transfer robustness reading (no overlap) still yields S ≈ 0.16–0.31, far below 1.15.
- **"The constancy arm should have fired"**: it does not (spread 6.6–29.3% vs ±5% bar); H2's kill rests on the S arm alone, which is the stronger arm (both arms are pre-registered as sufficient).

# Control Experiments

- Additive null baseline (byte-linear, zero per-fetch cost): every deviation measured against it.
- F-sweep (2 vs 10.55 µs) and α-sensitivity (raw S exactly invariant): verdict robust.
- W1 vs W2 regime contrast: the tier axis testability is workload-dependent, as pre-registered.

# Hypothesis Verdict

**H1: INCONCLUSIVE** — untestable at this gate. The R×B quality interaction requires E2a (A100; quality is layout-independent, GPU-only). No quality data exists on this machine.

**H2: REJECTED (analytical screening kill, pre-registered).** Under the corpus cost models, S = 0.000 ≤ 1.15 at the (20%, 2-bit) corner wherever the tier axis is testable (W1, both F); the raw-transfer robustness reading (0.155–0.312) is 4–7× below the bar; byte-normalized latency is not constant (so the constancy arm doesn't fire — the S arm carries the kill). The mechanism list in H2 cannot produce the >1.5× super-additive penalty under the field's own cost structure. **Pre-registered caveat:** this is a screening kill — E2b (real-machine microbenchmark) is the only remaining empirical gate for H2 (root-complex contention, fragmentation, concurrency are unmodeled); E2b is GPU-blocked here. W2 is tier-untestable (Ambiguous Outcome).

**H3: INCONCLUSIVE** — explicitly NOT assessed at this stage (pre-registered: the cheap-null path does not evaluate H3; a Pareto reversal can arise from main effects alone, and falsification needs measured quality × latency × HBM coordinates from E3/E4).

**Primary: INCONCLUSIVE** — the full 24-cell matrix (E4) is blocked; E1 cannot touch it (measurement claim).

# Effect Size

The tier axis's *non-linearity* is small and misplaced: at the corner the per-byte cost premium over the additive null is ≤ 0.31× the single-axis sum (raw), i.e. the corner is *cheaper* per byte, not dearer; the largest deviation from byte-linearity (28.8% spread at F=10.55 µs) sits at (100%, 2-bit). No mechanism-level interaction survives the screening.

# Practical Importance

**Medium.** The negative screening result reclassifies the tier story as engineering composition (bytes × bandwidth) rather than mechanism interaction — for the RAG workload at these retention/bit-width points. It materially de-risks E3/E4: a serving-tier interaction, if any, must come from effects the cost models cannot see (root-complex contention, fragmentation, concurrency). The residual-window byte-accounting finding (76.5% FP16 at W2/20%) is a practical observation for anyone budgeting 2-bit KV at low retention.

# Reproducibility

```
cd research/stage3/RQ_08
python code\e1_screening.py      # deterministic, stdlib-only; 48-row table
```
Script SHA-256 `0f0c03a09181805022519a0b90cd43046179dd75322dd530fd30b7992d0e3070`; constant-set citation log in `processed/e1_report.md` §7; machine-constant substitution note pre-registered (re-run logged as a sensitivity check when the A100 machine's measured constants replace the published set).

# Recommended Next Step

- **E2a (quality matrix, H1) is the decisive next gate** — needs one A100. If E2a also kills H1 (R×B interaction non-significant AND <10% of the bit-width main effect in both workloads), the cheap-null path completes: primary + H1 + H2 null at microbenchmark cost → RQ-8 reclassified as engineering composition, E3/E4 canceled (H3 still not assessed).
- E2b (microbenchmark) is the only way H2 can be resurrected after the screening kill (T5 effects); run it on the same A100 (host-DRAM path required).
- Proceed to Stage 3C for adjudication; hardware is the blocker.