# Stage 3B Summary — Cheap Validation & Microbenchmark (Wave 1)

> Date: 2026-08-27. Scope: Wave-1 CPU-only kill gates for all four Tier-1 RQs, per `stage3a_priority.md`, executed strictly against the frozen `LOCKED_PLAN.md` contracts. **No hypothesis, metric, baseline, or threshold was modified.** All GPU-gated cells (RQ-1 E3–E5; RQ-2 E2–E5; RQ-4 E4; RQ-8 E2a–E4) are BLOCKED on this machine (single 8 GB laptop GPU; no A100, no PyTorch, no serving framework; see `environment.md`). RQ-4's E2/E3 (CPU-only, hours) were additionally executed because E1 survived and they are the decisive remaining simulator gates.

## 1. Verdict table

| RQ | Hypothesis | Verdict | Effect Size | Confidence | Cost | Recommended Action |
|---|---|---|---|---|---|---|
| RQ-1 | H1 transfer materiality | **WEAK** (survives E1; anchored support 41.8% P95 fraction, 71.6% gap; sweep flips → "narrow band → E3") | Large at anchored point; narrow robustness region | Medium (model) | ~0 GPU-days | E3/E4 hardware to confirm |
| RQ-1 | H2 queue hotspot | **REJECTED** (E2: P99 ratio 0.07–0.87 in 270+30 runs; direction reversed; C discriminator negative; kill conjunct unreachable) | Predicted effect absent | High (sim, 5 seeds × 9 scales) | ~0 GPU-days | Drop unless `LOCKED_PLAN_v2` re-derives the kill condition |
| RQ-1 | H3 baseline overtake | INCONCLUSIVE (blocked) | — | — | — | needs 3× A100 testbed |
| RQ-2 | H1/H2/H3 | INCONCLUSIVE (all; E1 triage only) | n/a | High on triage | 0 GPU-days | A100 for E2/E5; falsification-first posture (K3 fired) |
| RQ-4 | H1 hotspot | **REJECTED** (conjunction: hit leg dead at E0 G=0.00pp; agg-mean leg 1.68–2.69× vs ≤0.8×; queue leg 5.4–10× ✓, cold ≤1.2× ✓) | Large | High (sim, 780 runs) | ~0 GPU-days | E4 anchors optional; per-class quantification stands |
| RQ-4 | H2 SLO boundary | **SUPPORTED** (sim; p_hot\*=0.7, c\*=16, monotone, hot P99 > 2.0 s everywhere, cold ≥99%; γ/burst-robust) | Large (5.9–6.6 s vs 2.0 s SLO) | High (sim, 5,000+ runs) | ~0 GPU-days | E4 (4 GPUs) to anchor the tail on hardware |
| RQ-4 | H3 controllability | **REJECTED** (attribution ✓ 42/45; restoration fails 1/13 cells, median 2.63× vs ≤1.3×) | Large gap vs line | High (sim, 2,700 runs) | ~0 GPU-days | E4 ≥3 nodes if the RQ continues |
| RQ-8 | H1 retention×bit-width quality | INCONCLUSIVE (blocked) | — | — | — | E2a (A100) — decisive next gate |
| RQ-8 | H2 tier super-additivity | **REJECTED** (analytical screening kill: S = 0.000 ≤ 1.15; raw S 0.16–0.31; W2 untestable) | Mechanism absent in model | Medium (model; T5 residual) | 0 GPU-days | E2b only if T5 resurrection is pursued |
| RQ-8 | H3 Pareto reversal | INCONCLUSIVE (not assessed at this stage, pre-registered) | — | — | — | requires E3/E4 |
| RQ-8 | Primary interaction | INCONCLUSIVE (blocked) | — | — | — | E4 only if E2a/E2b survive |

## 2. Classification

### Strong Signal
- **RQ-4 H2** — hot-class SLO-boundary claim fully confirmed at simulator level with monotone thresholds, reproduced expected point values, robust to γ/burst/budget. First per-class quantification of the affinity-vs-tail trade-off.
- **RQ-4 queue hotspot (H1's queue leg)** — 2.8–10× hot P99 ratios, grid-wide, γ-invariant (even though the full H1 conjunction is rejected).

### Weak Signal
- **RQ-1 H1** — transfer materiality is arithmetically reachable and supported at the corpus-anchored point (41.8% / 71.6% gap), but the robustness region is narrow (flips when queue/prefill dominate); hardware confirmation (E3/E4) required.

### Rejected
- **RQ-1 H2** (queue hotspot absent at simulator level; direction reversed).
- **RQ-4 H1** (as a conjunction: hit leg dead at E0; aggregate-mean leg fails 2×–3×; queue/cold legs hold).
- **RQ-4 H3** (replication restoration fails; the attribution leg holds).
- **RQ-8 H2** (tier super-additivity absent under the corpus cost models; screening kill with the pre-registered T5 caveat).

### Inconclusive
- **RQ-2 all** (E1 triage only; matched-harness verdicts need A100).
- **RQ-1 H3**, **RQ-8 H1 / primary / H3** (hardware-blocked or pre-registered not-assessed).

### Design Problems (recorded, not fixed — freeze rules)
- **RQ-1 E2 kill condition:** the frozen conjunct "P99 queue fraction < 15%" is structurally unreachable at CV ≥ 2 under the calibrated model (qfrac ≥ 0.93 at any load) — same failure class the pre-freeze review fixed for E1's kill lines. The ratio leg fired at every cell; the conjunct could not.
- **RQ-4 E0:** the frozen "hit leg killed if G < 10pp" rule fired immediately (G = 0.00pp) — the hypothesis's own cache model gives load-only parity on memory-accounted hit rate; the expected ≥10pp gain was unreachable by construction, leaving H1's conjunction short one leg before any run.
- **RQ-4 E3 check (iii):** the suffix-sensitivity control shows the effect scales with BOTH suffix length and p_hot — the "rather than" formulation is not cleanly decidable in this design (prefill-cost component present).
- **RQ-2 E1:** K3 fired (no native-safe AND ≥10%-crossing cell exists in the corpus) — the H1 matrix is a falsification-first verification set, not a discovery instrument.

## 3. Data integrity and reproducibility

- No runs deleted or hidden; every failed/buggy run's fix is recorded in the per-RQ `experiment_log.md` (RQ-1: four simulator bugs found and fixed during bring-up, full re-run after each; RQ-4: one simulator fix with E1 anchor re-verified bit-identical). All raw CSVs retained (`raw/` per RQ; RQ-4: ~7,500 per-request logs + 6,500+ aggregate runs; RQ-1: 300 runs + diagnostics; RQ-8: 48-row deterministic table; RQ-2: 52-cell auditable triage).
- Seeds, script SHA-256s, commands, and environment (`environment.md`) recorded per RQ. No git repo exists on this machine (noted as a reproducibility limitation).
- Sanity checks were run before every verdict-bearing sweep: E1/E2 (RQ-1) probe + monotonicity; RQ-4 E0-vs-simulator anchors (±2% util; −4.3% wait at realized ρ) and bit-identical E2/E3 overlap; RQ-8 deterministic model with cross-checked byte accounting against the published 21.7% anchor.

## 4. Kill-gate consequences (pipeline state)

| Gate | Outcome | Consequence |
|---|---|---|
| RQ-1 E1 | H1 survives (narrow band) | E3/E4 (hardware) still justified for H1 |
| RQ-1 E2 | H2 rejected | E4 for H2 not justified without a v2 plan; H3 (joint overtake) only informative via H1's transfer leg |
| RQ-2 E1 | triage done; K3 fired | E2 shrinks; E5 unconditional; A100 required |
| RQ-4 E0+E1+E2+E3 | H1/H3 rejected; H2 supported (sim) | E4 (4 GPUs) is optional-value (anchors + cold-class scheduling observation), not required by any surviving claim except H2's hardware anchor |
| RQ-8 E1 | H2 analytically falsified | E2a (quality) becomes the sole decisive cheap gate; if it kills H1 → overall null at microbenchmark cost |

## 5. Hardware-block register (Wave 2/3)

- RQ-1 E3/E4/E5: 3× A100 + 25 Gbps link + NVLink pair — absent.
- RQ-2 E2–E5: single A100-80GB — absent (8 GB laptop GPU cannot host FP16 7B weights, let alone LongBench full-KV baselines).
- RQ-4 E4: 4× GPU node (or 2-node floor) — absent.
- RQ-8 E2a/E2b/E3/E4: single A100-80GB + host-DRAM path — absent.

## 6. Recommended actions (for the Stage 3C Senior Reviewer's adjudication)

1. **RQ-4**: the per-class quantification is the strongest deliverable of this wave. Recommend: accept the H2 simulator support as a Stage-4-relevant result (with the "tail claim untested until E4" label); treat H1/H3 as rejected at simulator level; schedule E4 (4 GPUs) only if the H2 hardware anchor is considered essential.
2. **RQ-1**: keep only the H1 transfer-materiality thread (E3/E4 on the 3× A100 testbed); H2's queue-hotspot thread is dead at the simulator and should not be re-funded without a v2 kill condition.
3. **RQ-8**: E2a (quality matrix, 1 A100) is the single highest-value remaining GPU gate in the program — it completes the cheap-null path (primary + H1 + H2) if H1 dies, reclassifying RQ-8 as engineering composition; E2b is secondary (T5 resurrection only).
4. **RQ-2**: lowest marginal value per GPU-day at this point *unless* the falsification-first posture is accepted as the goal (E2 wave 1 + E5 open cells ≈ 1–2 GPU-days on one A100); the E1 triage already prunes ~half a day of E4 sweeps.
5. **Design fixes to carry forward** (if any RQ returns to Stage 3A): RQ-1 E2's qfrac<15% conjunct; RQ-4's hit-leg pre-registration interaction with E1's exemption clause; RQ-4 E3's suffix/p_hot decoupling.

**This summary is the experimental record, not the final decision.** The choice of which RQs proceed to Stage 4 is adjudicated by the independent Senior Reviewer in Stage 3C.