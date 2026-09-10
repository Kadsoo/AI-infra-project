# Research Question

**RQ-1:** When does P/D disaggregation lose to colocated execution? (P/D vs colocated crossover under cross-node placement, ≤25 Gbps interconnect, mixed prompt lengths, bursty arrivals.)

# Hypotheses

- **H1 (transfer materiality):** ≥8k prompts, ≤25 Gbps cross-node, no NVLink co-location → P95 transfer fraction of E2E TTFT ≥ 30% (P50 ≥ 15%); NVLink keeps P95 ≤ 5%.
- **H2 (queue hotspot, tested at transfer ≈ 0 / NVLink):** CV 2–3 + cache-affine pairing → P99 queue fraction ≥ 20% and P99 TTFT ≥ 1.5× colocated; CV = 1 keeps P99 ≤ 1.1× colocated.
- **H3 (baseline overtake):** at joint condition (25 Gbps, CV ≥ 2, cache-affine), colocated ≥ 100% of P/D SLO goodput at equal GPU count, or P/D needs ≥ 1.25× GPUs for parity; benign cell reversed (P/D ≥ 1.3× colocated).

# Experimental Setup

Wave-1 CPU gates only (this machine has no A100 testbed, see `environment.md`):
- **E1 (analytical transfer-materiality model, 72-cell uncertainty sweep):** KV bytes 128 KiB/token; T_transfer = bytes/(25 Gbps × f), f ∈ {0.2,0.5,1.0} (corpus anchor f=0.24); NVLink leg min(bytes/600 GB/s, 8 ms); T_prefill = 2·N·tokens/(312 TFLOPS·0.5), scale ∈ {0.5,1,2}; M/D/1 queue, ρ ∈ {0.3,0.5,0.7}, P95 queue = 2.75× mean; transfer on TTFT critical path (zero overlap); fixed 5,000-request trace (seed 20260827, 70% short median 1018, 30% long mean 9262/P95 12399, output 200, 59.6% hot share, arrival CV 0.996/2.054/2.996).
- **E2 (calibrated discrete-event queue simulator):** 2 P/D pairs, NVLink transfer (lognormal mean 8 ms × {0.7,1,1.4}), prefill 0.1026 ms/token × {0.5,1,2}; hot prefix 6144 tok (59.6% of long class); cache-affine longest-prefix vs load-balanced routing; colocated baseline = 2 chunked-prefill instances (tau=1024, +25%, **equivalent prefix cache ON** per the frozen hit-rate guard); λ=3.0 req/s (probe-chosen matched-SLO rate); conditions A (CV1 affine), B (CV3 affine), C (CV3 loadbal), D (CV2 affine), E (CV3 affine, 2P); 9 scale combos × 5 seeds = 225 P/D + 45 colocated runs, plus a low-load diagnostic (λ ∈ {0.5,1.0}).
- **E3/E4/E5 (hardware):** NOT RUN — 3× A100 testbed absent (environment.md).

# Baseline

Colocated Sarathi-style chunked prefill (tau=1024, FCFS + JSQ), same 2 GPUs, same trace, same λ, same SLO (TTFT P95 ≤ 4.0 s, attainment ≥ 95%), equivalent prefix cache (hit-rate guard). Fairness: only phase split and placement differ.

# Results

## E1 (H1 arithmetic gate; raw: `raw/e1_cells.csv`, `raw/e1_verdict.json`)

| Metric (long-class, 25 Gbps) | Corpus-anchored point (f=0.24, ρ=0.5, scale 1×) | Sweep min | Sweep max |
|---|---|---|---|
| P95 transfer fraction of E2E TTFT | **41.8%** | 4.6% | 72.0% |
| P50 transfer fraction | **41.8%** | — | — |
| NVLink P95 fraction | **0.09%** | 0.05% | 0.27% |
| 25 Gbps-vs-NVLink P95 TTFT gap | **71.6%** | 4.8% | 256.4% |
| T_transfer / T_prefill at P95 length | 2.17 s / 1.27 s | — | — |

## E2 (H2 queue-hotspot gate; raw: `raw/e2_summary.csv`, `raw/e2_pd_runs.csv`, `raw/e2_lowload_diag.csv`)

λ = 3.0 req/s (probe: colocated attainment 100% at CV=1; matched-SLO point). Means over 5 seeds, transfer-scale 1.0:

| Cond | CV | Policy | P99 TTFT (s) | P99 queue fraction | **P99 ratio vs colocated** | colocated P99 (s) | max-pair share |
|---|---|---|---|---|---|---|---|
| A | 1 | affine | 1.42 | 0.88 | **0.67** | 2.12 | 0.63 |
| **B** | **3** | **affine** | **3.57** | **0.97** | **0.63** | 5.63 | 0.53 |
| C | 3 | loadbal | 4.89 | 0.98 | **0.87** | 5.63 | 0.57 |
| D | 2 | affine | 2.11 | 0.94 | **0.60** | 3.50 | 0.56 |
| E | 3 | affine, 2P | 1.45 | 0.89 | **0.26** | 5.63 | 0.52 |

All 225 P/D runs: P99 ratio vs colocated ∈ [0.07, 0.87] — never ≥ 1.0. Low-load diagnostic (λ ∈ {0.5,1.0}): ratios 0.67–0.82; P99 queue fraction at CV=3 ≥ 0.93 at ANY load (the frozen kill conjunct "qfrac < 15%" is structurally unreachable at CV ≥ 2).

# Main Observations

1. **H1 (E1):** At the corpus-anchored point the P95 transfer fraction is 41.8% (≥30% support line) with P50 41.8% (≥15%) and NVLink 0.09% (≤5%); the 25 Gbps-vs-NVLink P95 TTFT gap is 71.6% (≥25%). But the sweep flips across the plausible band (fraction 4.6–72%, gap 4.8–256%): when the prefill pipeline is loaded (ρ=0.7) or prefill is slow (2×), the queue dominates TTFT and the fraction drops below 30% even at the corpus anchor bandwidth. Pre-registered outcome: **"narrow band → E3 required"** — H1 is arithmetically reachable and supported at the anchored point, not killed.
2. **H2 (E2):** The predicted queue hotspot does NOT form. P/D P99 TTFT is *below* colocated at every cell (max ratio 0.87 at the load-balanced CV=3 cell). Concentration does occur (max-pair share up to 0.91 at CV=1 low load) but the concentrated pair's work share is small: hot-prefix requests are only ~18% of arrivals and their prefill is ~3× cheaper (cache skip), so no pair approaches saturation in the stable region.
3. The load-balanced discriminator (C) is *worse* than cache-affine (B) at CV=3 (0.87 vs 0.63): load-balancing sacrifices the cache benefit without queueing relief. Affinity at NVLink is a tail *asset* in this model, not a liability.
4. Doubling prefill servers (E) drops the ratio to 0.26 — consistent with no concentration-driven hotspot.
5. P99 queue fraction ≥ 20% holds (0.88–0.99) — but trivially: burst queueing dominates the P99 of TTFT in *both* systems at the matched-SLO load.

# Confounders

- Service-time asymmetry: colocated pays +25% chunking overhead by design (Sarathi tau=1024), which structurally favors P/D (~0.8× service). The equivalent-prefix-cache guard was applied to colocated to keep the comparison fair (hit-rate differential otherwise >10 points).
- Scale-2.0 cells: colocated baseline utilization > 1 at λ=3.0 (unstable region) — reported, excluded from verdicts.
- Hot-prefix share (18% of arrivals) is the frozen trace's own value; concentration strength is trace-driven.

# Alternative Explanations

- **"The hotspot is suppressed by the cache service benefit, not by the absence of the mechanism."** The cache-skip makes hot requests cheap, which is the plan's own mechanism (affinity routes to the cached prefix); a hotspot could still form if the concentrated pair's utilization approached 1. Measured hot-node utilization stayed ≤ 0.55 (B, scale 1.0) — the queue cannot diverge. The E sanity cell (2× prefill) confirms the effect scales with capacity, not with load total.
- **"Burst-load artifact":** excluded — C (load-balanced at CV=3) does not reproduce the ≥1.5× tail (0.87).
- **"Length distribution / chunking artifact":** ratios are stable across service scales 0.5–2 (0.31–0.87); direction unchanged.
- **Low-load diagnostic:** the ratio never exceeds 1 even at λ=0.5 where queues are short — the result is not load-driven.

# Control Experiments

- C (load-balanced at CV=3): discriminator — no burst-load artifact (excluded).
- E (N_P doubled): concentration-vs-load-total diagnostic — hotspot dissolves further (no concentration mechanism).
- Low-load diagnostic (λ ∈ {0.5, 1.0}): rules out load-level effects.
- Equivalent-prefix-cache in colocated baseline: rules out the hit-rate unfairness confound (frozen guard).

# Hypothesis Verdict

**H1: WEAK** — survives the E1 arithmetic gate with support at the corpus-anchored point (41.8%/41.8%/0.09%/71.6% gap); the sweep flips across the band (pre-registered "narrow band → E3 required"). Confirmation requires E3/E4 hardware (blocked). Not killed.

**H2: REJECTED** (at the E2 simulator gate). The ≥1.5× P99 tail failed at every one of 225 runs plus 30 diagnostic runs (observed ratio 0.07–0.87, direction reversed); the load-balanced discriminator did not reproduce the tail; the steady-state leg (≤1.1×) held; the P99 queue-fraction leg held trivially. The pre-registered kill conjunct "qfrac < 15%" did not literally fire (qfrac ≥ 0.93 at CV=3 at any load — structurally unreachable under the calibrated model), but the empirical content of the kill condition (no pairing concentration forms a hotspot) is confirmed; the ratio leg of the kill fired at every cell. E4 for H2 is not justified without a v2 plan that re-derives the kill condition.

**H3: INCONCLUSIVE** — not testable at this gate; E4/E5 require the 3× A100 testbed (blocked).

# Effect Size

H1 at the anchored point: transfer = 41.8% of E2E TTFT at P95 — large; but the robustness region is narrow (support requires realized bandwidth ≤ ~corpus anchor and queue/prefill not at maxima). H2: the predicted effect (≥1.5×) was absent; the observed effect is opposite (−13% to −37% P99 relative to colocated at the nominal cell).

# Practical Importance

**Medium.** The transfer-materiality claim remains plausible for 25 Gbps cross-node long-prompt serving (H1), which is a real operating constraint for disaggregation; but the queue-hotspot half of the P/D-vs-colocated story (H2) was not reproduced in the calibrated model. If H2's null survives hardware, the P/D-vs-colocated crossover reduces to a pure bandwidth question (H1).

# Reproducibility

```
cd research/stage3/RQ_01
python code\trace_gen.py --lam 3.0 --out raw\trace_5000.csv   # trace (seed 20260827)
python code\e1_cost_model.py                                   # E1: 72 cells + anchored point
python code\e2_run.py                                          # E2: probe -> 225+45 runs
python code\e2_diag_lowload.py                                 # low-load diagnostic
```
Script SHA-256s in `experiment_log.md`. Environment: `environment.md` (CPU-only; Windows 11; numpy 2.3.5/scipy 1.17.1/pandas 3.0.1).

# Recommended Next Step

- **Needs more evidence (hardware):** E3 transfer microbenchmark + E4 serving on the 3× A100 testbed are required to confirm H1 and to test H3. H1 is the only surviving mechanism; the E4 joint condition (25 Gbps, CV=3, cache-affine) is still worth measuring because H1's transfer cost may itself produce the crossover H3 describes, even without the H2 queue hotspot.
- **H2:** drop at the simulator gate unless a `LOCKED_PLAN_v2.md` re-derives the kill condition (the qfrac < 15% conjunct is unreachable at CV ≥ 2 under the calibrated model — same failure class the pre-freeze review fixed for E1).
- Proceed to Stage 3C for adjudication of the H1 hardware path.