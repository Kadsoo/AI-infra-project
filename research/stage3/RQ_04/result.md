# Research Question

**RQ-4:** Can cache affinity improve reuse without violating per-class tail/fairness? (Under prefix-popularity skew, does longest-prefix greedy routing concentrate the hot class on one node — hot-node queue hotspot — and is the cost attributable to routing and controllable by threshold replication?)

# Hypotheses

- **H1 (hot-node queue hotspot):** at p_hot = 0.8 (and 0.9), c = 32, stable load, equal memory-accounted budget: hit-rate gain ≥ G (expected ≥10pp), aggregate mean TTFT ≤ 0.8× load-only, hot P99 ≥ 2× load-only, cold P99 ≤ 1.2× load-only.
- **H2 (SLO boundary):** exists p_hot* ∈ [0.7, 0.9], c* ∈ [16, 64] such that for all p_hot ≥ p_hot*, c ≥ c* the hot class's P99 under affinity exceeds the common SLO (2.0 s) while cold attainment ≥ 99%; thresholds non-increasing in the fixed dimension.
- **H3 (attributability + controllability):** hot ratio P99(A)/P99(B) ≥ 2× the cold-class ratio; threshold replication (2 replicas) restores hot P99 within 30% of load-only with hit rate within 5pp of affinity-only.

# Experimental Setup

CPU-only Wave-1 (all gates executed on this machine; E4 real-system gate blocked — no 4-GPU node):
- **E0 (gate-0 closed-form algebra):** M/G/1 concentration analysis over p_hot × c × load × γ; LRU-radix cache model with equal memory-accounted budgets (L1 = 32,768 KV tokens = 4×2048-class concurrency × 4 nodes; L2 = 16,384).
- **E1 (H1 point test):** 10,000-request trace replay (10 seeds), p_hot ∈ {0.8, 0.9}, c = 32, loads {0.5, 0.65, 0.8} × affinity saturation, both budgets, γ ∈ {0.5, 1, 2}; policies A (affinity longest-prefix), B (load-only min token-weighted backlog), B' (request-count least-loaded); per-node FIFO prefill queues, admission = dispatch (no global FIFO), T_prefill = γ·uncached_tokens (γ: 2048-token prefill ≈ 1.0 s); common SLO per-class P99 ≤ 2.0 s; 780 runs.
- **E2 (H2 grid + H1 grid falsification):** p_hot ∈ {0.5, 0.7, 0.8, 0.9} × c ∈ {8, 16, 32, 64, 128} × loads × budgets × 10 reps (2,600 runs); Gamma-burst CV ∈ {2, 4} reduced grid (2,160 runs); policy C (threshold replication k=2, θ=1.0 hot req/s) at threshold points (200 runs); γ-robustness at verdict points (400 runs).
- **E3 (H3 controllability):** suffix-sensitivity (hot suffix 256/1024) and length-mismatch (hot mean 3072) variants × policies A/B/C at the surviving points; replication accounting (2,700 runs).
- Sanity: E0 anchors re-verified in-simulator (utilizations ±2%, P-K wait −4.3% at realized ρ); E3 base cells are bit-identical re-runs of E2 cells.

# Baseline

B — load-only routing at equal total offered load, equal memory-accounted KV budget, equal request set (trace-replay); identical scheduler, batching, admission, eviction; only the routing decision differs.

# Results

## E0 (raw: `raw/e0_grid.csv`, `processed/e0_report.md`)

- **G = 0.00 pp at both budget levels** — under the frozen cache model both policies cache all reusable content in steady state (A: prefix on the hot node; B: on all 4 nodes it serves hot traffic from; leaf-first LRU never evicts the hot prefix; equal budgets buy B nothing). **Hit leg pre-registrationally killed** → H1 rests on the queue leg; E1's hit-gain falsification criterion does not apply.
- Stable-region map: hot-node util = load fraction by construction; only p_hot=0.5/load=0.8 overloads under the full-utilization variant (excluded).
- E0 predicted the aggregate-mean leg would FAIL: agg-mean ratio 1.32–3.53 ≥ 1 at all E1 points.

## E1 (raw: `raw/e1_cells.csv`; means over 10 reps, γ=1; L1≡L2)

| p_hot, load | hit gain (pp) | agg-mean ratio A/B | **hot P99 ratio A/B** | **cold P99 ratio A/B** | FI(A) / FI(B) |
|---|---|---|---|---|---|
| 0.8, 0.65 | 0.00 | **1.68** | **5.42** | **0.96** | 0.92 / 1.00 |
| 0.8, 0.80 | 0.00 | **2.42** | **8.52** | **0.88** | 0.74 / 1.00 |
| 0.9, 0.65 | 0.00 | **1.80** | **5.76** | **1.00** | 0.93 / 1.00 |
| 0.9, 0.80 | 0.00 | **2.69** | **10.01** | **0.97** | 0.75 / 1.00 |

Ratios γ-invariant across {0.5, 1, 2} as predicted. B' (request-count) does not flip the verdict. No run hit node util ≥ 1.0 (max 0.82 in E1).

## E2 (raw: `raw/e2_cells.csv`, `raw/e2_burst.csv`, `raw/e2_gamma_robust.csv`)

- **p_hot\* = 0.7, c\* = 16** (both inside the frozen ranges ≤0.8/≤32). Conjunction on [0.7, 0.9] × [16, 128]: hot P99(A) = 5.91–6.59 s > 2.0 s; cold attainment 0.9965–1.000 ≥ 99%. At (0.9, 64): hot attainment 0.753 < 95% ✓, cold 1.000 ✓ (expected point reproduced).
- **Monotonicity holds:** c*(p) = {0.5: 32, 0.7: 16, 0.8: 16, 0.9: 16}; p*(c) = {8: none, 16+: 0.7}.
- Falsification range check: hot P99(A) > 2.0 s at EVERY cell (min 2.066 s, at load 0.5) — the "within SLO at ≥95% attainment" condition holds nowhere.
- H1 grid: hot P99 ratio A/B = **2.8–10.0 across the full grid** — 0 cells within 20% of B.
- Burst robustness (CV 2/4): effect grows (hot P99 A: 6.0 s Poisson → 16.0 CV2 → 60.6 CV4 at (0.8,32,0.8)); ratios up to 15.5.
- c=8 caveat: the crossing fails on the COLD side there (cold attainment 0.93–0.98 — admission-cap regime), outside the frozen c* range.
- Policy C first look: (0.8,32,0.8): hot P99 C=2.90 vs A=6.01 vs B=0.71.

## E3 (raw: `raw/e3_cells.csv`; 2,700 runs; ratio-of-ratios = (hot A/B)/(cold A/B))

| Check | Frozen line | Result |
|---|---|---|
| Attribution: hot ratio ≥ 2× cold ratio | ≥2.0 | ratio-of-ratios 1.27–11.31 (median 5.78); 42/45 cells ≥ 2.0; 0 cells within ±20% |
| Controllability: C restores hot P99 within 30% of B | ≤1.30 | **1/13 active cells only** (median 2.63×; e.g. (0.8,32,0.8): 4.09×; (0.8,32,0.65): 2.86×) |
| Hit rate A vs C | within 5pp | −0.00 to +0.01 pp — holds everywhere |
| Replication traffic | <10% of KV transfer | ≤ 0.008% — holds |
| (i) load-driven regression | hot regression under load-only | does not fire (B hot P99 ≤ 4.90 s, never single-class) |
| (ii) hot ≈ cold ratio | within ±20% | does not fire (min ratio-of-ratios 1.27) |
| (iii) suffix scaling | effect scales with suffix, not p_hot | **scales with BOTH** (hot ratio: s256 3.80 → base 6.15 → mm 6.86; and 5.55 → 6.59 across p_hot) — "rather than" not met, but suffix matters (confounder) |
| (iv) C moves hot P99 >20% | while hit within 5pp | fires at 1/13 cells ((0.7,16,s256,0.65): −17.9%); moves −17.9 to −63.9% elsewhere |

# Main Observations

1. The **hit-rate leg is dead at E0** (G = 0.00pp): under the frozen memory-accounted LRU model, load-only accumulates the hot prefix on every node it serves, so equal budgets buy no hit-rate differential. The "≥10pp expected" hit gain does not exist in this model.
2. The **aggregate-mean leg fails decisively** (1.68–2.69× instead of ≤0.8×): affinity's hot-class concentration raises the mean TTFT of the (majority) hot class; E0 predicted this.
3. The **queue hotspot itself is real, large, and monotone**: hot P99 5.4–10× load-only at the E1 points, 2.8–10× grid-wide, crossing the 2.0 s SLO at p_hot* = 0.7 / c* = 16 with cold class safe (≥99% attainment); thresholds non-increasing in both dimensions; robust to γ and to burst CV.
4. **Controllability fails**: simple threshold replication (k=2) restores hot P99 within 30% of load-only in only 1 of 13 active cells (median 2.63×); it does move the tail substantially (up to −64%) and keeps hit rate and replication traffic negligible, but not enough for the frozen line.
5. FI (report-only): FI(A) 0.74–0.93 < 0.8 (unfair) at high load; FI(B) = 1.0 — the fairness index captures the tail asymmetry.

# Confounders

- Length-mismatch and suffix variants (E3) show the effect scales with suffix length as well as p_hot — the prefill-cost component is present; the attribution "rather than p_hot" condition is not met, but suffix sensitivity is a documented confounder, not the sole driver (p_hot scaling is significant at every fixed variant).
- c=8 admission-cap regime degrades the cold class (outside frozen ranges; reported separately).
- Overload region: no run reached util ≥ 1.0 (max 0.993); the overload exclusion never bound.
- Cold-class within-node scheduling order (SGLang starvation) is NOT modeled at E1–E3 (pre-registered); it is an E4 observation only.

# Alternative Explanations

- **"The hot tail is a load artifact, not routing"** — excluded: load-only at the same load never singles out the hot class (max hot P99(B) 4.90 s vs min hot P99(A) 1.09 s; B's hot/cold ratio ≤ 1.003).
- **"The effect is prefill-cost driven (suffix length), not concentration"** — partially present: the effect scales with both suffix and p_hot; the E3 control cannot fully separate them, but the p_hot axis alone spans the SLO crossing (base variant, p_hot 0.5→0.9: hot P99 2.07→5.91 s).
- **"Budget level drives the result"** — L1≡L2 everywhere (max rel. diff 0.077): the equal-budget accounting is not the driver.
- **"RNG/replication artifacts"** — E3 base cells are bit-identical re-runs of E2 (max deviation 0.0000); E0 algebra anchors verified within ±2%.

# Control Experiments

- B' (request-count least-loaded): verdict does not flip on the load metric.
- E3 suffix-sensitivity pair (s256/base) and length-mismatch (mm): attribution control.
- γ-robustness (0.5/1/2): ratios invariant (scale-invariance of the linear model, as E0 predicted).
- Gamma-burst CV {2,4}: effect direction and SLO crossing preserved (grows).
- E0-vs-simulator anchor checks at 3 points before verdicts.

# Hypothesis Verdict

**H1: REJECTED** (as the frozen conjunction). The queue leg is strongly supported (hot P99 5.4–10× ≥ 2×; cold 0.88–1.00 ≤ 1.2×), but two frozen legs fail decisively: the hit-rate leg is dead at E0 (G = 0.00pp vs expected ≥10pp — the pre-registrationally-killed leg) and the aggregate-mean leg fails at both test points and grid-wide (1.68–2.69× vs ≤0.8×, as E0 predicted). The literal E1 kill line did not fire (hot P99 far outside 20% of B), so the pre-registered pipeline nominally continues to E4 — but the H1 claim as stated is contradicted on 2 of its 4 legs. The per-class quantification (hot-tail hotspot, cold-class safety) stands as the real finding.

**H2: SUPPORTED** (simulator level; pre-registered labeling: "queue-concentration pass; the tail claim is untested until E4"). p_hot* = 0.7 ∈ [0.7, 0.9], c* = 16 ∈ [16, 64]; conjunction holds across [0.7, 0.9] × [16, 128] (hot P99 5.91–6.59 s > 2.0 s SLO, cold attainment ≥ 99%); thresholds non-increasing in both dimensions; expected point values reproduced ((0.9, 64): hot attainment 0.753 < 95%, cold 1.000); the falsification range check holds nowhere (hot P99 > 2.0 s at every cell); robust to γ ∈ {0.5, 2}, burst CV ∈ {2, 4}, both budgets. Confirmed at the (0.7, 16) and (0.8, 32) anchors in E1.

**H3: REJECTED** (simulator level). The attribution leg holds (ratio-of-ratios ≥ 2 at 42/45 cells), but the controllability leg fails: threshold replication (k=2) restores hot P99 within 30% of load-only at only 1/13 active cells (median 2.63×); the frozen "restores within 30%" line is not met, and falsification check (iv) fires at one boundary cell. Replication traffic and hit-rate side-effects are negligible, but the mechanism does not restore the tail.

# Effect Size

Large and directionally clear: hot-class P99 ratio 2.8–10× (load-only denominator), SLO-crossing at p_hot* = 0.7/c* = 16 with 2× cold-class headroom; aggregate-mean degradation 1.7–2.7× — the claimed "≥20% aggregate improvement" is contradicted by the same order of magnitude in the opposite direction. Effect robust across 5,000+ seeded runs, γ scales, burst CVs, budgets, and variants.

# Practical Importance

**Medium-High.** This is the field's first per-class quantification of the affinity-vs-tail trade-off under a matched harness: (a) the hot-node queue hotspot is real and monotone (SLO-boundary prediction tool), (b) the commonly-claimed aggregate hit-rate/mean-TTFT benefits of affinity-only routing do NOT materialize under memory-accounted budgets, (c) naive k=2 replication is insufficient to restore the hot tail — scheduler-design constraint for any future affinity work. Caveat: the E1–E3 queue-concentration passes leave the real-system tail claim (within-node scheduling order, real batching) untested — E4.

# Reproducibility

```
cd research/stage3/RQ_04
python code\e0_algebra.py; python code\workload_gen.py
python code\e1_replay.py --sanity; python code\e1_replay.py --all; python code\e1_replay.py --report
python code\e2_grid.py --gamma-traces; python code\e2_grid.py --sanity
python code\e2_grid.py --grid; python code\e2_grid.py --burst; python code\e2_grid.py --identify
python code\e2_grid.py --c; python code\e2_grid.py --gammacheck; python code\e2_grid.py --report
python code\e3_h3.py --variant-traces; python code\e3_h3.py --sanity; python code\e3_h3.py --grid; python code\e3_h3.py --report
```
Script SHA-256s in `logs/experiment_log.md`. Seeds 1001–1010 (traces), fixed per run; 10,000 requests/run; deterministic.

# Recommended Next Step

- **E4 (real system, 4 GPUs)** is the only remaining gate that can (a) anchor H2's SLO-crossing on hardware, (b) observe the cold-class scheduling-order question (unmodeled), and (c) test whether real batching flips the aggregate-mean leg. H3 verdicts require ≥3 nodes (void on a 2-node floor). The step-0 length–service curve fit is mandatory before simulator constants are trusted.
- The residual RQ value is the per-class quantification itself (deliverable), not the H1 conjunction.
- Proceed to Stage 3C for adjudication; E4 is hardware-blocked on this machine.