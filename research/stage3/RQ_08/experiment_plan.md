# Experiment Plan — RQ-8: Do retained-token count, KV bit-width, and storage tier interact non-additively?

**Stage 3A — Minimum-cost, kill-early experimental design.**
Governing contract: `research/stage3/RQ_08/hypothesis.md` (locked). The 3×2×2×2 = 24-cell factorial is the eventual **confirmation** design; it is not the cheapest first step. This plan re-orders the hypothesis's design into a staged kill-early pipeline (E1→E2a/E2b→E3→E4) in strict cost priority (offline analysis → simulation → instrumentation → microbenchmark → harness → minimal modification → complex modification). No experiment below proposes any algorithm, controller, or joint optimizer: all three axes use only existing mechanisms (H2O retention, KIVI 2-bit scheme, host-DRAM tier), per the Stage 2B "do not promote" list.

---

## 1. Pipeline overview (kill-early structure)

```
E1  analytical/numerical screening (corpus cost models)        [minutes, no GPU]
 |-- KILLS H2 analytically if byte-normalized latency constant under model (A8 holds)
 |-- KILLS the tier program if A2 regime check shows compute-dominance in every cell
 |-- CANNOT kill: H1, primary, H3, and H2's empirical truth (T5 effects unmodeled)
 v
E2a quality microbenchmark: 3×2 quality cells (R×B), 2 workloads, GPU-only  [hours]
 |   (existing KIVI/GEAR code, NO serving framework)  -> KILLS H1 if interaction
 |   non-significant AND <10% of bit-width main effect in BOTH workloads
E2b kernel/transfer microbenchmark (dequant, small-transfer efficiency,        [hours]
 |   PCIe utilization at 2 sizes)  -> KILLS H2 empirically if byte-normalized
 |   transfer latency constant on the real machine; else supplies constants
 v  (E3 runs only if E1/E2b leave H2 alive, OR E2a leaves H1 alive, OR E2b
 |   shows nonlinear transfer; if ALL dead => overall null at microbenchmark cost)
E3  2×2×2 tier subset (R∈{20%,100%} × B∈{2,16} × T∈{GPU-only,GPU+host})     [~1–2 days]
 |   on ONE workload, vLLM-class paged-attention harness; estimates interaction
 |   directions; includes emulator calibration anchor (≤15% TTFT mismatch)
 v  (E4 runs only if E3 shows surviving interaction directions)
E4  full 3×2×2×2 = 24 cells, both workloads, + max-batch capacity sweep,      [multi-day]
    + Pareto analysis for H3 (3-seed replication restricted to dominance pairs)
```

Hard gating rule (from hypothesis kill conditions, restated as gates): E2b runs only if E1 did not kill H2; E3 runs only if at least one of (H1 alive after E2a, H2 alive after E1+E2b) holds; E4 runs only if E3 shows interaction directions that survive. If E1 kills H2 **and** E2a kills H1, the cheap-null path declares **primary + H1 + H2 null** from E1+E2 alone (the RQ's interaction surface is empty at microbenchmark cost) — **H3 is explicitly NOT assessed at this stage** (revision R3: a Pareto reversal can arise from main effects alone, and H3's falsification requires measured quality × latency × HBM coordinates that only E3/E4 can produce). E3/E4 then never run.

---

## 2. Pre-registered statistics (global, applied to every experiment below)

- **α = 0.05** per test. **Correction:** Benjamini–Hochberg (BH) across **metric × workload families** (revision R3; previously only metric families): primary: {TTFT, TPOT, throughput, HBM, quality} × {W1, W2}; H1: quality × {W1, W2}; H2: {TTFT, TPOT} separately; H3: none — dominance is a set-membership test, not a multiplicity test.
- **Variance-explained definitions** (UNIFIED denominator per revision R2; Type III sums of squares from the factorial ANOVA computed on **cell means**, not per-request data — per-request within-cell variance is not part of the interaction calculus):
  - `R²_int_rel = SS_interactions / SS_main_effects` — the SINGLE quantity used for both support ("R²_int_rel > 0.50") and falsification ("R²_int_rel ≤ 0.10"). The old absolute denominator (SS_total, per-request) is dropped: retention's main effects dominate per-request variance by construction, which made the support bar unreachable and the falsification bar near-automatic.
  - H1's effect-size ratio ("effect ≥50% of the bit-width main effect" / "<10%") = estimated interaction contrast magnitude vs the bit main-effect contrast, as before.
  - Interaction terms included: all two-way (R×B, R×T, B×T) and the three-way (R×B×T) for the primary; R×B only for H1 (quality regression, no tier factor); tier-arm contrasts for H2.
  - **Gray zone (pre-registered):** a significant interaction with R²_int_rel ∈ [0.10, 0.50] is INCONCLUSIVE for the primary — requires the E4 full matrix before any verdict.
  - **Coding (revision R3):** R is coded with categorical (dummy/orthogonal-polynomial) contrasts in BOTH E2a and E4, identical, with a reported robustness check against linear coding. Linear coding cannot separate a nonlinear R main effect from an R×B interaction at 3 levels; the categorical coding fixes the semantics.
- **Power / N per cell:** pilot of 16–32 requests at the anchor cell and at the corner cell (20%, 2-bit, GPU+host for tier experiments; (20%,2-bit) for quality) per workload → per-metric σ on **cell means** (pilot N ≥ 4 cells per condition for a means-level σ). `N = max(128, ceil(2·(z_0.975 + z_0.80)²·σ²/δ²))`, where δ = smallest interaction of interest = 20% of the smallest pilot main effect, floored at 10% of the anchor-cell mean (80% power). N is capped at 512 (interactions below that resolution are declared unmeasurable, per hypothesis "Ambiguous Outcome — Statistical power"). Cells with effective N < 32 after failures are excluded, not interpolated. For binary quality (GSM8K per-request exact match), σ² = p(1−p); achieved power at N=128 is reported alongside effect-size confidence intervals — H1's criteria are effect-magnitude-dominated, so significance alone cannot carry H1. CIs on R²_int_rel are reported (bootstrap over cell means).
- **Saturated-cell exclusion (confounder 6):** any quality cell whose metric shows ≤2% variance across cells (e.g., saturated retrieval tasks) is excluded from quality analysis, pre-registered.
- **Seeds:** fixed per cell (list logged); H3 replication uses ≥3 seeds on the dominance-pair cells only (deviation D4, §3).

---

## 3. Deviations from the hypothesis's 24-cell design — justifications

- **D1 — E1 runs before any GPU spend.** The hypothesis's full matrix is the confirmation instrument. Interaction claims on the tier axis rest on a stated mechanism list (small transfers, residual buffers, dequant/rebuild, fetch count) that is fully expressible in the corpus's own closed-form cost models (FlexGen `T = max(I/O, compute)`; ShadowKV 31.5 GB/s PCIe + equivalent-bandwidth formula; KIVI residual R=128; GEAR buffer n_b=20). If those models predict additive tier behavior, H2's mechanism has zero quantitative support at zero hardware cost; the empirical tier experiment is then justified only if a microbenchmark (E2b) resurrects it (T5: real root-complex contention is not in the model — hence the screening kill is conditional, never a proof of A8 on hardware).
- **D2 — E2a tests H1 without the tier factor and without a serving framework.** H1 is a *quality* claim. Tier placement and paged attention change memory layout, not numerics; quality is a per-sequence function of the KV representation. Running the tier axis or the serving harness for H1 would add cost with zero discriminative power. KIVI's own codebase (HuggingFace-based) plus an H2O accumulated-attention retention wrapper is the cheapest sufficient instrument, and it matches how the corpus itself reports quality (KIVI/GEAR report LongBench/GSM8K from HF-style harnesses, batch-1 greedy).
- **D3 — E3 is a 2×2×2 subset on one workload, not 24 cells.** (a) The 60% retention level's only hypothesis role is H1's ratio test (20% vs 60%), which E2a covers at full 3×2; for estimating *interaction directions on the tier axis*, the extremes (20%, 100%) are the informative contrast — H2's corner is (20%, 2-bit) and the additive null is checkable at 20/100. Residual risk: 60% exhibiting non-additivity while 20/100 are additive (curvature without endpoints) is accepted and pre-registered as a known E3 blind spot, closed by E4 if E3 survives. (b) One workload: E3 estimates directions on the workload most prone to the failure scenario (pre-registered selection rule, §E3); if the interaction-prone workload shows none, the probability that the other workload produces interactions in the full matrix is negligible given the mechanism analysis — the full matrix would only confirm a null. The hypothesis's both-workloads falsification requirement is preserved by a pre-registered contingency: if E3 is null on W1, the same 8 cells are repeated on W2 (16 cells total, still 2/3 cheaper than 24) before the primary null is declared.
- **D4 — E4's 3-seed replication is restricted to H3's dominance-pair cells.** H3 is a pairwise dominance claim ("some other cell dominates the axis-best cell"). Replicating all 24 cells × 3 seeds × 2 workloads (144 runs) would test far more than H3 needs. Primary interaction estimates come from the single-seed matrix with N ≥ 128/cell (the hypothesis's control section fixes seeds per cell; 3-seed replication is named only in H3's falsification clause, which is about the dominance pair). Every cell participating in a dominance comparison (axis-best vs all 23 candidates) gets 3 seeds.
- **D5 — E2b (microbenchmark, cost tier 4) replaces the serving-level tier experiment as H2's empirical kill point when byte-normalized transfer latency is constant on the real machine.** The hypothesis's H2 falsification has two arms — the super-additivity bound AND "byte-normalized transfer latency is constant across all cells (A8 holds)". The second arm is measurable with a kernel/transfer microbenchmark; a serving experiment is only warranted if the microbenchmark shows the nonlinearity the serving-level test would be built to find.
- **D6 — E4's max-batch capacity sweep is deferred out of E3.** Confounder 3 is neutralized in E3 by fixed concurrency (batch 32); the capacity sweep (where memory savings surface) is folded into E4, which runs only if E3 survives. This does not weaken any interaction estimate (fixed concurrency is the controlled factor) and cuts E3 cost.

---

## 4. Shared control constants (from hypothesis.md, copied, fixed across E2a–E4)

Model: Llama-2-7B (MHA, 4K context), weights FP16, no weight quantization. GPU: one A100 80GB, clocks locked, fixed warmup. 2-bit = existing KIVI scheme (key per-channel, value per-token, group G=32, residual R=128), hyperparameters fixed in every cell. Retention = H2O-style accumulated-attention budget policy (R ∈ {100%, 60%, 20%}). Tier = host DRAM via PCIe, block-aligned 512-token chunks (page block = 16 tokens). Quality tolerance pre-registered: ≤3% relative drop vs (100%, 16-bit, GPU-only). Output cap 256 tokens. Serving cells: fixed max batch 32, continuous batching, saturated arrival. Byte accounting rule (confounder 2): actual bytes measured/accounted (quantized + scales/zero-points + FP16 residual R=128 + GEAR n_b=20 buffer + sparse/low-rank components where present) — never nominal bits.

---

# Experiment E1

# Experiment ID
**E1 — Analytical/numerical screening of the tier axis from corpus cost models.**

# Target Hypothesis
Primary (screening contribution), **H2** (primary kill candidate), plus the hypothesis's pre-registered A2 regime check (GPU-bound regime → tier hypotheses untestable).

# Purpose
Compute, per cell, the predicted tier-transfer volume, fetch count, and byte-normalized tier latency from the corpus's own cost models, and test the two quantities H2 hinges on — (i) the super-additivity index at the joint minimum, and (ii) constancy of byte-normalized latency (A8). Also detect analytically whether prefill/decode compute dominates every cell (regime check) before any GPU time is spent. **This experiment can KILL H2** — conditionally — if, under the corpus models, byte-normalized latency is constant across cells (A8 holds), because then tier effects are trivially additive and the >1.5× super-additive penalty is unreachable by the mechanism list in H2 (small transfers, residual buffers, dequant/rebuild, fetch count). **This experiment CANNOT kill:** H1 (the corpus contains no quality model — quality requires real generations), the primary hypothesis (it is a *measurement* claim on real hardware; E1 is model-based and inherits the corpus's own assumptions), H3 (needs measured quality × latency × HBM coordinates and a real frontier), and H2's *empirical truth* (T5: root-complex contention, fragmentation, and concurrency effects are not in FlexGen's/ShadowKV's models — E2b is the empirical gate). E1's kill is a *screening* kill: it means H2's mechanism cannot produce the effect under the field's own cost structure, shifting the burden of proof to E2b before any serving-tier experiment is justified.

# System
Trace analysis / analytical (offline script, e.g., numpy, CPU only). Uses only published constants and closed-form models: FlexGen `T = max(I/O, compute)` (per-layer, per-phase); ShadowKV PCIe 31.5 GB/s nominal and equivalent-bandwidth formula `B̃ = 2·S·B_GPU / (S/C + 2(K+O)·C + (1−α)·K·C·B_GPU/B_PCIe)` (with chunk C, fetch budget K, outliers O, temporal hit rate α = 60%); KIVI residual R=128 FP16 tokens; GEAR n_b=20 FP16 buffer, s=2% sparse (3× per non-zero), rank-4 low-rank (2-bit GEAR total ≈ 27.6% of FP16; KIVI backbone ≈ 21.7% per GEAR Table 1); per-fetch fixed latency constants (2–10.55 µs, from Beluga's measured 10.55 µs per 16 KB transfer with ~75% sync overhead, assumption_map A5) and decode transfer fraction 96.9% of per-block time in the FlexGen baseline (InfiniGen Fig 18). **Cheapest sufficient:** no GPU, no serving framework, no instrumentation — the hypothesis's entire H2 mechanism list is parameterized by these constants. This experiment is the only one that does not need the serving framework *and* does not need a GPU.

# Workload
None — a parameterized model of the two workload levels: RAG = 4K-class contexts (retained set up to 4096 tokens), reasoning = GSM8K 8-shot CoT (~900-token prefill). Request count: n/a (deterministic model; no stochasticity). Arrival pattern: n/a. Reuse characteristics: **explicitly n/a** — this is not a serving test; per-request independence is assumed (A3), no prefix reuse modeled.

# Hardware
None required for execution (published constants). Outputs are tagged with the constant set used and a substitution note: before E2b/E3, machine-specific constants (measured PCIe nominal bandwidth of the actual A100 platform, per-fetch latency) replace the published values. All *ratio* conclusions (constancy of byte-normalized latency, super-additivity index) are designed to be machine-invariant.

# Instrumentation Required
None measured. Outputs are logged tables: per cell (12 tier cells: 3R × 2B × 2T) — (a) actual retained bytes per sequence per layer (quantized + scales/zero-points + FP16 residual + buffer components, never nominal bits), (b) predicted transfer bytes per decode step, (c) fetch count per step = layers × ceil(retained_tokens/chunk=512), (d) predicted tier latency per step = fetch_count × (fixed + bytes/fetch_count/B_PCIe), compared against compute estimate (GPU-bound regime flag), (e) byte-normalized latency (µs/MB) per cell, (f) super-additivity index `S = pen(20%,2-bit)/[pen(20%,16-bit)+pen(100%,2-bit)]` with `pen(R,B) = T_tier(R,B) − T_gpu(R,B)`, (g) predicted peak-interaction cell. Script, inputs (constant list with paper citations), and version hash logged.

# Baseline
Additive-composition null inside the model: tier penalty linear in *actual* bytes with zero per-fetch overhead (the strongest additive form). Fairness: the null is constructed so that every mechanism H2 names (fixed per-fetch latency, chunk-granularity fetch count, residual FP16 fraction, dequant cost) is added as a *deviation* from it — the screening asks whether the deviations, at published magnitudes, can reach the 1.5× threshold.

# Experimental Conditions
Cells: R ∈ {100%, 60%, 20%} × B ∈ {16-bit, 2-bit KIVI G=32 R=128} × T ∈ {GPU-only, GPU+host}, evaluated under the model with fixed hyperparameters (copied from hypothesis.md): chunk = 512 tokens (block-aligned), temporal locality α = 60% (ShadowKV), per-fetch fixed latency F ∈ {2, 10.55} µs sensitivity sweep, PCIe = 31.5 GB/s, HBM = 2 TB/s, decode transfer fraction anchor 96.9% (FlexGen baseline). Residual accounting: KIVI R=128 FP16 tokens are a *constant* absolute size but a retention-dependent fraction of the retained store (128/4096 = 3.1% at 100% vs 128/820 = 15.6% at 20% retention) — the screening must show this explicitly. GEAR-style repair buffers (n_b=20) and the dequant/rebuild cost (ShadowKV `K_sparse = A·B` rebuild) enter as per-step constants from published time breakdowns (GEAR Fig 3a: fused quant + low-rank + sparse ≈ negligible).

# Success Criterion
Quantifiable, matching hypothesis thresholds: H2 survives screening iff the model predicts (i) `S > 1.15` at the (20%, 2-bit) corner (i.e., non-additive tier behavior is reachable at the corpus constants), AND (ii) byte-normalized tier latency varies by >10% across cells at the published constants, AND (iii) the regime check passes (tier fraction of predicted step latency >10% in at least one cell — the tier axis is testable in this regime). Output: the predicted peak-interaction cell, pre-registered as the place E3 must look first.

# Falsification Criterion
Quantifiable, matching hypothesis kill conditions: H2 is **analytically falsified** iff, under the model, byte-normalized transfer latency is constant across all 12 cells (within ±5%; pre-registered tolerance for a noiseless model) OR `S ≤ 1.15` at the corner — either makes the H2 mechanism list incapable of producing the hypothesis's >1.5× super-additive penalty. Independently: if the regime check shows `max(I/O, compute) = compute` at every cell, the tier hypotheses are declared "not testable in this regime" (hypothesis, Ambiguous Outcome) and E2b/E3 tier arms are canceled. E1 cannot falsify H1, the primary, or H3 (stated above) — those require E2a/E4 respectively.

# Estimated Complexity
Very Low (single offline script; no GPU, no harness).

# Expected Runtime
Minutes (one-time; re-run only if the machine's measured constants change the verdict, which is logged as a sensitivity check).

---

# Experiment E2a

# Experiment ID
**E2a — Quality sensitivity of 2-bit KV to the retained-token set (H1 test, GPU-only).**

# Target Hypothesis
**H1** (retention × bit-width interaction on quality). Primary kill candidate after E1 (E1 cannot touch H1).

# Purpose
Test H1's mechanism directly: H2O-style eviction retains exactly the high-attention-mass tokens; KIVI-style per-token value quantization confines error per token — so pruning forces 2-bit error onto the tokens the method most relies on, while the FP16 residual window (R=128) protects only recent tokens. The quantified anchors: quality drop at 2-bit relative to 16-bit is ≥2× larger at 20% retention than at 60%, and the R×B interaction term is significant at α=0.05 with effect ≥50% of the bit-width main effect, in ≥1 workload. This is a pure quality experiment: no tier, no serving framework — quality is layout-independent (D2). It also produces the per-workload quality coordinates E3's workload-selection rule and E4's H3 Pareto analysis depend on.

# System
Existing method implementation: KIVI codebase (HuggingFace Transformers-based; fused CUDA Q_MatMul + Triton quantization kernels, G=32, R=128 fixed) with an H2O accumulated-attention retention wrapper. **Pre-registered glue semantics (revision R1):** (i) H2O eviction is scored on **FP16 attention** — the retention wrapper runs its accumulation on the FP16-reconstructed attention path, so the retained set is decided independently of 2-bit numerics; quantization is applied to the retained set afterwards (the **decoupled arm**, isolating "2-bit error lands on retained heavy hitters" from "2-bit changed which tokens are heavy hitters"); (ii) H2O recent-window protection = 128 tokens = KIVI R (exact overlap, so the residual window is redundant inside the protected tokens — a pre-registered semantics, not an accident); (iii) quantization groups (G=32 contiguous retained tokens) are formed over the retained set in position order, with scales/zero-points recomputed per group after eviction compaction; (iv) the KIVI residual queue holds the last 128 *retained* positions. A second arm (same budget) runs H2O scores on 2-bit-reconstructed attention as a sensitivity contrast; the difference between arms is reported, and a large difference flags glue sensitivity that must be reported as a confounder, not an interaction. Contingency (pre-registered): if KIVI-only 2-bit shows borderline H1 behavior (interaction within ±20% of the 50%-of-main-effect threshold), add a GEAR repair arm (low-rank + sparse + n_b=20 buffer) as a cross-check that the interaction is not an artifact of one quantization recipe — GEAR is corpus-motivated because repair is exactly what reasoning workloads need (GEAR CoT: 40.20 with repair vs 7.67 plain per-token 2-bit). **No serving framework** — justification: H1 is about numerics; paged attention and batching alter memory layout and scheduling only. KIVI/GEAR's published quality numbers come from batch-1 greedy HF-style harnesses, so this instrument reproduces the corpus's quality-reporting conditions exactly. This is the cheapest sufficient instrument for H1.

# Workload
Both workload levels (H1's falsification requires "both workloads"):
- **W1 — RAG:** LongBench 2WikiMQA (multi-document QA, dispersed attention, avg input ~4.9K → truncated to the 4K-class cap per hypothesis control; F1 metric). Qasper/NarrativeQA are excluded as primary: Qasper is single-doc and NarrativeQA's ~18K inputs violate the 4K-class control (D3-adjacent scope control). Pre-registered contingency: if 2WikiMQA saturates (confounder 6 exclusion), fall back to Qasper.
- **W2 — Reasoning:** GSM8K 8-shot CoT, exact-match accuracy (KIVI/GEAR's documented sensitivity split: LongBench near-lossless vs GSM8K needing the residual window/repair — this is where the quality tolerance is expected to bind).
N per cell: N ≥ 128 target, pilot formula of §2 (pilot 32 requests at (100%,16-bit) and (20%,2-bit) per workload; for binary GSM8K, report achieved power at N=128 plus effect-size CIs). Context 4K class (2WikiMQA truncated; GSM8K ~900-token prefill). Output cap 256. Arrival pattern: n/a (offline batch-1 greedy, no concurrency). Reuse characteristics: **explicitly n/a** — not a serving test; per-request independence (A3).

# Hardware
Single GPU A100 80GB, clocks locked. GPU-only (no tier factor — D2). No host-DRAM tier involvement.

# Instrumentation Required
Per cell: quality per official metric (2WikiMQA F1; GSM8K exact-match accuracy), per-request quality scores (for interaction regression), seeds, KIVI commit hash + retention-wrapper commit hash, G/R hyperparameter log (G=32, R=128 everywhere), saturated-cell variance flag (≥2% cross-cell variance exclusion rule), token-level diagnostic for the mechanism claim if H1 survives: per-token quantization error (|X − X″|) by retention status (retained heavy-hitter vs evicted) on a 64-request subsample — the hypothesis's Expected Observation ("largest per-token quantization errors landing on retained heavy-hitter tokens").

# Baseline
(100%, 16-bit, GPU-only) cell as reference; additive-composition null = quality regression with main effects only (R linear in budget %, B dummy), no R×B term. Fairness: identical model, weights FP16, identical prompts/decoding (greedy, fixed seed) across all cells; the only difference between cells is (R, B); the 2-bit scheme's hyperparameters (G=32, R=128) are constants, not factors (hypothesis, confounder 9).

# Experimental Conditions
3 × 2 = 6 quality cells per workload × 2 workloads = 12 cells: R ∈ {100%, 60%, 20%} (H2O accumulated-attention policy) × B ∈ {16-bit FP16, 2-bit KIVI G=32 R=128}. **Plus the decoupled-arm contrast (revision R1):** the (20%, 2-bit) cell on each workload is also run with H2O scores computed on 2-bit-reconstructed attention (2 extra cells) — the difference vs the FP16-scored arm is the glue-sensitivity report. Fixed: Llama-2-7B, FP16 weights, batch 1 greedy, output cap 256, seeds fixed per cell, quality tolerance ≤3% relative drop vs baseline pre-registered (if every 2-bit cell passes tolerance, H1's quality coordinates are uninformative — recorded per the hypothesis's Ambiguous Outcome, and the interaction test still runs on relative drops). Saturated cells excluded per §2. R coded as categorical contrasts (revision R3); linear-coding robustness check reported.

# Success Criterion
Quantifiable, copied from hypothesis.md H1: in **at least one workload**, (i) [Q(16-bit) − Q(2-bit)] at 20% retention ≥ 2× the same drop at 60% retention, AND (ii) the R×B interaction term in the quality regression is significant at α=0.05 with effect ≥50% of the bit-width main effect (per §2 definitions). Secondary support evidence: token-level diagnostic shows concentrated per-token 2-bit error on retained heavy hitters. If H1 survives: E3's workload selection rule (below) applies; E3 proceeds (gate: H1 alive).

# Falsification Criterion
Quantifiable, copied from hypothesis.md H1: H1 is falsified if the R×B interaction term on quality is non-significant at α=0.05 AND its effect size is <10% of the bit-width main effect in **both** workloads. (Support in only one workload is partial support, not a kill — that workload becomes E3's target per the selection rule.) If H1 dies here AND E1/E2b killed H2, the primary's interaction surface is empty and E3/E4 are canceled (overall null; §1 gate).

# Estimated Complexity
Low (existing code + ~20-line retention wrapper; no serving framework, no tier).

# Expected Runtime
Hours (~4–8 GPU-hours total incl. pilot and token-level diagnostic: 12 cells × 128 requests, GSM8K decode ~60 ms/req, 2WikiMQA prefill ~0.3 s/req at batch 1).

---

# Experiment E2b

# Experiment ID
**E2b — Kernel/transfer microbenchmarks: dequant cost, small-transfer efficiency, PCIe utilization at two sizes.**

# Target Hypothesis
**H2** (empirical gate after E1). Informs whether E3's tier arm is worth running and supplies the calibration constants E3 requires.

# Purpose
H2's mechanism list claims that halving tokens and halving bytes/token does not halve tier latency because (a) small transfers under-utilize PCIe, (b) residual/buffer bytes scale with retention, (c) low-bit state needs dequant/rebuild on the fetch path. E1 tests this under the corpus model; E2b tests it on the real machine. If byte-normalized transfer latency is constant across transfer sizes on the actual A100 platform, A8 holds empirically for the transfer path and H2's second falsification arm ("byte-normalized transfer latency is constant across all cells") is satisfied without any serving experiment — H2 dies at microbenchmark cost. If small transfers are materially less efficient, the constants measured here feed the E1 model's sensitivity sweep and E3's emulator.

# System
Microbenchmark (standalone PyTorch/CUDA harness, no serving framework — justification: the quantities measured are kernel-level and transfer-level properties that are layout- and scheduler-independent; a serving harness would add confounds (queueing, batching interference) to exactly the measurements meant to be clean). Components: (1) dequant-matmul: KIVI's fused 2-bit Q_MatMul vs FP16 matmul at equal byte volume (2 sizes: 1 MB and 8 MB of KV state); (2) small-transfer efficiency: pinned-memory `cudaMemcpyAsync` D2H/H2D at two sizes matching the hypothesis's chunking (512-token chunk = 8 MB FP16 per layer; and a large 4096-token 64 MB transfer), reporting achieved GB/s vs the machine's nominal PCIe bandwidth; (3) fetch-count scaling: N small transfers vs 1 aggregated transfer at equal total bytes (per-fetch fixed-latency isolation); (4) PCIe utilization at 2 sizes under concurrency 1 vs 32 concurrent streams (T5 condition: high concurrency, root-complex contention).

# Workload
Synthetic KV tensors only (Llama-2-7B shapes: 32 layers, hidden 4096, per-token per-layer 16 KB FP16; 2-bit KIVI-shaped at ≈21.7% of FP16 with R=128 residual). Request count: n/a — synthetic transfers. Arrival pattern: fixed-size transfer bursts, concurrency {1, 32}. Reuse characteristics: **explicitly n/a** — not a serving test.

# Hardware
Single A100 80GB + host DRAM via the real PCIe path (pinned host buffers). This is the one experiment that must use the real host path (not the emulator) — its outputs *define* the emulator constants. Nominal bandwidth measured on-machine (A100 may be PCIe 4.0, ~64 GB/s nominal, vs the corpus's 31.5 GB/s PCIe 3.0 constant; all conclusions are stated as ratios so they transfer across constants; the corpus constant is retained as a secondary sensitivity point).

# Instrumentation Required
Per benchmark: kernel times (dequant matmul µs and µs/MB at both sizes), achieved transfer bandwidth (GB/s, median + P95 over ≥200 transfers), per-fetch fixed latency (µs), byte-normalized transfer latency (µs/MB) at both sizes and both concurrencies, seeds n/a (deterministic), commit hashes of benchmark harness + KIVI kernel code, machine constants log (nominal vs achieved PCIe, pinned-buffer page size), calibration log feeding E3's emulator (see E3).

# Baseline
Linear model: latency = bytes/measured_bandwidth (no fixed per-fetch cost) — the additive null. Fairness: all sizes measured on the same pinned buffers, same stream discipline, same GPU clocks; equal byte volume for the dequant comparison (confounder 1's equal-volume rule).

# Experimental Conditions
Fixed hyperparameters as in hypothesis.md: chunk = 512 tokens, page block = 16 tokens, KIVI G=32 R=128 shapes, GEAR n_b=20 buffer shape (as a fetch-path component), per-fetch sizes {8 MB, 64 MB}, concurrency {1, 32}, transfer count ≥200 per cell. Dequant at byte volumes {1 MB, 8 MB} of KV, FP16 vs 2-bit fused.

# Success Criterion
Quantifiable, matching H2's mechanism: H2 survives E2b iff measured byte-normalized transfer latency is non-constant across the two sizes (small-transfer inefficiency >15% relative to large, at the published/tolerance threshold) AND per-fetch fixed latency is material (>15% of per-step transfer time at the 512-token chunk size) — i.e., the machine exhibits the nonlinearity H2's mechanism requires. Outputs: measured constants F (per-fetch µs) and B (achieved GB/s) substituted into the E1 model; if the substituted model now predicts `S > 1.15`, E3's tier arm is warranted; if E2b's measured constants also feed the emulator (≤15% fidelity requirement, below).

# Falsification Criterion
Quantifiable, copied from hypothesis.md H2's second arm: if byte-normalized transfer latency is constant across the two sizes within ±15% AND the per-fetch fixed latency is negligible (<15% of per-step transfer time at the 512-token chunk), A8 holds empirically on the transfer path — H2 is falsified without any serving-tier experiment, and E3's tier arm is canceled (E3 then runs GPU-only cells only, or is skipped per §1 gating). Kernel artifact note: if dequant cost differs from FP16 matmul by >15% at equal byte volume, that difference is recorded as a confounder for E3/E4 attribution (hypothesis, confounder 1), not as tier interaction.

# Estimated Complexity
Low.

# Expected Runtime
Hours (~2–4 GPU-hours including the concurrency sweep).

---

# Experiment E3

# Experiment ID
**E3 — Minimal tier experiment: 2×2×2 subset on one workload (interaction-direction estimation + emulator calibration).**

# Target Hypothesis
Primary (interaction directions), **H2** (super-additivity at the joint minimum), with H1's surviving workload as the quality envelope. Runs only if the §1 gate passes (at least one of H1 alive, H2 alive, or E2b nonlinear transfer).

# Purpose
Estimate the direction and magnitude of the tier-axis interactions with the minimal matrix before committing to the full 24-cell run. This is the first experiment that needs the paged-attention serving harness (D2's exemption does not apply: TTFT/TPOT/throughput/HBM under fixed concurrency require a scheduler, and memory-savings realization requires paging/fragmentation behavior — hypothesis, confounder 4). It also executes the pre-registered tier-emulator calibration (anchor cell, ≤15% TTFT mismatch) and the A2 regime check under real serving conditions. Its outcome is a yes/no gate for E4.

# System
vLLM-class serving framework (paged attention) with **minimal modification** (cost tier 6, deliberately not complex modification): (a) KIVI 2-bit cache backend (G=32, R=128) ported from existing code, (b) H2O accumulated-attention retention as the cache-budget policy, (c) host-tier path = either real host-DRAM offload (pinned buffers, 512-token chunks, async copy — the H2O/FlexGen offload pattern) or the calibrated emulator (nominal PCIe bandwidth + measured per-fetch latency constants from E2b). No new controller, no joint optimizer, no scheduler change (hard constraint). Tier-emulator calibration requirement: one anchor cell (100%, 16-bit, GPU+host) run both real-offload and emulated; if TTFT mismatch >15%, tier results are ordinal-only and H2/H3 conclusions restricted accordingly (pre-registered, hypothesis confounder 5). Cheapest sufficient: real host offload is preferred on this machine (Llama-2-7B KV at 4K × batch 32 ≈ 2 GB — trivially host-resident); the emulator exists only for portability of the results, never as the primary measurement.

# Workload
**One workload** (D3): pre-registered selection rule — the workload in which E2a's H1 interaction was significant and in the H1-supporting direction; if both, the reasoning workload (GSM8K 8-shot CoT) by default (KIVI/GEAR's documented 2-bit sensitivity split and the failure scenario's "reasoning/RAG where quality repair or small transfers dominate" arm; GSM8K's ~900-token prefill also makes it decode-heavy, which the A2 regime check requires for tier testability). Contingency: if H1 was significant only in RAG, E3 runs on 2WikiMQA (4K class) instead. Request count: N per cell from §2 pilot formula, floor 128, cap 512; cells with effective N < 32 excluded. Context 4K class (GSM8K ~900; 2WikiMQA truncated to 4096). Output cap 256. Arrival: fixed concurrency 32, continuous batching, saturated arrival, no idle gaps (hypothesis control). Reuse: **explicitly n/a** — prefix caching off, requests independent (A3); reuse is deliberately excluded because the factorial is the object of measurement, not reuse policy.

# Hardware
Single A100 80GB, clocks locked. Host DRAM tier via real offload (preferred) or calibrated emulator, with the anchor-cell calibration log and ≤15% mismatch pre-registration.

# Instrumentation Required
Per cell: TTFT median + P95, TPOT median + P95, throughput (tokens/s) and SLO attainment at the ≤3% quality tolerance, quality per official metric (to confirm the corner stays in/out of tolerance), measured peak HBM via `cudaMemGetInfo` (confounder 4: never inferred), transfer bytes + fetch count (instrumented offload path — required to distinguish "bytes saved" from "latency saved", hypothesis DV list), kernel times (dequant/rebuild on fetch path), per-request latency logs, seeds, commit hashes (harness, KIVI backend, retention wrapper, offload path), emulator calibration log (anchor-cell comparison, mismatch %, pre-registered outcome), fake-quantization variant cell (2-bit numerics dequantized-in-compute at (20%,2-bit,GPU-only)) to attribute any 2-bit latency delta to kernels vs numerics (confounder 1).

# Baseline
(100%, 16-bit, GPU-only) as reference cell; additive-composition model (main-effects-only regression on the 2×2×2) as the statistical null; H2's additive penalty bound `pen(20%,2-bit) ≤ 1.15×[pen(20%,16-bit) + pen(100%,2-bit)]` with `pen(R,B) = T(R,B,GPU+host) − T(R,B,GPU-only)` as the H2 null. Fairness: identical kernels/workloads/seeds across all cells; actual-bytes accounting; fixed concurrency 32; TTFT and TPOT analyzed separately (confounder 7, no aggregate latency variable).

# Experimental Conditions
2 × 2 × 2 = 8 cells: R ∈ {20%, 100%} (H2O policy) × B ∈ {16-bit, 2-bit KIVI G=32 R=128} × T ∈ {GPU-only, GPU+host}, one workload, fixed hyperparameters as copied from hypothesis.md (chunk 512, block 16, batch 32, output cap 256, quality tolerance ≤3%). Plus: one calibration anchor cell (100%,16-bit,GPU+host, real-offload vs emulator), one fake-quant cell at (20%,2-bit,GPU+host) — NOT only GPU-only (revision R1: the dequant-on-fetch-path kernel cost is attributed at the tier cells where it can masquerade as R×B×T) — and, on the W2 repeat (when triggered), the corner cell (60%, 2-bit, GPU+host) added as the 9th cell to close the R=60% tier blind spot (revision per critic D). E3's null verdict is relabeled "additive at R ∈ {20%, 100%} endpoints"; the full primary falsification is reserved for E4 (revision per critic #1/#D).

# Success Criterion
Quantifiable, matching hypothesis thresholds: E3 supports proceeding to E4 iff (i) any interaction term involving tier (R×T, B×T, R×B×T) is significant at α=0.05 (BH across metric×workload families) with `R²_int_rel > 0.50` (unified statistics, revision R2) in ≥1 of TTFT/TPOT/throughput/HBM, OR (ii) H2's super-additivity holds at the corner: `pen(20%,2-bit) > 1.5×[pen(20%,16-bit) + pen(100%,2-bit)]` on TTFT or TPOT, OR (iii) the A2 regime check passes (tier fraction of step latency >10% in ≥1 cell) while interactions are borderline (R²_int_rel ∈ [0.10, 0.50], the pre-registered gray zone) — borderline cases trigger the W2 confirmation repeat (including the 9th cell (60%,2-bit,GPU+host)) rather than cancellation. Output: signed interaction directions + magnitudes, pre-registered as E4's confirmatory targets.

# Falsification Criterion
Quantifiable, matching hypothesis kill conditions: H2 is falsified if `pen(20%,2-bit) ≤ 1.15×` the sum of single-axis penalties, OR byte-normalized transfer latency is constant across all cells (slope of tier latency vs actual transfer bytes within ±15% of linear through origin; A8 holds). The primary is NOT declared falsified at E3: an E3 null is only "additive at R ∈ {20%,100%} endpoints" (revision per critic D); after the W2 confirmation repeat (when triggered), if all interaction terms are non-significant and `R²_int_rel ≤ 0.10` at the endpoints, the primary claim is downgraded to "no interaction at the endpoints; full-matrix verdict requires E4" — E4 is canceled only if the 9th cell (60%,2-bit,GPU+host) also shows no interaction AND the endpoints are null (this closes the 60% blind spot on the kill path, revision per critic D). Emulator-fidelity failure (>15% mismatch) downgrades tier results to ordinal-only, making H2/H3 non-discriminable (recorded, not papered over — hypothesis, Ambiguous Outcome).

# Estimated Complexity
High (revision R4 — the KIVI-2-bit-in-paged-framework port is a mini-project: per-block quantized metadata in the paged allocator, residual-flush across 16-token pages, fused dequant attention over gathered non-contiguous layouts, block-manager eviction wrapper, instrumented offload path; this is NOT "minimal modification"). Pre-registered port budget: ≤10 person-days, with a trigger at 10 person-days to invoke the fallback (HF-based batch-1 harness with a real offload path, or emulator-based tier emulation at the ≤15% anchor-calibration standard) — a stalled port cannot silently cancel the tier question.

# Expected Runtime
~1–2 weeks engineering (port + calibration) + ~1–2 days of cell runs (10–18 cell-runs × 128–512 requests at fixed concurrency 32, plus calibration anchor).

---

# Experiment E4

# Experiment ID
**E4 — Full factorial confirmation + Pareto analysis (runs only if E3 survives).**

# Target Hypothesis
Primary (full-matrix interaction confirmation), **H3** (Pareto reversal), with H1's effect sizes re-estimated at full resolution.

# Purpose
Confirm the interaction terms found in E3 at full 3×2×2×2 resolution on both workloads, and execute H3's Pareto test: is the axis-best composition (20%, 2-bit, GPU-only) on the frontier of the joint metric (quality × TTFT/TPOT × HBM)? This is the hypothesis's own confirmation design, reached only after every cheaper kill has failed. It also runs the separated max-batch capacity sweep (hypothesis, confounder 3 neutralization — the memory savings surface there, not as uncontrolled batch changes) and the SLO-attainment coordinates for H3's composite.

# System
Same vLLM-class harness as E3 (paged attention, minimal modification, three existing mechanisms, no new controller). Cheapest sufficient: E3 already built and calibrated the entire stack; E4 adds only matrix breadth, the capacity sweep, and the replication runs. All experiments except this one and E3 were designed to avoid the serving framework (E1: model-only; E2a: quality is layout-independent; E2b: kernel/transfer properties are layout-independent) — this experiment is the sole justification for the framework requirement in the hypothesis.

# Workload
Both workload levels: RAG = LongBench 2WikiMQA (4K class, F1; Qasper fallback if saturated) and reasoning = GSM8K 8-shot CoT (exact match). Request count: N per cell from §2 formula (floor 128, cap 512), effective-N ≥ 32 exclusion; 24 cells × 2 workloads. Context 4K class, output cap 256. Arrival: fixed concurrency 32, saturated, continuous batching; separate max-batch capacity sweep (batch 1→OOM at the two memory-extreme cells per workload, reported separately, not merged into the interaction analysis). Reuse: **explicitly n/a** — prefix caching off, independent requests (A3).

# Hardware
Single A100 80GB, clocks locked; host tier via the E3-calibrated path (real offload or emulator with ≤15% anchor match; if E3's calibration failed, tier results are ordinal-only and H3 is restricted accordingly).

# Instrumentation Required
Per cell, as E3 plus: per-cell quality per official metric, TTFT median + P95, TPOT median + P95, throughput + SLO attainment at ≤3% quality tolerance, measured peak HBM (`cudaMemGetInfo`), transfer bytes + fetch count, kernel times, seeds, commit hashes, emulator calibration log carried over from E3; additionally the max-batch capacity per workload (separate sweep) and, for H3, ≥3-seed replication of every dominance-pair cell (axis-best vs each candidate dominator; D4).

# Baseline
(100%, 16-bit, GPU-only) as reference cell on both workloads; additive-composition model (main-effects-only regression) as the statistical null. Fairness: identical kernels, workloads, seeds, and concurrency across all 24 cells; actual-bytes accounting (confounder 2); saturated cells excluded from quality analysis (confounder 6); fixed batch 32 for the matrix (confounder 3); measured not inferred HBM (confounder 4); TTFT/TPOT separate (confounder 7); 2-bit scheme hyperparameters constant (confounder 9).

# Experimental Conditions
Full 3 × 2 × 2 × 2 = 24 cells: R ∈ {100%, 60%, 20%} × B ∈ {16-bit, 2-bit KIVI G=32 R=128} × T ∈ {GPU-only, GPU+host} × W ∈ {2WikiMQA, GSM8K 8-shot CoT}, fixed hyperparameters as copied from hypothesis.md (chunk 512 tokens, page block 16, max batch 32, output cap 256, quality tolerance ≤3%, weights FP16, Llama-2-7B). Interaction regression: all two-way + three-way terms, Type III SS, BH across the 5 metric families.

# Success Criterion
Quantifiable, copied from hypothesis.md (unified statistics, revision R2): **Primary** — in ≥1 of TTFT/TPOT/throughput/HBM/quality (and ≥1 workload), at least one interaction term is significant at α=0.05 (BH-corrected across metric×workload families) AND `R²_int_rel > 0.50`. **H1** (re-estimated at full resolution) — 2-bit drop at 20% ≥ 2× the drop at 60% AND R×B interaction significant with effect ≥50% of bit-width main effect in ≥1 workload (per-workload verdicts reported). **H2** — `pen(20%,2-bit) > 1.5×[pen(20%,16-bit) + pen(100%,2-bit)]` on TTFT or TPOT, with transfer volume dropping monotonically while fetch count/small-transfer latency explain the gap (byte-normalized latency non-constant). **H3** — the axis-best cell (20%, 2-bit, GPU-only) is dominated by ≥1 other cell with ≥10% improvement in ≥1 primary metric and ≤5% regression in every other, outside measurement error (95% CI on the per-metric difference), replicated across ≥3 seeds; expected candidates per the hypothesis's Expected Observation: (20%,2-bit,GPU+host) improving HBM >30% with ≤5% TTFT loss, or (60%,2-bit,GPU-only) improving quality ≥10% with ≤5% latency loss.

# Falsification Criterion
Quantifiable, copied from hypothesis.md: **Primary** — falsified if, for every dependent variable and both workloads, all two-way and three-way interaction terms are non-significant (α=0.05, BH across metric×workload families) AND `R²_int_rel ≤ 0.10` for all metrics (additive-composition null; legitimate, publishable outcome reclassifying RQ-8 as engineering composition). Gray zone pre-registered: significant interaction with R²_int_rel ∈ [0.10, 0.50] is inconclusive — reported as such, not as support or null. **H1** — falsified if R×B interaction non-significant AND effect <10% of bit-width main effect in both workloads. **H2** — falsified if penalty ≤1.15× the sum of single-axis penalties, or byte-normalized transfer latency constant across cells. **H3** — falsified if the axis-best cell is on the Pareto frontier within measurement error (no cell achieves ≥10% improvement in any primary metric with ≤5% regression in all others), replicated across ≥3 seeds and both workloads. **Overall null** — all three falsified ⇒ axes compose additively under this workload × hardware × metric envelope; no compound mechanism warranted (hypothesis, Falsification Condition). R coded as categorical contrasts, identical to E2a (revision R3).

# Estimated Complexity
High (full factorial serving runs + capacity sweep + replication; still minimal system modification — no new components beyond E3's).

# Expected Runtime
Multi-day (24 cells × 2 workloads × N≥128 at fixed concurrency 32 with instrumentation ≈ 48+ cell-runs at 1–2 h each, plus the max-batch sweep and 3-seed dominance-pair replication ≈ 60–80 additional runs).

---

## 5. Gating summary

| Gate | Kill condition (from hypothesis.md, verbatim thresholds; unified R²_int_rel per revision R2) | Survives ⇒ |
|---|---|---|
| E1 | H2 analytically falsified: byte-normalized latency constant under corpus model (A8) or `S ≤ 1.15`; or A2 regime check → tier untestable | E2a (always), E2b (if H2 alive) |
| E2a | H1 falsified: R×B interaction non-significant AND <10% of bit-width main effect in **both** workloads | E3 (if H1 or H2 alive) |
| E2b | H2 empirically falsified: byte-normalized transfer latency constant (±15%) on real machine | E3's tier arm |
| E3 | Endpoint null only (R ∈ {20%,100%}): all interactions non-significant, `R²_int_rel ≤ 0.10`, `pen ≤ 1.15×`, or A8 constancy (after W2 repeat incl. the 9th cell (60%,2-bit,GPU+host)); full primary falsification reserved for E4 | E4 |
| E4 | Full-matrix falsification per §E4 (gray zone [0.10, 0.50] = inconclusive) | Overall null (publishable): RQ-8 reclassified as engineering composition |

All thresholds above are copied verbatim from the locked hypothesis (primary: significant interactions α=0.05 + R²_int_rel > 0.50; H1: ≥2× drop ratio + ≥50%-of-main-effect interaction; H2: >1.5× super-additive; H3: ≥10% improvement with ≤5% regression, ≥3 seeds). Every experiment records seeds, commit hashes, measured (never inferred) HBM, transfer bytes + fetch count, the emulator calibration log, and the glue-semantics log (revision R1), per the hypothesis's instrumentation requirements. No experiment proposes a controller, joint optimizer, or new algorithm; the three axes use only existing H2O/KIVI/GEAR mechanisms and a host-DRAM tier.