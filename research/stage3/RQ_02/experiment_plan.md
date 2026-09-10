# Stage 3A · RQ-2 Experiment Plan — Fragility of Fixed Retention, Precision, and Chunk Budgets

**Contract:** Hypothesis `research/stage3/RQ_02/hypothesis.md` is LOCKED. All thresholds, tolerances, operating points, control variables, and confounder-mitigations below are copied from it verbatim (Chinese quotes retained) and are never softened. This plan designs measurement only. It does not propose optimization algorithms, adaptive controllers, or system modifications beyond config-level use of published implementations and profiler instrumentation.

**Cost-tier ladder used for ordering (strictly prefer cheaper):** 1 trace/existing-data analysis → 2 simulation → 3 instrumentation of existing implementation → 4 small microbenchmark (single GPU) → 5 existing benchmark harness → 6 minimal system modification → 7 complex system modification. Each later experiment runs **only if** every earlier one that could have stopped it failed to stop it.

**Global controls (from hypothesis.md, apply to all experiments):** Model weights FP16/BF16, KV-only compression (no weight quantization in quality runs); single A100-80GB; unified HF Transformers + FlashAttention-2 eager; greedy decoding; fixed seed (≥3 seeds on one anchor cell to confirm determinism, confounder 3); full official splits per condition (LongBench 4,750-case family; synthetic/stress ≥100 requests); batch 8 for latency (TPOT ≥1,000 decode steps); official evaluation harnesses per task (LongBench official script, lm-eval-harness, official PG19/∞Bench/LongGenBench protocols); tolerance predeclared — any post-hoc adjustment is cheating.

**Global instrumentation contract (all GPU experiments):** per-condition quality scores (official metrics); measured peak HBM (nvidia-smi @100 ms + `torch.cuda.max_memory_allocated`, reported in **bytes**, never nominal percentages — confounder 6); TPOT P50/P95 over ≥1,000 decode steps; kernel-overhead share (dequant / selection / recompression as fraction of decode-step time); TTFT; seed, framework versions, method-repo commit hashes.

**Global pre-registered exclusions (confounder 4, hypothesis.md):** metrics with full-KV baseline <5 or >95 are reported in absolute terms but excluded from relative-drop statistics; single-dataset crossings do not count (requires ≥2 task classes or ≥2 datasets within a class crossing simultaneously); TPOT differences appearing only at batch 1 are batching noise, not kernel overhead (H2 not supported); oracle gains obtained only by switching method family or increasing memory budget count as non-evidence (H3 not supported).

**Gating diagram**

```
E1 (offline screening, no GPU) ── triage matrix for all downstream cells
        │ (never a verdict)
        ▼
E2 (single-GPU quality matrix, HF+FA2 eager) ── KILL H1 (positive) or SURVIVE
        │ positive → skip E5 quality-completion parts; negative → E5 before any H1 verdict
        ▼
E3 (TPOT/kernel microbenchmark, 1 method first) ── KILL H2 or SURVIVE
        │ negative → expand to remaining H2 mechanisms; all survive → H2 falsified
        ▼
E4 (per-condition oracle re-tuning, same measured HBM) ── KILL H3 or SURVIVE
        │ (runs on methods/conditions that survived E2/E3/E5)
        ▼
E5 (open-value cells unconditional — StreamingLLM evicted-distance, H2O long-form/CoT, GEAR per-task, PyramidKV per-task; completion parts only if E2 negative or universal-fragility rule fires)
```

---

# Experiment E1 — Offline screening: within-paper paired-delta triage of published tables

# Target Hypothesis
All three (H1, H2, H3), at the level of **candidate pruning and witness eligibility**. Explicitly **no verdict** on any hypothesis: cross-paper and even within-paper published numbers are not matched to our harness; per hypothesis.md, matched-harness evidence is produced only in E2+. E1's outputs are (i) the triage that decides which methods/cells E2-E5 must measure first, and (ii) a list of cells that E1 can already *kill* (see Falsification Criterion).

# Purpose
Maximize the chance that the first matched-harness run (E2) lands on a cell that actually crosses the tolerance, and prevent GPU time from being spent on conditions the published data already show are dead or ambiguous. Compute, for every method, its **native-task delta** (must be <10% for H1 to be witnessable) and its **non-native-condition delta** (must be ≥10% for H1 to be killable) from the *same table / same model* of each paper — within-paper pairs are screening-strong, cross-paper pairs are screening-weak — and convert them to relative percentages so the pre-registered tolerance applies mechanically.

# System
Trace analysis / existing-data analysis (cost tier 1; no GPU, no code execution beyond arithmetic on tables already in the corpus). Why cheapest sufficient: every number needed to triage already exists in the six paper notes; the cost of mis-triage (a wasted E2 cell) is far below the cost of this step being anything heavier.

# Workload
Published tables only (request counts/context lengths are attributes recorded for the downstream E2 cells, not consumed here): SnapKV Table 1 (LWM-Text-Chat-1M, LongBench 16, capacity 1024/2048/4096), KIVI Tables 3–5 (Llama-2-7B/13B, Falcon-7B, Mistral-7B; CoQA/TruthfulQA/GSM8K; LongBench-8; G/R sweeps), GEAR Tables 1–2 (CoT trio + LongBench-21, 2/4-bit), PyramidKV Table 1/Fig. 3 (Mistral-7B, LLaMA-3-8B, KV=64/2048), SCOPE Tables 1–3 (LongGenBench-4K/8K, GSM8K+ phase-split, efficiency), StreamingLLM Tables 1–2/6 (PG19 PPL, sink sweeps), H2O Tables 1–6 (lm-eval/HELM). Arrival pattern: n/a (tables are aggregated). Reuse: n/a (tables are aggregated).

# Hardware
None required (desk analysis). The first GPU becomes necessary at E2; see E2's justification.

# Instrumentation Required
No new instrumentation. Output artifact: a per-method×condition matrix with three columns per cell — (a) paired relative delta = (compressed − full)/full from the *same* paper table/model; (b) evidence class = within-paper (strong) vs cross-paper (weak); (c) eligibility flags: native-safe (<10%), crossing (≥10%), floor-excluded (<5 baseline), ceiling-excluded (>95), ambiguous (native also ≥10%). Record the source (note §, table) for every number so the triage is auditable.

# Baseline
Within-paper full-KV rows of the same tables (e.g., SnapKV "All KV" row, KIVI "16-bit" row, SCOPE "Full Cache"). Fairness: only same-model, same-harness pairs are used for deltas; cross-paper averages are labeled screening-weak and never used for classification.

# Experimental Conditions
Screening cells (one row per method), with the fixed published operating points copied from hypothesis.md; **same-model rows only** (revision R4 — in-paper predictions are read from the exact model/harness column E2 will run):
- **SnapKV** (capacity 1024, window 32, kernel 7): native = LongBench 16 suite INCLUDING summarization (per revision R1). In-paper native deltas on Mistral-7B: NrtvQA 26.82→25.54 = −4.8%; **GovReport-on-Mistral: no in-paper value → flagged "measure"** (the −29.3% figure is the LWM-Text-Chat column and is NOT used for a Mistral prediction). Non-native (out-of-suite) H1 cells: ∞Bench En.Sum, GSM8K 8-shot CoT → flagged "measure". QMSum cell: no in-paper value → flagged "measure".
- **KIVI** (2-bit, G=32, R=128): native = Llama-2-7B/13B + Falcon-7B (MQA) + Mistral-7B on CoQA/TruthfulQA/GSM8K + LongBench 8 (per revision R1; Falcon cells are in-suite observations, NOT H1 evidence). In-paper: Llama-2-7B native CoQA 63.88→63.05 = −1.3%, GSM8K 13.50→12.74 = −5.6%; Falcon in-suite observations: CoQA −3.9%, GSM8K −25.1% (floor-excluded), MultiNews −38.9%, TREC −23.1% — all **published-point validity observations**. Note: Falcon numbers come from max_seq=4096-truncated runs (KIVI.md §9) — E2's truncation protocol matches this (revision R4). Non-native H1 cells: ∞Bench En.Sum, long-decode phase cell (output ≥1024) → flagged "measure".
- **GEAR** (2-bit, s=2%, r=4, n_b=20): native CoT reasoning near-lossless (avg 40.20 vs FP16 40.52 = −0.8%); LongBench-21 2-bit avg 25.48 vs 26.82 = −5.0% → **borderline, per-task unresolved** → E5-c (unconditional).
- **PyramidKV** (α=8, β=20, KV=64): Mistral-7B avg 32.19 vs FKV 42.71 = −24.6%; per-task native values not recorded in notes → **ambiguous until per-task native delta is computed** (E5-a, unconditional).
- **StreamingLLM** (4 sinks + rolling, total 2048): PG19 PPL stable to 4M tokens; **no in-paper test of retrieval at evicted distance** → new measurement required (E5-b, unconditional; needles placed beyond the window per revision R7).
- **SCOPE** (λ1+λ2=2048/4096 ≈60%, λ2=256): H2 anchor Full 36.57 vs Slide 18.28 tok/s (batch 8, RTX 3090) = TPOT +100%; Discontinuous 25.92 = +41%; H3 anchor GSM8K+ prefill-only 27.75 → prefill+decode 52.17 at same 35% total budget = +88%.
- **H2O** (20% KV): native lm-eval/HELM all <10% (OpenBookQA −0.5%, RTE +0.73%); no in-paper non-native (long-form/CoT) data → no signal, excluded from E2-primary; long-form/CoT cells in E5-d (unconditional).
- **H3-knob screening** (in-paper same-budget sweeps): KIVI G 32→64: GSM8K 20.77→21.00 (+1.1%); G→128: 17.29 (−16.8%); R sweep 32/64/96/128: 20.62/19.86/20.55/20.77 → **predicted oracle quality gain <5%** for KIVI G∈{32,64}; StreamingLLM sinks 1 vs 4 on Falcon: flat → **predicted <5%**; SCOPE phase-split +88% and frequency +42% → **predicted ≥10%/≥15%**.

# Success Criterion
E1 succeeds as a triage iff it produces, for every method in the E2/E3/E4 matrix, a pre-registered prediction of (i) which cells will cross the H1 tolerance, (ii) which H2 mechanism cells have an in-paper effect anchor, (iii) which H3 knob sweeps have an in-paper gain anchor. Success does **not** depend on any crossing being found; a cleanly negative triage is equally successful (it shrinks E2 to a verification set and re-focuses on falsification).

# Falsification Criterion
E1 can already kill (screening-level kills, never hypothesis-level verdicts): **(K1)** a method's **H1 witness eligibility** — if its own published native-task deltas are ≥10% (e.g., PyramidKV KV=64, pending per-task native values), it cannot witness H1 (native-safety precondition violated) and is moved to the "ambiguous outcome / universal fragility" track, excluded from the E2 kill matrix; **(K2)** specific E4 oracle sweeps whose in-paper same-budget sweeps are flat at <5% (KIVI G∈{32,64} on GSM8K; StreamingLLM sink count on Falcon) are pruned from E4, saving GPU time; **(K3)** if the triage finds *no* (method, dimension) cell that is simultaneously native-safe and ≥10%-crossing, E2 shrinks to the minimal verification set and the experiment's posture flips to falsification-first (E5 triggered unconditionally). These are the maximum kill powers available at cost tier 1; any claim beyond them would violate the matched-harness requirement stated in the task brief.

# Estimated Complexity
Very Low

# Expected Runtime
hours (desk analysis; 0 GPU)

---

# Experiment E2 — Single-GPU quality matrix (kills H1)

# Target Hypothesis
H1 primarily (native-safe vs non-native-crossing at the method's own published budget). Secondary: produces the measured-HBM budgets and surviving conditions that E4's oracle study will use, and the paired-delta baseline cells for every condition.

# Purpose
First matched-harness verdict. In our harness (A100-80GB, HF+FA2 eager, greedy, official splits), verify whether SnapKV at capacity 1024 crosses the ≥10% tolerance on long-form summarization while staying <10% on its native LongBench QA (task-class dimension), and whether KIVI 2-bit crosses on the MQA layout (Falcon-7B) while staying <10% on its native Llama-2-7B cells (KV-layout dimension), plus one cheap decode-phase cell (SnapKV long decode). Positive result on either method kills H1 immediately (per hypothesis: "H1 成立 iff 至少一个非原生条件出现 ≥10% 相对质量下降，且同方法原生任务 <10%").

# System
HF Transformers + FlashAttention-2 eager — the hypothesis's declared control environment — plus the published method implementations as-is, config-level only: official SnapKV repo (HF-native, listed in SnapKV.md §5) and official KIVI repo (HF-native, KIVI.md §4/§5). Justification for the method-side path: each method's own attention kernels are its published operating mechanism (e.g., KIVI fused dequant Q_MatMul); using the repos is the only way to hold the operating point identical to the paper, and quality is unaffected by kernel choice because KIVI/GEAR quantization is round-to-nearest deterministic. No vLLM/SGLang (paged layer would introduce block-granularity confounds, hypothesis control variables). No deviation from the declared control environment except the method's own required kernels, which the hypothesis's control block implicitly requires ("统一 HF Transformers + FlashAttention-2 eager 实现（SnapKV/StreamingLLM/KIVI/SCOPE 的原生实现环境）").

# Workload
- SnapKV cells (model Mistral-7B-Instruct-v0.2, GQA — the model both methods published on; GQA additionally covers the GQA-layout condition):
  - A1 native: NrtvQA (108), Qasper (168) — official LongBench splits.
  - A2 non-native (out-of-suite, revision R1): ∞Bench En.Sum (long-form summarization, official split) and GSM8K 8-shot CoT (reasoning) — the only cells eligible as H1 evidence for SnapKV.
  - A3 phase cell (unconditional, revision R2): GovReport, 100-request seed-fixed subset, forced gen length 1024 (decode-phase state beyond the prefill-selected setting; SnapKV never re-compresses decode KV).
  - A4 observation cell (in-suite, never H1 evidence): GovReport (173) at official gen length — records published-point validity on Mistral (no in-paper anchor; flagged "measure" in E1).
- KIVI cells:
  - B1 native (Llama-2-7B, MHA): CoQA (dev split), GSM8K 5-shot (full 1,319) via lm-eval-harness.
  - B2 non-native (out-of-suite, revision R1): ∞Bench En.Sum + long-decode phase cell (Llama-2-7B, output forced to 1024) — the only cells eligible as H1 evidence for KIVI.
  - B3 observation cells (in-suite, never H1 evidence): Falcon-7B (MQA) CoQA, MultiNews (216), TREC (200) at the paper's max_seq 4096 truncation protocol (revision R4) — records published-point validity on the MQA layout; plus one untruncated sensitivity cell (MultiNews untruncated) to quantify truncation sensitivity.
- Context lengths: LongBench official splits (avg input ~13K, range 2K–18K per task) for native/observation cells; ∞Bench per official protocol; CoQA/GSM8K short-context; KIVI Falcon cells truncated at max_seq 4096 per the papers (revision R4). Output length: official harness-specified per task (nominal gen 256; long-form tasks per official script); A3/B2 forced 1024.
- Arrival pattern: batch 1 sequential quality eval, no serving, no concurrency (state: batch=1, no arrival dynamics).
- Reuse characteristics: n/a for quality eval (no prefix reuse, no KV sharing; each request is an independent full prefill) — stated explicitly.
- ≥3 seeds on the custom-kernel anchor cell KIVI-Falcon (revision R5 — the cell with real nondeterminism risk); 1 seed elsewhere (greedy, deterministic).

# Hardware
Single A100-80GB. Why not less: A100-80GB is the primary hardware of all corpus methods (SnapKV.md §9, KIVI.md §9, SCOPE.md §9) and the hypothesis's control variable; a 24–40GB card would OOM the full-KV paired baselines at LongBench context lengths (avg 13K, up to 18K) and would make B2 (Falcon + LongBench-8) infeasible; the paired-delta design additionally requires the full-KV baseline to run without HBM-pressure slowdowns, which smaller cards would bias.

# Instrumentation Required
Per condition: official metric (F1 / ROUGE-L / EM as the task specifies); **bootstrap CI (≥1000 resamples) on the paired delta per condition, with the pre-registered ±2 metric-point non-discriminating band (revision R5)**; measured peak HBM (bytes) for compressed and full-KV runs (confounder 6 budget accounting, feeds E4's budget constraints); TTFT; seed + method-repo commit + HF version; truncation protocol logged per cell (max_seq 4096 vs untruncated, revision R4). TPOT not primary here (E3 owns it) but prefill-side selection cost appears in TTFT.

# Baseline
Full-KV cache on the identical task: same model, same split, same prompt template, same forced gen length, same seed, HF+FA2 eager fp16 (KIVI cells: FP16 full-KV, matching the papers' "16-bit" rows). Fairness: paired delta per task only (confounder 1 — never compare cross-task absolute scores); identical measured context (no truncation asymmetry between compressed and full runs); identical max_new_tokens; identical decoding (greedy).

# Experimental Conditions
Minimum runs to kill H1 (first wave, in order, revision R1/R2 applied): **SnapKV A1+A2+A3 (4 native/observation datasets + ∞Bench En.Sum + GSM8K + phase cell ≈ 12 runs) and KIVI B1+B2 (CoQA, GSM8K native = 4 runs; ∞Bench En.Sum + long-decode = 4 runs)** — 2 methods covering 2 different non-native dimensions (out-of-suite task class; generation phase), each with its own native verification; the phase cell (A3) runs unconditionally in wave 1. Second wave, only if first wave survives (no out-of-suite crossing anywhere): the observation cells (A4 GovReport, B3 Falcon) and, if the universal-fragility rule fires (native also ≥10% on the same method), the budget scan slot B4 = KIVI 4-bit on its native cells and A5 = SnapKV 4096 on its native cells (each 2 runs) — these scan slots classify "published point globally unsafe" vs "task/phase/layout-specific fragility"; they never rescue H1. Expansion beyond this only if first wave is fully negative (→ E5). Cell operating points (from hypothesis.md, fixed): SnapKV capacity 1024 (window 32, kernel 7); KIVI 2-bit G=32 R=128; KIVI Falcon cells truncated at max_seq 4096 (revision R4). Prompt templates fixed across conditions; summarization cells use "summarize the whole document" instruction-at-front templates (T8 breakdown condition, confounder 7), reported on the template dimension.

Pre-registered predictions from E1: A1 native safe (−4.8% on the same-model anchor); A2 (∞Bench En.Sum, GSM8K) and A3 no in-paper anchor → open; B1 safe (−1.3%, −5.6%); B2 (∞Bench En.Sum, long-decode) → open; B3 (Falcon observations) in-suite, reported but not H1 evidence.

# Success Criterion
**H1 true** iff, matching hypothesis.md exactly: at least one out-of-suite non-native condition (∞Bench En.Sum task class, or the long-decode generation-phase cell) shows main-metric relative drop ≥10% (PPL-type relative rise ≥15%) vs the full-KV baseline, *while the same method's native-task delta stays <10%*; and the crossing is confirmed by ≥2 datasets/conditions with bootstrap CIs outside the pre-registered ±2-point non-discriminating band (revision R5). Concretely: SnapKV ∞Bench En.Sum and/or GSM8K crossing with NrtvQA+Qasper <10%, or KIVI ∞Bench En.Sum and/or long-decode crossing with CoQA+GSM8K <10%. In-suite observation cells (GovReport, Falcon) are reported but cannot witness H1. This kills H1; the pipeline then proceeds to E3.

# Falsification Criterion
**H1 not killed by E2** (survival, not yet falsified) iff neither method shows any out-of-suite crossing while native <10%. **H1 falsified** iff no out-of-suite crossing appears across the completed matrix, OR every tested condition (native and non-native alike) drops ≥10% — universal fragility, unified classification per revision R6 (the mechanism attribution to task/phase/layout change fails; the "static setting is globally unsafe" finding is still recorded). Because H1's falsification sentence spans the full method matrix, a negative E2 does **not** falsify H1 by itself: PyramidKV/StreamingLLM/GEAR/H2O coverage is completed by E5 (which now runs unconditionally for its open-value cells — revision R3). E2's negative result triggers E5's completion parts.

# Estimated Complexity
Low (official repos + official harnesses, config-only; no code changes)

# Expected Runtime
multi-day wall (single A100; ~20 runs × 10–30 min = ~8–12 GPU-h for wave 1, plus setup/checkpoint caching; second wave +4–8 GPU-h if triggered)

---

# Experiment E3 — TPOT/kernel microbenchmark, one method first (kills H2)

# Target Hypothesis
H2: "使内存节省最大化的静态设置（低 bit 量化 + fused dequant、decode-phase 选择、周期性重压缩）在至少一个条件下使 TPOT P50 相对 full-KV 基线上升 ≥15%，或单步 kernel 开销（dequant/selection/recompression）> 解码步时间的 10%，或 measured HBM 节省 ≥2× 而吞吐优势 <5%" (at fixed memory budget, matched batch ≥8).

# Purpose
First matched-harness verdict on the performance consequence. Start with the single cheapest mechanism that the E1 triage ranks highest for kill power, then add mechanism cells only if earlier ones survive. Primary cell = KIVI 2-bit (G=32, R=128) on Llama-2-7B (MHA) and Falcon-7B (MQA): the fused-dequant mechanism is in the corpus (KIVI.md §3.3), the repos are HF-native, and the MQA cell is the hypothesis's stated dequant-overhead condition ("在 MQA layout 上 KV 本已很小，dequant 固定开销占比上升"). Conditional cells: GEAR 2-bit (n_b=20 periodic recompression — TPOT P95 spikes at the 20-step cadence, GEAR.md §4) and, only if all cheaper cells survive, SCOPE Slide (decode-phase per-step Top-K selection at λ1+λ2=2048, λ2=256 — in-paper anchor Full 36.57 vs Slide 18.28 tok/s = TPOT +100%, SCOPE.md Table 3).

# System
Small microbenchmark + instrumentation of existing implementations (cost tiers 3–4; no new kernels, no system changes): official KIVI/GEAR/SCOPE repos on HF, with torch.profiler instrumentation around the decode loop to attribute per-step time to dequant/quantize/Top-K/recompression kernels. Why cheapest sufficient: TPOT and kernel-share are measurable from the published implementations directly; no serving framework is needed (hypothesis's control environment excludes vLLM paging, and H2 is defined at the step level, not the scheduler level). SCOPE cell uses the published model LLaMA-3.1-8B-Instruct (its operating point).

# Workload
Fixed operating points from hypothesis.md: KIVI 2-bit G=32 R=128 on Llama-2-7B and Falcon-7B; GEAR 2-bit s=2% r=4 n_b=20 on Llama-2-7B; SCOPE λ1+λ2=2048, λ2=256, decode budget 512 on LLaMA-3.1-8B-Instruct. Synthetic decode sessions: context 4K, output ≥1024 tokens (long decode; error accumulation and decode-phase state), batch 8, ≥1,000 decode steps per measurement for TPOT P50/P95; batch 1 run kept as a noise check only (per pre-registered exclusion). Arrival pattern: n/a (fixed batch, no serving). Reuse characteristics: n/a (no sharing).

# Hardware
Single A100-80GB (hypothesis control); clocks locked; fixed warmup. Why not less: the hypothesis's control environment; SCOPE's published efficiency numbers use RTX 3090 but the control fixes A100 across all conditions to keep comparisons matched (T11).

# Instrumentation Required
Per condition: TPOT P50/P95 (ms/token) over ≥1,000 decode steps; kernel-share breakdown per step via torch.profiler (dequant/quantize/Top-K/recompression as fraction of decode-step time); periodic-spike detection for GEAR (TPOT time series at n_b=20 cadence); measured peak HBM (bytes) for compressed vs full-KV; throughput (tok/s); seed, versions, commits.

# Baseline
Full-KV FP16 baseline at identical batch, context, output length, and decode-step count; same model and harness. Fairness: matched batch ≥8; identical request set; only the KV representation/selection mechanism differs.

# Experimental Conditions
- A: KIVI 2-bit on Llama-2-7B (MHA) — the cheapest primary cell.
- B: KIVI 2-bit on Falcon-7B (MQA) — the stated dequant-overhead condition.
- C: GEAR 2-bit on Llama-2-7B (periodic recompression) — runs only if A/B survive.
- D: SCOPE Slide vs Discontinuous vs Full on LLaMA-3.1-8B-Instruct (decode-phase selection) — runs only if A–C survive.
Each condition measured against the full-KV baseline at the same batch.

# Success Criterion
H2 supported iff at least one condition shows: TPOT P50 ≥15% above full-KV baseline (or measured HBM saving ≥2× with throughput advantage <5%), OR per-step kernel overhead >10% of decode-step time (e.g., dequant share on the MQA cell), OR GEAR's periodic recompression produces detectable TPOT P95 spikes at the n_b cadence. Matched batch ≥8 and matched measured HBM per hypothesis.

# Falsification Criterion
H2 falsified iff at matched batch ≥8 and matched measured HBM, all conditions show TPOT P50 within 15% of full-KV, kernel overhead share <10%, and no "≥2× memory saving with <5% throughput gain" configuration exists. TPOT differences appearing only at batch 1 are excluded as batching noise (pre-registered). If H2 dies here, the pipeline proceeds only to E4 for H3 (quality-adjacent knobs), never back to H2.

# Estimated Complexity
Low

# Expected Runtime
hours (~3–8 GPU-h for A–B; C/D add ~4–6 GPU-h if triggered)

---

# Experiment E4 — Per-condition oracle re-tuning (kills H3)

# Target Hypothesis
H3: "同一方法、同一实现、同一 measured HBM 预算，仅按条件重调其自有旋钮，在至少一个条件下比固定设置获得 ≥10% 相对质量提升或 ≥15% TPOT 改善"; killed iff oracle's best setting differs from the fixed setting by <5% on both quality and latency on every condition, or the gain requires switching method family or increasing the budget.

# Purpose
Decide whether the static setting itself — not the method family — is the problem. On each method×condition that survived E2/E3, re-tune the method's OWN published knobs (SnapKV capacity/window/kernel; KIVI group size within G∈{16,32,64,128} and residual R within {32,64,96,128}; GEAR n_b and r within its published ranges; SCOPE λ1/λ2/decode-budget split and selection frequency; StreamingLLM sink count 1–8 — NOT 4-bit, which is a budget increase) at the SAME measured HBM bytes as the fixed setting, and measure the best achievable quality/TPOT. Candidate knobs pre-selected in E1 (K2 pruning): KIVI G∈{32,64} and StreamingLLM sink count on Falcon are predicted flat (<5%) and are pruned.

# System
Same HF+FA2 eager harness and official repos as E2/E3 (config-level re-tuning only; no code changes, no new kernels). Why cheapest sufficient: re-tuning is a config sweep over existing implementations; the knobs are the methods' own published parameters, so no system construction is involved.

# Workload
The conditions that survived E2/E3 (minimum: SnapKV on GovReport/QMSum, KIVI on the Falcon cells, GEAR on GSM8K; plus the anchor cell SnapKV-GovReport for the determinism check). Request counts per official split; context/output per task as in E2. Batch 8 for TPOT cells. Arrival: n/a. Reuse: n/a.

# Hardware
Single A100-80GB (hypothesis control).

# Instrumentation Required
Per (method, condition, knob setting): quality per official metric; measured peak HBM (bytes) — the budget-constraint enforcement; TPOT P50/P95 (≥1,000 steps) for latency knobs; seeds, versions, commits. Oracle selection rule (pre-registered): the best setting is the one maximizing quality at ≤ the fixed setting's measured HBM + 1% tolerance, or minimizing TPOT at ≥ the fixed setting's quality − 1% (both directions reported; the comparison uses the better of the two, declared before measurement).

# Baseline
The method's own fixed published setting at its measured HBM (same condition). Fairness: same method, same implementation, same measured budget — only the knob values differ.

# Experimental Conditions
Per method×condition: fixed setting vs 3–5 same-budget knob settings (as listed above). Budget equalization enforced by measured bytes: any knob setting that exceeds the fixed setting's measured HBM is excluded (not re-measured at a higher budget).

# Success Criterion
H3 supported iff on at least one condition, the oracle's best setting exceeds the fixed setting by ≥10% relative quality or ≥15% TPOT improvement at equal measured HBM. (Corpus anchors suggest: SCOPE phase-split re-tuning ≈ +88% on GSM8K+ at equal 35% budget; SCOPE selection-frequency Discontinuous vs Slide ≈ +42% tok/s; PyramidKV β re-allocation at equal total budget.)

# Falsification Criterion
H3 falsified iff on every condition the oracle's best setting differs from the fixed setting by <5% on both quality and latency; or the only gains require a different method family or a higher memory budget (both counted as non-evidence). A "not supported" (gains only at higher budget) is recorded as unproven, not falsified — re-test at equal budget per hypothesis.

# Estimated Complexity
Low–Medium (config sweeps over existing repos; ~20–40 extra runs)

# Expected Runtime
multi-day wall-clock if all waves run; single-day if E1 pruning is effective (KIVI G sweep pruned, StreamingLLM sink sweep pruned)

---

# Experiment E5 — Completion + open-value matrix (open cells unconditional; completion parts conditional)

# Target Hypothesis
H1 (completion of the falsification coverage), plus E3/E4 leftovers for late methods (PyramidKV, StreamingLLM, GEAR, H2O).

# Purpose
E2's first wave covers SnapKV (task class) and KIVI (generation phase) on out-of-suite conditions only. Two distinct roles (revision R3 — decoupled):
- **Open-value cells (UNCONDITIONAL — they measure things the corpus has never measured):** (a) PyramidKV KV=64 per-task LongBench native vs long-form/reasoning (in-paper only aggregate −24.6% on Mistral; per-task native values unrecorded); (b) StreamingLLM 4-sink+window on retrieval-at-evicted-distance (needle placed BEYOND the window — positions 3000/5000/10000 with window ~2048, revision R7; in-paper never tested); (c) GEAR 2-bit per-task LongBench-21 resolution (in-paper average −5.0%, per-task unresolved); (d) H2O 20% on long-form summarization (∞Bench En.Sum) and CoT (GSM8K) (no in-paper non-native data).
- **Completion cells (conditional — only if E2 negative or the universal-fragility rule fires):** (e) if native also ≥10% everywhere: a budget scan for the affected method (SnapKV 1024/2048/4096 on its own native tasks; KIVI 2/4-bit on its own native tasks) to classify universal fragility vs task-specific fragility — unified classification per revision R6 (universal fragility = H1 falsified, attribution fails).

# System
Same harness and repos as E2 (HF+FA2 eager; official method implementations at their published operating points).

# Workload
Per method: 2 native + 2 non-native datasets per the E2 conventions (official splits, official metrics, fixed operating points from hypothesis.md); StreamingLLM: PG19-style PPL + a synthetic needle-in-mid-sequence task (needles at positions 3000/5000/10000 within a rolling-window cache of total capacity 2048 — beyond the window, revision R7; 100 requests); H2O: XSUM/CNN-Daily-Mail short-summarization native pair + ∞Bench En.Sum long summarization + GSM8K 5-shot CoT. Arrival: n/a. Reuse: n/a.

# Hardware
Single A100-80GB.

# Instrumentation Required
As E2 (official metrics, bootstrap CI + non-discriminating band, measured HBM bytes, TTFT, seeds, commits, truncation protocol); plus PPL for the StreamingLLM cells.

# Baseline
Full-KV baseline per task, same model/harness (paired delta only).

# Experimental Conditions
As listed in Purpose (a)–(e); each method's own published operating point fixed; the budget scan (e) uses the method's own budget ladder (SnapKV 1024/2048/4096; KIVI 2/4-bit).

# Success Criterion
H1 supported iff ≥1 out-of-suite non-native condition crosses ≥10% relative drop (≥15% PPL rise) with native <10% on the same method, confirmed by ≥2 datasets/classes with bootstrap CIs outside the non-discriminating band. This completes the H1 kill after E2's negative first wave.

# Falsification Criterion
H1 falsified iff across the full completed matrix (E2+E5), no method shows an out-of-suite crossing with native-safe delta; i.e., "没有任何非原生条件在方法自己的公开发表预算下出现超过预声明容差的相对下降" while its native-task delta stays <10%. Crossings where native ALSO drops ≥10% are universal fragility (revision R6): H1 falsified (attribution fails), recorded as "static setting globally unsafe".

# Estimated Complexity
Medium (≈30–40 runs across 4 methods)

# Expected Runtime
multi-day (single A100, ~2–3 days wall)

---

# Execution summary

| Gate | Runs if | GPU | Kills | Proceeds if |
|---|---|---|---|---|
| E1 | always | no | (screening only) prunes cells, prunes E4 sweeps, flips posture to falsification-first | always (produces triage) |
| E2 | E1 done | 1× A100 | H1 (positive) | H1 alive |
| E3 | E2 done | 1× A100 | H2 (falsified if no condition crosses) | H2 alive |
| E4 | E2/E3 done | 1× A100 | H3 (falsified if all knobs flat) | H3 alive |
| E5 | open cells unconditional; completion parts if E2 negative/universal-fragility | 1× A100 | H1 (completes coverage; open-value measurements) | — |

Pre-registered verdicts: H1–H3 each receive a binary PASS/FAIL from the last gate that runs; ambiguous outcomes (universal fragility, single-dataset crossing, batch-1-only TPOT, budget-only oracle gains) are reported as non-discriminating per hypothesis.md, never as support. No optimization algorithm or new system is proposed anywhere in this plan; the per-condition oracle is a measurement of the method's own published knobs at equal measured budget.