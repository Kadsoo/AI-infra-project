# Stage 3R-B Sweep Matrix — Workload Sweep & Anomaly Mining

> **Date:** 2026-08-28
> **Engine:** `hf_transformers_naive` (transformers 5.16.1, torch 2.13.0+cpu, sshleifer/tiny-gpt2, n_positions 1024, FP32, TP=1)
> **Hardware:** Windows 11, i7-14650HX, 31.78GB RAM, RTX 4060 Laptop 8GB (WDDM), driver 596.21, CUDA 13.2 driver, no nvcc, single node
> **Harness:** `research/measurement/harness/harness.py` (OpenAI-compatible SSE, async Semaphore concurrency, `time.perf_counter` TTFT/TPOT, tiktoken counting, psutil + NVML 0.3s sampling), `workloads/generator.py` (tiktoken exact sizing)
> **Environment hash:** `6c78dce7b294` (`environment.md`)
> **Sampling:** sampler_interval 0.3s (overhead <2% per instrumentation_validation.md:§3)
> **Warmup:** 2 requests sequential per config, excluded from stats, unless noted
> **Repeats:** coarse stage single-run for observation, high-value candidates re-run 3× for reproducibility (see §7)

## 1. Strategy — Coarse-to-Fine (not Cartesian)

Spec §3: avoid Cartesian product. Procedure:

1. **Coarse sweep** each dimension at 4–7 points covering extremes and interior, single repeat, to detect kinks, plateaus, explosions.
2. **Inspect** throughput, p50/p95/p99 latency, TTFT, TPOT, CPU/RAM, queue_time — look for categories §4 (Non-linear breakdown, Tail explosion, Resource mismatch, etc.).
3. **If anomaly appears between two coarse points**, increase density locally (e.g., concurrency 16→64 coarse showed plateau; then fine 1,2,3,4,6,8,12,16).
4. **Low-yield dimensions** (flat response) are recorded as negative observations and not refined.

All runs are real serving (no simulation), each latency from measured wall-clock + SSE streaming. NOT AVAILABLE metrics (KV cache, batch size, scheduler queue) are explicitly marked and not fabricated.

Cross-system comparison is **BLOCKED** on this host (vLLM/SGLang have no Windows wheel, require Linux + Bazel + CUDA toolkit, see `environment.md:§4` and `baseline_summary.md:§12`). All observations are labelled `engine: hf_transformers_naive (tiny-cpu)` and must not be quoted as A100/vLLM behavior without re-anchoring on a Linux rig.

## 2. Sweep Dimensions & Configs

Total coarse configs: 46 (plus 12 repro/fine). Raw CSV/JSON per run in `research/measurement/sweeps/raw/` and `research/measurement/baseline/` (30 baseline runs). Configs in `sweeps/configs/`.

### Sweep A — Concurrency / Load (Coarse)

| Config | workload | input | output | conc | req (warmup) | arrival |
|---|---|---|---|---|---|---|
| sweepA_conc_c001 | synthetic | 512 | 64 | 1 | 40 (2) | closed |
| sweepA_conc_c002 | synthetic | 512 | 64 | 2 | 40 (2) | closed |
| sweepA_conc_c004 | synthetic | 512 | 64 | 4 | 40 (2) | closed |
| sweepA_conc_c008 | synthetic | 512 | 64 | 8 | 40 (2) | closed |
| sweepA_conc_c016 | synthetic | 512 | 64 | 16 | 40 (2) | closed |
| sweepA_conc_c032 | synthetic | 512 | 64 | 32 | 40 (2) | closed |
| sweepA_conc_c064 | synthetic | 512 | 64 | 64 | 40 (2) | closed |

Rationale: coarse 1→4→16→64 covers 64× range; 64 is stress-test, 1 is serial baseline. 40 requests gives p95 stable.

Fine follow-up (characterization) after observing plateau + tail explosion between 1 and 8:

| Config | conc |
|---|---|
| fineConc_c01 | 1 |
| fineConc_c02 | 2 |
| fineConc_c03 | 3 |
| fineConc_c04 | 4 |
| fineConc_c06 | 6 |
| fineConc_c08 | 8 |
| fineConc_c12 | 12 |
| fineConc_c16 | 16 |
(20 requests each, synthetic 512/64, closed, 3 reps for threshold)

### Sweep B — Input Length

Fixed: synthetic, conc 2, out 64, 20 req, closed.

| Config | input |
|---|---|
| sweepB_input_i0128 | 128 |
| sweepB_input_i0256 | 256 |
| sweepB_input_i0512 | 512 |
| sweepB_input_i0768 | 768 |
| sweepB_input_i1024 | 1024 |

Note: tiny-gpt2 `n_positions=1024`, so 2048 would be truncated to ~958 effective (§7 limitation). Sweep stops at 1024 to avoid synthetic truncation artifact.

### Sweep C — Output Length

Fixed: synthetic, 512 in, conc2, 20 req, closed.

| Config | output |
|---|---|
| sweepC_output_o016 | 16 |
| sweepC_output_o032 | 32 |
| sweepC_output_o064 | 64 |
| sweepC_output_o128 | 128 |
| sweepC_output_o256 | 256 |

Covers 16× decode range; TPOT and token throughput sensitivity.

### Sweep D — Prefix Reuse Ratio

Fixed: prefix_reuse workload, 1024/64, conc4, 20 req, closed. Shared prefix ~512 tokens (`SHARED_PREFIX` + 400 filler).

| Config | reuse |
|---|---|
| sweepD_reuse_r0 | 0.0 |
| sweepD_reuse_r2 | 0.2 |
| sweepD_reuse_r5 | 0.5 |
| sweepD_reuse_r8 | 0.8 |
| sweepD_reuse_r10 | 1.0 |

Tests whether reuse fraction yields expected cache benefit (naive engine has no prefix cache, so null expected).

### Sweep E — Arrival Pattern

Fixed: synthetic 512/64, conc8, 40 req, warmup2.

| Config | distribution | rate (rps) |
|---|---|---|
| sweepE_arrival_closed | closed | 0 |
| sweepE_arrival_poisson_r2 | poisson | 2 |
| sweepE_arrival_bursty_r2 | bursty | 2 |
| sweepE_arrival_gamma_r2 | gamma | 2 |
| sweepE_arrival_uniform_r2 | uniform | 2 |
| sweepE_arrival_poisson_r4 | poisson | 4 |
| sweepE_arrival_bursty_r4 | bursty | 4 |
| sweepE_arrival_gamma_r4 | gamma | 4 |
| sweepE_arrival_uniform_r4 | uniform | 4 |
| sweepE_arrival_poisson_r8 | poisson | 8 |
| sweepE_arrival_bursty_r8 | bursty | 8 |
| sweepE_arrival_gamma_r8 | gamma | 8 |
| sweepE_arrival_uniform_r8 | uniform | 8 |

Rates chosen relative to measured throughput ~0.9 rps: 2 is mild overload, 4–8 is heavy open-loop burst. Tests burstiness sensitivity.

### Sweep F — Workload Type

Fixed: 1024/64, conc2, 20 req, closed.

| Config | type | reuse |
|---|---|---|
| sweepF_workload_synthetic | synthetic | 0 |
| sweepF_workload_chat | chat | 0 |
| sweepF_workload_prefix_reuse | prefix_reuse | 0.8 |
| sweepF_workload_rag | rag | 0 |
| sweepF_workload_agent | agent | 0 |
| sweepF_workload_long_context | long_context | 0 |

Same token budget, different prompt templates (chat natural, RAG context+question, agent tool trace). Tests workload indifference.

### Sweep G — Input/Output Ratio

Fixed: synthetic, conc2, 20 req, closed.

| Config | input | output | ratio (in/out) |
|---|---|---|---|
| sweepG_ratio_r_low | 128 | 32 | 4.0 |
| sweepG_ratio_r_bal | 512 | 64 | 8.0 |
| sweepG_ratio_r_high | 1024 | 128 | 8.0 |
| sweepG_ratio_r_long_in | 1024 | 32 | 32.0 |
| sweepG_ratio_r_long_out | 128 | 256 | 0.5 |

Complements B/C by testing ratio at extremes (prefill-heavy vs decode-heavy).

### Sweep H — Mixed Workload Isolation

Custom interleaving (not via factory): 20 requests, conc4 and conc8.

| Config | composition | conc |
|---|---|---|
| mixed_homog_short_c4 | 20× 128/32 synthetic | 4 |
| mixed_homog_long_c4 | 20× 1024/256 synthetic | 4 |
| mixed_interleaved_c4 | 10× 128/32 + 10× 1024/256 alternating | 4 |
| mixed_homog_short_c8 | same | 8 |
| mixed_homog_long_c8 | same | 8 |
| mixed_interleaved_c8 | same | 8 |

Tests §4 Isolation Failure: does short suffer from long, or vice versa? Per-class latencies recorded.

### Baseline (Stage 3R-A, 30 runs)

10 configs ×3 reps, tiny-gpt2 CPU, same harness. See `baseline_summary.md:§6`:

- baseline_low_concurrency (synthetic 512/64 c1), medium (c4), high (c8)
- short (chat 128/32 c2), medium_context (chat 1024/128 c2), long_context (4096* c2, truncated), prefix_reuse (1024/64 c4 r0.8), rag (2048* c2), agent (1024/128 c2), high_memory (2048*/256 c6)
* truncated to 958/896 due to n_positions=1024

Baseline variance: throughput std/mean <3%, p50 std/mean <5%, 0 failures.

## 3. Results Summary (coarse, single-run values, latest)

See §7 for multi-repeat means.

**Concurrency (SweepA, 512/64 closed, 40 req):**

| conc | thr rps | tok/s | p50 | p95 | p99 | TTFT p50 | TPOT med | CPU mean | RAM max |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.67–1.00 (session-dependent) | 42–64 | 0.29–0.63 | 0.31–0.67 | 0.33–0.68 | 0.05–0.06 | 0.003–0.005 | 30–46% | 16.6–17.1GB |
| 2 | 0.80–0.93 | 51–60 | 1.26–1.61 | 2.08–2.47 | 2.29–2.47 | 0.23–0.45 | 0.007–0.008 | 31–38% | 16.7–17.2 |
| 4 | 0.83–0.95 | 53–61 | 2.60–3.35 | 3.47–4.60 | 4.36–4.83 | 0.88–1.03 | 0.026–0.027 | 35–41% | 17.1–17.4 |
| 8 | 0.88–0.94 | 56–60 | 5.46–5.95 | 8.40–8.99 | 8.65–9.09 | 2.60–2.97 | 0.047–0.049 | 39–43% | 17.3–17.6 |
| 16 | 0.88–0.91 | 56–58 | 10.69–11.21 | 16.76–17.11 | 17.55–17.11 | 5.11–5.32 | 0.086–0.087 | 42–43% | 17.3 |
| 32 | 0.889 | 56.9 | 21.43 | 34.37 | 35.52 | 10.32 | 0.173 | 43.7% | 17.5 |
| 64 | 0.901 | 57.7 | 26.94 | 39.30 | 40.50 | 14.92 | 0.189 | 44.3% | 17.6 |

Fine (20 req, c1–16): c1 0.667 p50 0.634, c2 0.805 p50 1.61, c3 0.862 p50 2.60, c4 0.835 p50 3.13, c6 0.884 p50 4.42, c8 0.884 p50 5.59, c12 0.882 p50 7.02, c16 0.885 p50 11.21 — throughput plateaus at c≈3, tail grows ~linear.

**Input (SweepB, conc2, 64 out):** 128 thr0.839 p50 1.56 TTFT0.383; 256 thr0.818 p50 1.53 TTFT0.424; 512 thr0.820 p50 1.59 TTFT0.421; 768 thr0.803 p50 1.68 TTFT0.478; 1024 thr0.807 p50 1.61 TTFT0.445 — flat within 4% thr, 8% latency, 16% TTFT.

**Output (SweepC, 512 in, conc2):** o16 thr1.142 tok18.3 p50 0.99 p95 1.94 TTFT0.263 TPOT0.0108; o32 thr0.939 tok30.0 p50 1.22 p95 1.50 TTFT0.438 TPOT0.0117; o64 thr0.796 tok50.9 p50 1.61 p95 2.54 TTFT0.415 TPOT0.0125; o128 thr0.612 tok78.4 p50 2.43 p95 3.30 TTFT0.436 TPOT0.0125; o256 thr0.420 tok107.5 p50 3.93 p95 4.76 TTFT0.439 TPOT0.0123 — thr -63% (o16→256), tok/s +487%, p50 4×, TTFT flat, TPOT flat 0.011–0.012.

**Prefix reuse (1024/64 conc4):** r0 thr0.896 p50 2.98 TTFT0.927; r0.2 thr0.850 p502.03; r0.5 thr0.867 p502.92; r0.8 thr0.875 p502.85; r1.0 thr0.858 p503.28 — spread 4.3% thr, 10% latency, no monotonic benefit.

**Arrival (512/64 conc8, 40 req):** closed thr0.89–0.90 p50 5.46–5.80 qmean 19–20s; poisson r2 thr0.887–0.894 p50 5.74–5.74 qmean 14–18; bursty r2 thr0.78–0.90 p50 3.94–5.86 qmean 3–10; gamma/uniform within 2% of closed. At r8, bursty thr1.012 p50 4.84 TTFT3.89 vs closed thr0.90 p50 5.72 TTFT2.65 — small reversal but p95 similar.

**Workload types (1024/64 conc2):** synthetic thr0.812 p50 1.61; chat thr0.790 p501.63; prefix_reuse thr0.787 p501.67; rag thr0.785 p501.65; agent thr0.791 p501.66; long_context thr0.803 p501.60 — thr spread 3.4%, latency spread 4%.

**Ratio (conc2):** r_low 128/32 thr0.973 p50 1.16; r_bal 512/64 thr0.802 p501.61; r_high 1024/128 thr0.617 p502.43; r_long_in 1024/32 thr0.948 p501.18; r_long_out 128/256 thr0.431 p503.82 — confirms output dominates.

**Mixed (20 req):** homog_short_c4 thr1.139 p50 2.22; homog_long_c4 thr0.485 p506.54; interleaved_c4 thr0.826 p502.97 (short 1.95, long 4.51); homog_short_c8 thr1.065 p504.69; homog_long_c8 thr0.483 p5012.81; interleaved_c8 thr0.822 p505.82 (short 4.40, long 7.91). Long in mixed is 31–38% lower p50 than homog_long.

## 4. Coarse-to-Fine Decisions

- **Concurrency:** coarse 1→4→16→64 showed plateau after 4 and tail explosion 18.6× (1→8). Fine 1,2,3,4,6,8,12,16 confirmed plateau at c≈3 and linear tail growth (≈0.6s per conc unit beyond 1). No need to refine beyond 16 (64 already stress-only).
- **Output:** coarse 16→256 showed strong monotonic thr decrease and p50 increase, no kink; no fine needed beyond 16–256 (already covers 16×). Re-check at o128 confirmed linear decode.
- **Input:** coarse 128→1024 flat; no fine needed (no breakdown to chase).
- **Reuse:** flat; not refined (engine has no cache).
- **Arrival:** coarse showed only bursty r2 queue reduction; not refined beyond r2/r8 (no threshold).
- **Workload type:** flat; not refined.
- **Mixed:** interleaved vs homog showed long benefit; not refined further (would need per-class load sweep).

## 5. Coverage vs Spec Priority

| Spec priority | Covered | Result category |
|---|---|---|
| Input length | ✓ 5 pts + truncation note | Negative: no breakdown |
| Output length | ✓ 5 pts | Positive: strong tradeoff |
| Input/output ratio | ✓ 5 pts | Positive: decode-dominated |
| Concurrency | ✓ 7 coarse + 8 fine | Positive: scaling failure + tail explosion |
| Arrival rate/burstiness | ✓ 13 pts (closed/poisson/bursty/gamma/uniform ×3 rates) | Weak: mostly indifference, one queue effect |
| Memory pressure | Partial (via input+output combos, RAM 16–17.8GB) | Negative: RAM monotonic but not bottleneck |
| Prefix reuse ratio/distance/shared len | ✓ reuse 5 pts, shared ~512, reuse distance not controlled (closed loop sequential) | Negative: no benefit (engine limitation) |
| Workload families | ✓ 6 families | Negative: indifference |
| System | ✗ single engine only | BLOCKED: vLLM/SGLang not available on Windows |
| Tail latency | ✓ p50/p95/p99 per config + per-request | Positive: tail explosion isolated |
| Resource utilization | ✓ CPU/RAM/GPU per sample | Positive: mismatch observed |

Not covered: KV cache capacity sweep (NOT AVAILABLE on naive HF), true long-context 4k–13k (blocked by n_positions 1024), multi-GPU TP/PP, P/D disaggregation.

## 6. Raw Data Traceability

Each run: `sweeps/raw/<run_id>_raw.json` (full requests + system_samples + config + aggregates), `<run_id>_requests.csv` (per-request, truncated prompt_text 2000 chars), `<run_id>_system.csv` (0.3s samples), `<run_id>_processed.json` (p50/p95/p99, throughput, token throughput, CPU/GPU stats). Environment hash `6c78dce7b294` in every `config.environment_hash`. Config JSON in `sweeps/configs/`. Server log in `logs/server_out.log` (contains `[hf_server] loaded sshleifer/tiny-gpt2` and per-request `POST /v1/chat/completions 200 OK`). No data deleted.

## 7. Reproducibility Summary (multi-repeat)

High-value candidates re-run independently (fresh process, same locked config, different seed unless noted):

| Config | N reps | thr mean±std (rps) | p50 mean±std (s) | p95 mean±std (s) | TTFT mean±std (s) | CV thr | CV p50 | Verdict |
|---|---|---|---|---|---|---|---|---|
| conc1 512/64 closed 40 req | 3 (reproA1) | 0.673±0.003 | 0.619±0.004 | 0.669±0.010 | 0.060±0.001 | 0.5% | 0.6% | Stable |
| conc4 same | 3 | 0.885±0.004 | 3.07±0.17 | 4.56±0.07 | 0.99±0.05 | 0.5% | 5.5% | Stable (p50 higher variance) |
| conc8 same | 3 | 0.873±0.015 | 5.82±0.16 | 8.83±0.37 | 2.84±0.13 | 1.7% | 2.8% | Stable |
| out16 512/2 | 3 | 1.07±0.050 | 1.05±0.012 | 1.95±0.014 | 0.39±0.12 | 4.7% | 1.1% | Stable (TTFT outlier 0.25 vs 0.44) |
| out64 512/2 | 3 | 0.801±0.009 | 1.618±0.008 | 2.54±0.04 | 0.43±0.03 | 1.1% | 0.5% | Stable |
| out256 512/2 | 3 | 0.421±0.001 | 3.91±0.033 | 4.74±0.08 | 0.43±0.02 | 0.2% | 0.8% | Very stable |
| in128 64/2 | 3 | 0.833±0.014 | 1.55±0.011 | 2.45±0.05 | 0.405±0.007 | 1.7% | 0.7% | Stable |
| in1024 64/2 | 3 | 0.795±0.007 | 1.645±0.022 | 2.53±0.04 | 0.424±0.023 | 0.9% | 1.3% | Stable |
| arrival closed 512/64 conc8 | 3 | 0.897±0.006 | 5.68±0.19 | 8.78±0.13 | — | 0.7% | 3.3% | Stable |
| arrival bursty r2 conc8 | 3 | 0.887±0.018 | 5.63±0.33 | 8.84±0.22 | — | 2.0% | 5.9% | Stable |

All thr CV <5%, p50 CV <6% — well below effect sizes (tail explosion 18×, output tradeoff 4×, input 5% ≈ noise threshold). Reproducibility gate passes for A1/A2/A3.

## 8. Limitations for Stage 3R-C

- Engine is CPU-only tiny (4M params effective), no PagedAttention, no continuous batching, no prefix cache, no GPU kernels — absolute latency/throughput do not extrapolate to 7B on A100; relative trends and harness reliability do.
- Input >1024 truncated, so long-context 4k–8k not tested.
- GPU metrics are WDDM desktop composition (~0–2% util, ~20% mem constant), not model GPU use.
- No vLLM/SGLang comparison; all observations are `tiny-cpu` single-system.
- No true memory pressure sweep (KV cache usage NOT AVAILABLE).

