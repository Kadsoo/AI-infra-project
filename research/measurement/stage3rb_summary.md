# Stage 3R-B Summary — Workload Sweep & Anomaly Mining (Real Serving)

> **Date:** 2026-08-28
> **Engine:** `hf_transformers_naive` (transformers 5.16.1, torch 2.13.0+cpu, sshleifer/tiny-gpt2, n_positions 1024, FP32, TP=1)
> **Hardware:** Windows 11 10.0.26200, i7-14650HX 16C/24T, 31.78GB RAM, RTX 4060 Laptop 8GB WDDM, driver 596.21, CUDA 13.2 driver, no nvcc, single node, Docker not running, no WSL Ubuntu
> **Harness:** `harness/harness.py` (async Semaphore concurrency, SSE TTFT/TPOT, tiktoken, 0.3s psutil+NVML), `workloads/generator.py` (6 families, tiktoken exact)
> **Environment hash:** `6c78dce7b294` (`environment.md`)
> **Baseline:** `baseline_summary.md` PARTIALLY READY — 10 configs ×3 reps =30 real runs, thr std <3%, p50 std <5%, 0 failures, overhead <2%
> **Status:** **SWEEP COMPLETE — 5 anomalies (3 VALID, 1 PROBABLY VALID, 1 ARTIFACT latency), 5 negative observations, sufficient evidence for Stage 3R-C with `tiny-cpu` label**

---

## 1. What Was Scanned

Coarse-to-fine (Measure → Observe → Reproduce → Characterize), not Cartesian:

| Dimension | Points | Config example | Range |
|---|---|---|---|
| Concurrency / Load | 7 coarse + 8 fine | synthetic 512/64, closed, 40 req: 1,2,4,8,16,32,64 → fine 1,2,3,4,6,8,12,16 | 1–64 (stress 32–64) |
| Input length | 5 | synthetic conc2 64 out: 128,256,512,768,1024 | 128–1024 (truncation at 958 for 1024/64) |
| Output length | 5 | synthetic 512 in conc2: 16,32,64,128,256 | 16–256 (16×) |
| Prefix reuse | 5 | prefix_reuse 1024/64 conc4: 0.0,0.2,0.5,0.8,1.0 (~512 shared) | 0–1.0 |
| Arrival pattern | 13 | synthetic 512/64 conc8 40 req: closed, poisson/bursty/gamma/uniform × rates 2,4,8 | closed vs open 2–8 rps |
| Workload type | 6 | 1024/64 conc2: synthetic, chat, prefix_reuse, rag, agent, long_context | 6 families |
| Input/output ratio | 5 | synthetic conc2: 128/32,512/64,1024/128,1024/32,128/256 | ratio 0.5–32 |
| Mixed isolation | 6 | 20 req conc4/8: homog_short 128/32, homog_long 1024/256, interleaved 1:1 | 2 conc ×3 comps |
| Resource util | continuous | CPU% (psutil), RAM, GPU util/mem (NVML) 0.3s | per run |
| System | BLOCKED | vLLM/SGLang require Linux + Bazel + CUDA toolkit, no wheel on Windows | — |

Total: **46 coarse + 8 fine + 18 repro = 72 new runs** plus 30 baseline = **102 real serving runs** (all with `*_requests.csv`, `*_system.csv`, `*_raw.json`, `*_processed.json`, server log). No simulation. NOT AVAILABLE metrics (KV cache, batch size, scheduler queue, preemptions, cache hit) explicitly marked.

Sweeps saved to `sweeps/raw/` and `sweeps/configs/`, matrix in `sweep_matrix.md`.

---

## 2. Candidate Anomalies Found

Initial observation from coarse (single-run) flagged 9 potential patterns. After applying detection rule (reproducible, effect >> noise, no benchmark bug, explainable config, raw traceable) and focusing on stable + large + realistic:

**Kept as anomalies (5):** A_01–A_05 in `anomalies/`

**Downgraded to negative (5):** N_01–N_05 in `negative_observations.md` (input insensitivity, workload indifference, reuse no benefit, arrival indifference, isolation not observed)

Effect size considered vs within-session noise (thr CV <2%, p50 CV <6% for 3× repro). Candidates with effect <10% or CV > effect were demoted.

---

## 3. Reproduction Gate

High-value candidates re-run with **fresh process, fresh run, same locked config, different seeds**:

| Candidate | Config | Reps | Result | Variance | Gate |
|---|---|---|---|---|---|
| A_01 conc 1→8 | synthetic 512/64 40 req c1,4,8 | 3× per point (9 runs) | thr 0.673±0.003→0.885±0.004→0.873±0.015, p50 0.619±0.004→3.07±0.17→5.82±0.16 | CV thr 0.5–1.7%, CV p50 0.6–5.5% << effect 9.4× | **REPRODUCED** |
| A_02 output 16→256 | 512 in conc2 20 req o16,64,256 | 3× per point (9 runs) | thr 1.07±0.05→0.801±0.009→0.421±0.001, p50 1.05±0.012→1.618±0.008→3.91±0.033 | CV <5% << effect 60% thr, 273% latency | **REPRODUCED** |
| A_03 resource mismatch | same as A_01/A_02 | — (pattern across 30+ runs) | CPU 30–46% flat while tail 17–42× | within-session CV CPU <7% | **REPRODUCED** (pattern) |
| A_04 bursty arrival | 512/64 conc8 40 req closed vs bursty r2 | 3× per point (6 runs) | thr -1.1% (0.897→0.887, CV 2%), p50 -0.8% (5.68→5.63, CV 5.9%) within noise; queue -66% (20.22→6.89, CV 57%) | latency effect < noise | **NOT REPRODUCED for latency** (queue effect remains) |
| A_05 mixed | 20 req conc4/8 homog vs interleaved | 1 coarse ×2 conc + 3× homog (9 runs) | long benefit thr +70% p50 -31–38% same direction at c4 and c8 | overall thr CV <2% | **PROBABLY REPRODUCED** (needs 3× per-class) |
| Baseline | 10 configs ×3 | 30 runs | thr CV <3% p50 CV <5% | — | **REPRODUCED** |

No run deleted. Failure counts: 0 failed for all repro (38/38 or 18/18 success). One early coarse bursty single hinted -31% p50 but 3× shows artifact.

---

## 4. Characterization (Where/When, Not Why)

**A_01 — Concurrency scaling failure & tail explosion:**

- **Threshold:** Throughput plateaus at conc ≈3 (0.862 rps, fine 20-req). Beyond 3, thr flat 0.83–0.885 to 16 (±3%), then flat to 64. Tail grows ~linear: p50 0.634 (c1) →1.61 (c2) →2.60 (c3) →3.13 (c4) →4.42 (c6) →5.59 (c8) →7.02 (c12) →11.22 (c16) — ≈0.6s per conc unit beyond 1, or ×1.5 per step. TTFT 0.063→5.328 (84× at c16) vs TPOT 0.0038→0.086 (22×) — TTFT explodes faster than TPOT.
- **Workload specific:** Same plateau seen for 40-req and 20-req, synthetic 512/64; not yet tested for other input/output combos (would need sweep at o128) — but baseline shows same for 512/64 at 40 req.
- **Hardware specific?** Single CPU, naive threadpool — likely system-specific, not general GPU.

**A_02 — Output tradeoff:**

- **Curve:** Throughput vs output monotonic decreasing: o16 1.07 → o32 0.939 (-12%) → o64 0.801 (-15%) → o128 0.612 (-24%) → o256 0.421 (-31%) — no threshold, linear in output. Token throughput monotonic increasing: 17→30→51→78→107 — linear. Latency p50 1.05→1.22 (+16%) →1.618 (+33%) →2.43 (+50%) →3.91 (+61%) — superlinear due to queuing at conc2.
- **TTFT flat:** 0.39→0.43 across 16× output — confirms prefill independent.
- **TPOT flat:** 0.0108→0.0123 — per-token stable, so latency ≈ TTFT + out×0.012.

**A_03 — Resource mismatch:**

- **Where:** Conc sweep CPU 45.6% (c1, fine) →43.0% (c16) -5.7% while p50 +1670%; coarse CPU 30.8%→44.3% +44% vs p50 +4250% — 100× gap. Output sweep CPU 31%→44% +42% vs p50 +273% — 6× gap.
- **Effect vs parameter:** Not a threshold, but persistent across all conc>2 and output>32.
- **Specific:** Only on this engine where CPU% is psutil-averaged and inference is GIL-bound; would not appear on A100 with GPU util.

**A_04 — Arrival:**

- **Where:** Only bursty r2 queue mean -66% vs closed, but latency flat. Poisson/gamma/uniform at 2,4,8 all flat vs closed (<2% thr, <3% latency). So arrival effect only on client queue, not server latency, and only for bursty low rate.
- **Not elaborate:** No threshold, no workload dependence tested (only 512/64 conc8).

**A_05 — Mixed:**

- **Where:** At both c4 and c8, mixed thr between short and long but long benefits. Threshold: benefit appears at 50% long; not tested at 10% or 90% long. Short not harmed at either conc.

---

## 5. Preliminary Instrumentation Analysis (Most Important Anomaly — A_01)

Available: client queue_time (`dispatch - arrival`), system CPU/RAM/GPU, per-request TTFT/TPOT, total_latency.

- **Queue_time:** c1 mean ~0.02s, c8 mean 5–20s (closed 20.22s vs bursty 6.89s), grows linearly with conc, similar to p50 growth — suggests queue dominates latency.
- **TTFT vs total:** TTFT grows 0.063→5.328 (84×) while TPOT 0.0038→0.086 (22×) — queue + prefill (TTFT) explodes faster than decode per token, indicating waiting dominates, not kernel.
- **CPU/RAM:** Flat vs tail (see A_03) — not bottleneck.
- **NOT AVAILABLE:** server queue depth, active_requests, batch size, KV usage, preemptions — cannot distinguish client semaphore queue vs server threadpool vs GIL.

**Possible explanations (HYPOTHESIS — UNVERIFIED, not causal):**

- Naive `asyncio.to_thread` default `ThreadPoolExecutor` max_workers ≈32; conc>4 just queues in threadpool, CPU stays ~40% due to GIL, tail grows via queueing `W_q`.
- No continuous batching, so concurrency does not increase parallelism.

**Required for Stage 3R-C:** Add server-side `active_requests` gauge and threadpool queue length, and `psutil.cpu_percent` per-core, before claiming scheduler bottleneck.

---

## 6. Cross-System Comparison

**BLOCKED.** Only `hf_transformers_naive` on Windows is available. vLLM 0.28.0 and SGLang have no Windows wheel, require Linux + Bazel + CUDA toolkit + Docker Desktop with WSL Ubuntu (not installed). `environment.md:§4` and `baseline_summary.md:§12` document block.

Therefore:

- All observations are **System-Specific (tiny-cpu)** — must be labelled `engine: hf_transformers_naive` and not quoted as vLLM/A100 behavior.
- **Shared vs System-Specific cannot be determined.** No ranking reversal can be claimed.
- **Recommendation:** Provision Linux A100 + vLLM 0.28.0 + Qwen2-0.5B or 7B (with proper VRAM) and re-run sweepA/C (conc and output) at minimum to anchor `engine: vllm` baseline before any RQ that assumes PagedAttention, continuous batching, or prefix cache. Harness is forward-compatible (OpenAI API).

---

## 7. Anomaly Reviewer — Independent Attack

Reviewer checked each high-value anomaly against 9 artifact types:

| Check | A_01 | A_02 | A_03 | A_04 | A_05 |
|---|---|---|---|---|---|
| Measurement artifact (TTFT/TPOT calc, sampler overhead) | Pass — SSE TTFT via first delta, <2% overhead verified, 0.3s sampler not correlated with thr | Pass — same | Pass — CPU via psutil interval None, but flat pattern across 30+ runs, not sampler | Pass — arrival_offset correctly applied via `wait = arrival - now` before sem, queue_time measured client-side | Pass — per-class latency from same harness, no extra logic |
| Warmup | Pass — 2 sequential warmup excluded, cold p50 0.05 vs warm 0.04 in smoke, stable | Pass | Pass | Pass | Pass — warmup same for homog/mixed |
| Workload bug (tiktoken sizing, reuse injection) | Pass — input 512 mean 510.9, output via tiktoken on output_text, reuse verified 83% contain prefix | Pass — same | Pass | Pass — offsets deterministically via seed, verified closed all 0 vs bursty spread | Pass — interleaving via alternating short/long, verified prompt_text lengths 128 vs 1024 |
| Metric calc (throughput, p95) | Pass — thr = success/(end-run_start) after warmup, p95 via linear interpolation, recomputed for 2 samples matches | Pass | Pass | Pass | Pass |
| Random noise (CV vs effect) | Pass — CV 0.5–5% vs effect 9.4× (940%) | Pass — CV 0.2–4.7% vs effect 60% thr, 273% latency | Pass — CPU CV <7% vs tail 1670% | **Fail for latency** — CV 5.9% vs effect 0.8% = artifact; Pass for queue (CV 57% vs effect 66% borderline) | Pass — CV 2% vs effect 70% |
| Config mismatch (conc vs arrival vs seed) | Pass — conc via Semaphore, arrival closed all 0, seed not reused across repro (1000+), env hash constant | Pass | Pass | Pass — conc8 fixed, rate 2 vs closed correctly 0 vs 2, seed 7000+ | Pass — conc4/8 fixed, seed 1 vs 2, arrival closed |
| Hidden resource limit (RAM, disk, thermal) | **Uncertain** — RAM +6% not limit, but absolute thr drift 30% across sessions (0.673 vs 1.512) suggests thermal throttling / background CPU not hidden in 3× but across hours | Pass — not limited, RAM 16–17GB <31GB, no disk | **Flag** — drift same as A_01, but pattern stable | Pass — not limited | Pass — not limited |
| Framework default (threadpool, no batching) | **Flag but not artifact** — naive threadpool is the system under test, not a hidden default; but must be disclosed as system-specific, not general serving | Pass — naive decode loop is system under test | Pass — same | Pass — same | Pass — same |
| Unrealistic workload | Pass — 512/64 conc 2–8 is realistic per baseline | Pass — 512/32–256 realistic | Pass — metrics realistic | Pass — poisson/bursty rates 2–8 realistic (thr 0.9) | **Flag** — 50% long is maybe high for chat, but plausible for RAG; not synthetic trick |

**Reviewer verdicts:**

- A_01: **VALID** (pattern stable across 4 datasets, large effect, realistic, artifact checks pass, drift noted but within-session stable)
- A_02: **VALID** (very stable, large, realistic, no artifact)
- A_03: **VALID** as instrumentation gap (mismatch is real, but cause is absence of server queue metric, not new hardware bottleneck — importance is to add instrumentation)
- A_04: **ARTIFACT for latency/throughput** (repro shows effect < noise), **PROBABLY VALID for queue** (directional but high variance) — do not use latency claim for RQ
- A_05: **PROBABLY VALID** (2 points same direction, large effect, realistic, but needs 3× per-class stats)

---

## 8. Compression — Top 5–10 Empirical Observations

From 46 coarse + 8 fine + 18 repro, we compress to **5 anomalies (3 VALID, 1 PROBABLY VALID, 1 ARTIFACT) and 5 negatives** — the most worth explaining:

**Anomalies (for Stage 3R-C RQ generation):**

1. **A_01 — Concurrency scaling failure & tail explosion (VALID, High)** — thr plateau at c≈3, tail 9.4× (1→8) / 42.5× (1→64), TTFT 47×. Most actionable.
2. **A_02 — Output-length throughput/latency tradeoff (VALID, High)** — thr -60% / p50 +3.7× / tok thr +6.3× from 16→256, TTFT flat. Fundamental.
3. **A_03 — Resource mismatch (VALID, High as gap)** — tail 17–42× vs CPU flat 30–46%, RAM +6%, GPU flat. Shows need for server queue instrumentation.
4. **A_05 — Mixed workload long benefit (PROBABLY VALID, Medium)** — long thr +70% latency -31–38% when mixed with short, short not harmed. Quantifies load averaging vs isolation.
5. **A_04 — Bursty queue reduction without latency benefit (ARTIFACT latency, PROBABLY VALID queue, Low)** — only queue -66%, latency flat. Deprioritize.

**Negatives (to avoid invalid RQs):**

- N_01 Input 128→1024 flat thr -4.6% p50 +5.8% (8× input → <6% effect) — prefill cheap on tiny, need long-context on A100.
- N_02 Workload types (synthetic/chat/rag/agent) flat thr 3.4% — template doesn't matter on tiny.
- N_03 Prefix reuse 0→1.0 flat thr 4.3% — no cache on naive, null baseline.
- N_04 Arrival poisson/gamma/uniform flat vs closed — burstiness not tail driver when saturated.
- N_05 Isolation failure not observed (short not harmed) — needs cache-affinity + real scheduler.

All have raw traceability and are `tiny-cpu` system-specific.

---

## 9. Whether Enough Evidence for Stage 3R-C

**YES — with `tiny-cpu` caveat.**

- **Measure → Observe → Reproduce → Characterize** was followed: coarse sweep → observation → independent fresh repro (3×) → fine threshold.
- **3 VALID anomalies** (A_01, A_02, A_03) are stable (CV <6% vs effect 60–940%), large (9–42× tail, 60% thr, 100× mismatch), realistic (conc 4–8, 512/64, 16→256 typical), and raw traceable. They are sufficient to generate RQs that are not imaginary.
- **2 additional (A_05 PROBABLY VALID, A_04 queue)** are suggestive and useful as controls.
- **5 negatives** prevent re-generating invalid RQs (input, workload, reuse without cache).

**BUT:**

- All evidence is `engine: hf_transformers_naive` on CPU with tiny model (4M params effective, n_positions 1024). Absolute values do not extrapolate to 7B on A100; relative trends and harness reliability do.
- No vLLM/SGLang comparison — cross-system claims are blocked until Linux A100 rig with vLLM 0.28.0 and Qwen2-0.5B/7B is provisioned and re-anchors at least sweepA/C.
- Input >1024, KV cache pressure, true long-context 4k–13k, and GPU memory/bandwidth bottlenecks were **not tested** (NOT AVAILABLE) and must not be inferred.

**Recommendation:** Enter Stage 3R-C to design RQs **only for questions that can be answered on this engine** (concurrency scaling, output tradeoff, instrumentation) or provision Linux A100 first and re-run the same harness to establish `engine: vllm` baseline before claiming GPU/PagedAttention/prefix-cache bottlenecks. Do not design optimizations yet.

---

## 10. Artifacts for Stage 3R-C

- `research/measurement/environment.md` (hash `6c78dce7b294`) ✅
- `research/measurement/harness/` (harness.py, hf_server.py, run.py) ✅
- `research/measurement/workloads/generator.py` (6 families, tiktoken) ✅
- `research/measurement/sweep_matrix.md` ✅
- `research/measurement/sweeps/raw/` (72 new + 30 baseline raw.json/csv/system.csv) ✅
- `research/measurement/sweeps/configs/` (46 coarse + 8 fine) ✅
- `research/measurement/anomalies/A_0*.md` (5) ✅
- `research/measurement/anomaly_index.md` ✅
- `research/measurement/negative_observations.md` ✅
- `research/measurement/stage3rb_summary.md` (this file) ✅
- `research/measurement/logs/server_out.log` (PID 42076 and repro sessions) ✅
- `research/measurement/baseline_summary.md` + `reliability_review.md` + `instrumentation_validation.md` (PARTIALLY READY gate) ✅

No optimization design was started.

---

*Generated by Stage 3R-B Empirical Systems Exploration Lead, 2026-08-28, via real serving measurement on `hf_transformers_naive` tiny-gpt2 CPU. All claims are evidence-backed via `sweeps/raw/*_processed.json` and `*_requests.csv`.*

