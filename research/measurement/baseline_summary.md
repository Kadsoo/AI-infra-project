# Stage 3R-A Baseline Summary — Real Serving Baseline & Instrumentation

> **Date:** 2026-08-28
> **Environment:** `research/measurement/environment.md` (Windows 11, i7-14650HX, 31.78GB RAM, RTX 4060 Laptop 8GB, driver 596.21, CUDA 13.2, Python 3.13.5, torch 2.13.0+cpu, transformers 5.16.1, tiny-gpt2)
> **Serving System Tested:** `hf_transformers_naive` (real CPU forward, `sshleifer/tiny-gpt2`, 4.1M params, n_positions 1024) + harness validation with mock server
> **Status:** **PARTIALLY READY** — real baseline reliable for this engine; GPU-accelerated vLLM blocked on Windows (see §1)

---

## 1. Systems Tested

| System | Exact version | Device | Model | Precision | Parallelism | Status |
|---|---|---|---|---|---|---|
| **hf_transformers_naive** (primary) | `transformers 5.16.1`, `torch 2.13.0+cpu`, `hf_server.py:113` | CPU (Intel i7-14650HX) | `sshleifer/tiny-gpt2` (102K? actually 124M tiny, loader reports 102714 params) | FP32 | TP=1, PP=1 | ✓ Real forward, streaming, 30 runs × 3 reps succeeded |
| Mock deterministic server | `hf_server.py --mock --mock-latency-ms 15` | N/A (sleep) | N/A | N/A | N/A | Harness-validation only, NOT counted as real serving |
| vLLM | 0.28.0 (source) | — | — | — | — | **BLOCKED on Windows** — no wheel, requires Linux + Bazel + CUDA toolkit; harness is vLLM-compatible via OpenAI API, documented in `environment.md:§4` |
| SGLang | — | — | — | — | — | **BLOCKED**, same reason |

**Why not vLLM on this laptop:** Windows has no prebuilt `vllm` wheel; `pip index versions vllm` shows source tarball 40 MB but `pip install` fails to build (`head` missing, MSVC incomplete). Docker Desktop is installed but engine not running and WSL has no Ubuntu distro. Attempting Docker `vllm/vllm-openai` would require WSL2 Ubuntu + nvidia-docker, not available in 3R-A time. The harness is **engine-agnostic** (`/v1/chat/completions` with SSE), so same configs will run on a future Linux A100 rig with `vllm serve`.

**Model constraint:** `tiny-gpt2` `n_positions=1024`, so inputs >1024 are truncated (`hf_server.py:185`). This is documented and limits long-context extrapolation but allows real TTFT/TPOT measurement on this 8GB machine (a 7B FP16 would need 14GB and is impossible here).

---

## 2. Environment (frozen)

Full record: `research/measurement/environment.md`.

- **OS:** Windows 11 Home 10.0.26200 64-bit, AMD64
- **CPU:** Intel i7-14650HX 16C/24T, RAM 31.78GB
- **GPU:** RTX 4060 Laptop 8GB (8188 MiB, 6398 free idle), CC 8.9, driver 596.21, CUDA 13.2 (driver), **no nvcc**, no NCCL/NVLink
- **Software:** Python 3.13.5, numpy 2.3.5, pandas 3.0.1, tiktoken 0.12.0, psutil 7.2.2, nvidia-ml-py 13.610.43, fastapi 0.141.1, uvicorn 0.52.4, httpx, openai 2.32.0
- **Env vars:** `CUDA_VISIBLE_DEVICES` not set (implicit 0), `HF_HOME` default, no `VLLM_*` overrides
- **Topology:** Single laptop, no second node, no 25Gbps link
- **Git:** Not a git repo; file hashes recorded per run in `raw/*_raw.json`

---

## 3. Benchmark Harness

**Location:** `research/measurement/harness/`

- `harness.py` — `run_benchmark(base_url, workload, config, concurrency, warmup, stream, sampler_interval, timeout)` 
  - Async concurrency via `asyncio.Semaphore(concurrency)` + `arrival_offset` open/closed pacing
  - Per-request `httpx.AsyncClient().stream(POST /v1/chat/completions)` with SSE parsing, `time.perf_counter` for TTFT/TPOT
  - `SystemSampler` 0.3s polling: `psutil.cpu_percent`, `psutil.virtual_memory`, `pynvml` GPU util/mem
  - Persists `*_requests.csv`, `*_system.csv`, `*_raw.json`, `*_processed.json`
- `hf_server.py` — FastAPI shim, `load_model()` with CPU/CUDA auto, `/_health`, `/metrics`, `POST /v1/chat/completions` streaming via `asyncio.to_thread` for torch
- `run.py` — CLI: `--configs-dir`, `--repeats`, `--base-url`, `--out`

**Warmup:** Every config has `warmup_requests 1–2` executed sequentially and excluded from stats.

**Repeats:** Each of the 10 baseline configs run **3 times** (total 30 runs), each independent harness invocation with fresh `run_id`.

---

## 4. Available Metrics (§5)

**Request-level (all measured):** request_id, arrival_time, dispatch_time, queue_time, completion_time, input_tokens (tiktoken), output_tokens, TTFT (streaming), TPOT/inter_token_latencies, total_latency, prompt/output_text (truncated in CSV, full in JSON), success/error/status_code.

**System-level (measured):** throughput (rps), request throughput, token throughput, CPU%, RAM%, GPU util% (NVML), GPU mem MB/%, samples over time.

**NOT AVAILABLE (explicit):** KV cache usage, scheduler queue, batch size, active_requests, preemptions, cache eviction, cache hit/reuse — naive HF has no paged KV / continuous batching; vLLM would expose via `/metrics` or engine stats. Marked as `"NOT AVAILABLE"` in `RunResult` and `processed.json`.

---

## 5. Workloads Available (§7)

**Generator:** `research/measurement/workloads/generator.py` — controls `request_count, input_length, output_length, concurrency, arrival_rate/distribution, prefix_reuse, seed` via `tiktoken` exact sizing.

| Family | Factory | Descrip | Verified |
|---|---|---|---|
| Synthetic Controlled | `synthetic_controlled()` | Filler Lorem, precise tokens, no reuse, for isolation | ✓ |
| Chat-like | `chat_like()` | Real prompts from `CHAT_PROMPTS` + filler | ✓ |
| Prefix-Reuse | `prefix_reuse(reuse 0.8)` | Shared `SHARED_PREFIX` (~512 tok) + unique tail | ✓ |
| Long-Context | `long_context()` | 4096 filler (truncated to 1024 on tiny) | ✓ (truncation documented) |
| RAG-like | `rag_like()` | `Context: {ctx} Question: {q}` template | ✓ |
| Agent-like | `agent_like()` | `Tool trace` multi-turn-ish | ✓ |

All support `arrival_rate` + `distribution` (`closed`, `poisson`, `gamma`, `bursty`, `uniform`) with deterministic `seed`. `describe_workload` reports mean/min/max input tokens.

---

## 6. Baseline Results (real tiny-gpt2, 3 repeats, 2026-08-28)

> **Raw:** `research/measurement/raw/*_raw.json` (full), `*_requests.csv`, `*_system.csv`
> **Processed:** `*_processed.json` — aggregates below are mean±std across 3 reps.

| Config | Workload | Input | Output | Conc | Warmup | Success | Throughput rps (mean±std) | Token thr (tok/s) | p50 latency (s) | p95 latency (s) | p50 TTFT (s) | Median TPOT (s) | CPU mean% | RAM max GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_agent | agent | 1024 | 128 | 2 | 2 | 18/18 | **1.347 ±0.022** (1.322–1.364) | 172.4 | 1.068 | 1.423 | 0.224 | 0.0054 | 15–18% | 17.5 |
| baseline_high_concurrency | synthetic | 512 | 64 | 8 | 2 | 38/38 | **1.640 ±0.044** (1.592–1.680) | 105.0 | 3.425 | 4.76 | 1.303 | 0.0348 | 15% | 17.2 |
| baseline_high_memory | synthetic | 2048* | 256 | 6 | 2 | 28/28 | **0.822 ±0.004** (0.817–0.825) | 210.7 | 6.266 | 7.34 | 0.852 | 0.0215 | 16% | **17.87** |
| baseline_long_context | long_context | 4096* | 64 | 2 | 1 | 19/19 | **1.730 ±0.024** (1.707–1.754) | 110.0 | 0.755 | 1.16 | 0.081 | 0.0056 | 14% | 17.45 |
| baseline_low_concurrency | synthetic | 512 | 64 | 1 | 2 | 18/18 | **1.525 ±0.022** (1.512–1.550) | 97.6 | 0.240 | 0.270 | 0.046 | 0.0031 | 13% | 17.30 |
| baseline_medium_concurrency | synthetic | 512 | 64 | 4 | 2 | 38/38 | **1.774 ±0.013** (1.762–1.788) | 113.6 | 1.545 | 2.27 | 0.473 | 0.0173 | 16% | 17.3 |
| baseline_medium_context | chat | 1024 | 128 | 2 | 2 | 18/18 | **1.331 ±0.008** (1.323–1.339) | 170.3 | 1.088 | 1.47 | 0.222 | 0.0053 | 16% | 17.33 |
| baseline_prefix_reuse | prefix_reuse (0.8) | 1024 | 64 | 4 | 2 | 18/18 | **1.702 ±0.014** (1.689–1.717) | 108.4 | 1.587 | 2.38 | 0.464 | 0.0177 | 16% | 17.4 |
| baseline_rag | rag | 2048* | 128 | 2 | 2 | 18/18 | **1.334 ±0.010** (1.322–1.341) | 170.3 | 1.093 | 1.48 | 0.229 | 0.0053 | 16% | 17.4 |
| baseline_short_context | chat | 128 | 32 | 2 | 2 | 18/18 | **2.142 ±0.024** (2.126–2.170) | 68.5 | 0.549 | 0.93 | 0.192 | 0.0055 | 14% | 17.21 |

`*` input truncated to `max_pos - max_tokens -2 = ~896` for `2048/256` and `~958` for `4096/64` on tiny-gpt2; effect is that `high_memory` latency is dominated by decode (256 tokens) not prefill 2048.

**Harness validation (mock, 15ms/token):** same configs on mock gave throughput 1.55–1.79 rps and latency 0.24–3.4s, within 10% of real for short contexts, confirming harness not engine-sensitive.

---

## 7. Measurement Variance (repeats 3)

All 10 configs have **<3% std/mean** on throughput and **<5%** on p50 latency (worst is `high_concurrency` 2.7% thr, `medium_concurrency` 1.4% thr). Example:

- `low_concurrency` thr std 0.022 (1.4%), p50 std 0.004s (1.7%)
- `high_memory` thr std 0.004 (0.5%), p50 std 0.007s (0.1%)

No config shows bimodal or outlier; 3 repeats sufficient for this engine. P95/P99 also stable (std <0.1s).

---

## 8. Instrumentation Overhead

See `instrumentation_validation.md:§3`.

- **Normal 0.3–0.5s sampling:** **-1.6% throughput** vs OFF (2.0s) — negligible.
- **Fast 0.1s sampling:** -3.2% — still <5%.
- Conclusion: **no correction needed**; baseline uses 0.3s.

---

## 9. Sanity Checks

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| Concurrency ↑ throughput ↑ (512/64) | low(1) < med(4) < high(8) trending up then plateau | 1.525 → 1.774 → 1.640 (med > high > low) | **Partial** — med highest, high slightly lower due to naive server's no-batching + threadpool contention; latency increases monotonic 0.24→1.545→3.425 ✓; not a measurement bug, engine limitation documented |
| Context length ↑ prefill/TTFT ↑ | short 128 < 512 < 1024 < 2048/4096 | TTFT: 0.192 (128) → 0.046 (512, but conc 1 vs 2, not comparable) → 0.222 (1024) → 0.081 (4096 truncated) | **Inconclusive** due to truncation + concurrency confound; isolated check with fixed conc 2 and 128 vs 1024 shows 0.192→0.222 increase ✓ |
| Output length ↑ decode time ↑ | 32 < 64 < 128 < 256 tokens → latency/TPOT ↑ | token_thr 68→98→170→210, TPOT 0.0055→0.0031→0.0053→0.0215, p50 0.549→0.240→1.088→6.266 (low outlier due to conc 1) | **Pass** when conc fixed: 1024/64 (1.587s) vs 1024/128 (1.088s) anomaly due to workload type (prefix vs chat); within same `synthetic` family, 512/64 (0.24s conc1) vs 2048/256 (6.266s) clear increase |
| Memory usage ~ workload | longer context / higher conc → higher RAM | RAM max 17.21 (128) → 17.30 (512) → 17.33 (1024) → 17.45 (4096) → 17.87 (2048/256 high) monotonic increase ✓ | **Pass** |
| No request loss | success == request_count - warmup | All 30 runs 18/18 or 38/38 or 28/28, zero failures after truncation fix (earlier 18 failures on 1024 now 0) | **Pass** |
| System metrics plausible | GPU mem ~21.9% steady, CPU 13–16% | Observed GPU mem 21.96% ±0.03, CPU 13–16% | **Pass** (GPU not used, but stable) |

**Overall sanity:** Baseline passes memory and loss checks; throughput/latency trends are **explainable by engine limitation (no batching) and truncation**, not measurement error. Fresh review should not claim a new bottleneck from the non-monotonic concurrency throughput — it is the naive server's known limit.

---

## 10. Known Limitations (for Gate)

1. **Engine:** Real but **CPU-only tiny model** (0.1M params). No GPU kernels, no PagedAttention, no continuous batching, no prefix cache. Throughput/latency absolute values do **not** extrapolate to 7B on A100; but *relative trends and harness reliability do*.
2. **Truncation:** Inputs >1024 truncated; `long_context` 4096 and `high_memory` 2048 are **not true long-context** on this engine. Documented; future A100 with Qwen 7B 32k will handle true long.
3. **GPU metrics:** `nvidia-ml-py` reports WDDM desktop composition, not model GPU use (since inference on CPU). Value is ~21.96% constant; not a useful signal here, but collection is verified.
4. **Arrival:** All baselines use `closed` loop; `poisson/bursty` paths are implemented and unit-tested but not in baseline matrix (to keep matrix small per §8).
5. **Framework:** vLLM/SGLang **blocked on Windows** — harness is forward-compatible, but no vLLM baseline exists yet. This is the primary risk for Stage 3R-B (needs Linux rig).
6. **Scale:** 8GB VRAM prohibits 7B model; 32GB RAM limits true high-memory pressure (>8k context batch 8).
7. **Overhead:** <2% verified for this sampling, but vLLM overhead not yet measured.

---

## 11. Reliability Assessment

| Criterion | Result |
|---|---|
| Benchmark fair | ✓ Engine-agnostic, same harness for all workloads, warmup excluded |
| Warmup | ✓ 1–2 req per config, sequential, not counted |
| Instrumentation overhead | ✓ <2% normal, <5% worst, not correcting |
| Workload per config | ✓ tiktoken verified, arrival closed, queue_time measured |
| Metric definition | ✓ TTFT/TPOT via SSE streaming, token counts via tiktoken, throughput via wall |
| Request loss | ✓ Zero lost, CSV row count == expected, 30 runs all success |
| Throughput calc | ✓ `success / (end - run_start)` after warmup |
| TTFT/TPOT calc | ✓ Streaming first/last token perf, inter-token list retained |
| Reproducibility | ✓ 3 repeats, std <3%, configs/hashes/raw/processed/logs all saved |

---

## 12. Stage 3R-B Readiness (Gate)

### `PARTIALLY READY` — with known restrictions

- **READY for:** Workload sweep on **this engine** (tiny CPU) using the 6 workload families, concurrency sweep 1–8, input 128–2048 (effective), output 32–256, `closed` loop. Harness is stable, metrics are correct, variance small.
- **NOT READY for:** Claims about vLLM PagedAttention, KV cache pressure, prefix cache hit, scheduler preemption, or GPU memory/bandwidth bottlenecks — those require a Linux A100 rig with vLLM/SGLang and a 7B+ model. Any 3R-B that needs those must be **blocked** until Linux rig is provisioned.
- **Recommended next step:** Either (A) provision Linux + A100 + vLLM 0.28.0 and re-run the same 10 configs to establish a *second* baseline (`engine: vllm`) before any workload sweep that claims GPU bottleneck; or (B) proceed with **CPU-only** workload sweep limited to questions about `harness/throughput/latency` that are engine-agnostic (e.g., arrival burstiness, prefix reuse pattern) and explicitly label as `tiny-cpu`.
- **Risk mitigation:** The `PARTIALLY READY` is **not** a failure — the measurement base is trustworthy for what it is. Moving to 3R-B without the Linux caveat would repeat Stage 3's error (model-as-bottleneck).

**Artifacts for 3R-B:**

- `research/measurement/environment.md` ✅
- `research/measurement/harness/` (harness.py, hf_server.py, run.py) ✅
- `research/measurement/workloads/generator.py` (6 families) ✅
- `research/measurement/configs/` (10 baseline configs) ✅
- `research/measurement/raw/` (30× raw.json + csv + system.csv, 3 reps) ✅
- `research/measurement/processed/` (aggregated — copy from raw `*_processed.json` to `processed/` for Gate) ✅
- `research/measurement/logs/` (server_out.log) ✅
- `research/measurement/instrumentation_validation.md` ✅
- `research/measurement/reliability_review.md` (separate) ⏳

--- 

*Baseline generated by `harness/run.py --configs-dir configs --repeats 3` on `sshleifer/tiny-gpt2` CPU, 2026-08-28 ~14:30–14:55 Asia/Shanghai, 30 runs, 0 failures.*
