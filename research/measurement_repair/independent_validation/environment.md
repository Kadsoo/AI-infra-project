# Environment Record — Stage 3M-D Independent Validation

> **Session:** independent-3md (new isolated serving session)  
> **Startup time:** 2026-08-29T07:36:30Z (UTC, server 8040) + health verified 07:38:26Z  
> **Purpose:** Independent validation of Candidate Stable Protocol D — must prove cross-session reproducibility

## 1. Serving Session

| Item | Value |
|---|---|
| Hostname | LAPTOP-1PB54QSI |
| OS | Windows 11 Home 10.0.26200 |
| CPU | Intel Core i7-14650HX 24 threads (Intel64 Family 6 Model 183) |
| RAM | 31.78 GB |
| Disk | NTFS F:\ |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU 8GB (8188 MiB), Driver 596.21, CUDA 13.2 |
| Python | 3.13.5 (tags/v3.13.5) |
| Torch | 2.13.0+cpu |
| Transformers | 5.16.1 |
| httpx | 0.28.1 |
| psutil | 7.2.2 |
| New PID | **34892** (distinct from Stage 3M-C 56704 and Stage 3M-B 19372) |
| Port | **8040** (isolated, new session, not reused) |
| Model | `sshleifer/tiny-gpt2` CPU FP32 `n_positions=1024` |
| Device | `cpu` |
| Framework | `hf_transformers_naive` via `hf_server.py` |
| hf_server.py hash | `39cc7cd90bac40f1f712fc4233b114ad64466ff98851ff35ce4049b58b06cb87` |
| harness.py hash | `aae22af6712ff9da` (first 16) |
| run_stage3md.py hash | `3b2d5ab11a6c686f` (first 16, new runner implements Protocol D) |
| Generator hash | `generator.py` same as Stage 3M-B |
| LOCKED plan hash | `983387852875e7bb` (LOCKED_INDEPENDENT_STATIONARITY_PLAN.md first 16) |
| Startup command | `python research/measurement/harness/hf_server.py --model sshleifer/tiny-gpt2 --device cpu --port 8040 --executor-workers 32 --torch-threads 16` |
| Uptime at health check | 109 s |

## 2. Fixed Configuration (Per Locked Plan §3)

| Item | Locked Value | Verification |
|---|---|---|
| Torch threads | 16 / 16 | `/health runtime.torch_num_threads==16` PASS |
| Torch interop | 16 | PASS |
| ThreadPoolExecutor | 32 | `fixed_executor_max_workers==32` PASS, startup log `fixed executor max_workers=32` |
| Client | pooled `httpx.AsyncClient(http2=False, limits=max_connections=max_keepalive=max(8, concurrency))` reused | logged per run `client_mode pooled` |
| Workload | synthetic 512 input / 64 output / no prefix reuse / closed burst / stream True / temp 0.0 / timeout 180 / sampler 0.3 | `describe_workload` logged |
| Request count | 40 = 2 warmup sequential pooled excluded +38 measured | `config.request_count 40, warmup_requests 2` |
| Sampler | 0.3 s fixed | SystemSampler interval 0.3 |
| Tracing Phase A | L0 OFF only (`level 0`) | `_set_level 0` before every run |
| Protocol D | 1×c4 n40 warmup (seed 8000+chrono) + `gc.collect()` + `sleep(2s)` + host check + formal | per-run manifest logs all 5 steps |

**PID isolation check:** New PID 34892 ≠ 56704 (candidate) and ≠19372 (Stage 3M-B). `Get-NetTCPConnection` shows 8040 Listening on 34892 only. No reuse of old session.

## 3. Startup Host State

At first health check (07:38:26Z, before any warmup bursts):

- `process_threads`: 64
- `python_threads`: 2
- `rss_mb`: 438.0
- `gpu_util`: 15.0% (WDDM desktop composition noise; prior Stage 3M-B saw 0–39% same)
- `gpu_mem_used`: 1769 MB / 8188 MB (21.6%)
- `uptime`: 109 s
- `loaded`: true, `mock`: false, `device`: cpu

This is the fresh process baseline against which Protocol D convergence will be judged (threads should remain ±5 after warmups, RSS pre-run should show small variance).

## 4. Hardware Topology Note

Single laptop, no A100, same as Stage 3M-B. GPU-gated experiments remain BLOCKED, but CPU tiny-gpt2 validation is fully executable and is entire scope of this independent validation. This does not affect stationarity judgment since reasoning is comparative (L0 vs L0 variance) within same host.

## 5. Reproducibility Notes

- Project root `F:\AIinfraResearch` is **NOT a git repository** — SHA-256 file hashes used instead of commit.
- All random generators seeded with **new seeds 5101,5102,5103 (c1) and 5401,5402,5403 (c4)** pre-registered, distinct from discovery 4401.
- Network not used except for initial model download (sshleifer/tiny-gpt2 already cached).
- Sampler interval fixed 0.3 s; all runs share same PID, same torch threads, same pool.

*— End environment —*
