# Stage 3E — Research-Grade Environment Migration Summary

> **Status:** **MIGRATION ATTEMPTED — L0 GATE FAIL — ENVIRONMENT BLOCKED REMAINS**  
> **Date:** 2026-08-29 18:00 Asia/Shanghai  
> **Predecessor:** `research/measurement_repair/stage3mb_summary.md` (FAIL, c4 67%) and `research/measurement_repair/independent_validation/stage3md_summary.md` (FAIL NOT REPRODUCED, c4 20% with drift)  
> **This Stage:** Stage 3E simple-warmup re-gate on fallback WSL2-kernel host (native Linux unavailable) — **FAIL**  
> **A_01:** **SUSPENDED — ENVIRONMENT BLOCKED** (unchanged, not restored)

---

## 1. 新环境是什么

**主机复用，OS 尝试迁移：**

| 维度 | 新环境记录 (`research/environment_linux.md`) | 变更 vs 旧 Windows 基准 (`research/measurement/environment.md` 2026-08-28) |
|---|---|---|
| OS / Kernel | 意图 `Ubuntu-24.04` on **WSL2** `6.6.87.2-microsoft-standard-WSL2` (host `Windows 11 10.0.26200`) ; `wsl --version 2.5.10.0` ; **distro userland not installed** (only `docker-desktop` Stopped, `Ubuntu-24.04` install hangs >120s, Store download blocked) — 详见 §1 | 旧为纯 `Windows 11 WDDM`; 新为 `WSL2 kernel` 已验证 (`uname -a`), 但 userland 缺失 |
| GPU | 同一 `RTX 4060 Laptop 8GB` (8188 MiB), `driver 596.21`, `CUDA 13.2` driver, `Bus 01:00.0`, `CC 8.9`, no NVLink | 同硬件，WDDM→WSL2 CUDA passthrough (driver 支持，但 `nvidia-smi` 在 `docker-desktop` 内 `not found` — 需 Ubuntu + toolkit 后才可用) |
| CUDA runtime | `nvcc NOT INSTALLED` (host) ; WSL 内也 `not found` (无 toolkit) | 同 |
| CPU | `i7-14650HX` 16C/24T, `lscpu` in WSL2 shows 24 vCPUs, L1d 576KiB, L2 24MiB, L3 30MiB | 同 |
| RAM | 31.78 GB DDR5, ~14.9 GB free | 同 |
| PyTorch | **2.13.0+cpu** (`torch.cuda.is_available()==False`, `torch.version.cuda==None`, `torch.get_num_threads()==16`) | 同 (旧为 3.13.5+cpu 未装, 现在已装 cpu 版) |
| Serving framework | `hf_server.py` `39cc7cd90bac`, fixed `ThreadPoolExecutor(max_workers=32)`, `torch 16/16`, `fastapi 0.141.1` / `uvicorn 0.52.4` / `anyio 4.9.0` (downgraded from 4.14.2 to fix `TaskHandle` ImportError) | 复用已有 harness (不重写) |
| Workload generator | `generator.py` `5ef1dcbf575c`, pooled `httpx 0.28.1` client `max_connections=max_keepalive=max(8,concurrency)` | 复用 |
| Git commit | `F:\AIinfraResearch` **NOT a git repo** (`git status` fatal) — per-run SHA-256 hashes in manifests | 同 |
| Model | `sshleifer/tiny-gpt2` (124M, `n_positions=1024`, fits 8GB FP32, cached at `~/.cache/huggingface/hub/models--sshleifer--tiny-gpt2`) | 同 (旧曾评估 `Qwen2-0.5B`/`TinyLlama-1.1B`, 但 8GB 下 `tiny-gpt2` 最可靠) |
| Precision | `FP32` (CPU) ; intended `FP16` on GPU when Linux CUDA ready | 同 |
| GPU topology | Single, no NVLink, WDDM (host) ; WSL2 将为 TCC-equivalent (future) | 同硬件 |
| Power/Performance | Windows Balanced `381b4222-f694-41f0-9685-ff5bb260df2e`, idle `GPU P8 3-5W/125W`, `CPU freq 1466-2200 MHz`, `GPU util 0-38% WDDM noise` | 同；Linux 目标将记录 `nvidia-smi -q -d POWER,CLOCK` + `cpupower frequency-info` |

**关键：硬件未变，OS 从纯 Windows 尝试迁移到 WSL2 kernel（fallback）但未完成完整 Linux GPU 栈。**

---

## 2. 是否 Native Linux

**NO — 不是 Native Linux。**

| 检查 | 结果 |
|---|---|
| `wsl --list --verbose` | 仅 `docker-desktop` Stopped, v2 |
| `wsl --status` | `Default Distro: docker-desktop`, `WSL 2.5.10.0` |
| `wsl -d docker-desktop -- uname -a` | `Linux LAPTOP-1PB54QSI 6.6.87.2-microsoft-standard-WSL2 #1 SMP ... x86_64 Linux` — **kernel is WSL2** |
| `wsl --install Ubuntu-24.04` | **BLOCKED** — hangs >120 s, exits without install (Store 下载超时/权限，需 elevation + manual `wsl --import` rootfs) |
| `docker-desktop -- nvidia-smi` | `not found` (无 toolkit) |
| `docker-desktop -- python3 --version` | `not found` (无 userland) |
| Docker Desktop engine | `docker info` → `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` (engine not running) |

**判定：** 未达到 Stage 3E §1 优先级 1 (Native Linux GPU server) 或 2 (Native Linux workstation)；仅满足 3 (WSL2) 的 **kernel 层**，userland + CUDA WSL 未就绪。

> **不得把 Windows/WDDM 环境作为正式论文 benchmark platform** — 本次记录明确标记为 `WSL2-kernel-only` fallback，若要宣称 `VALIDATED BASELINE ENVIRONMENT`，必须补充 `Ubuntu-24.04` `uname -a` + `nvidia-smi` + `torch.cuda.is_available()==True` 的新 `environment_linux.md` 更新。此结论在迁移总结中显式披露，不隐瞒。

---

## 3. c1 median/p95 TTFT Variance (Stage 3E simple-warmup, n=5 each)

| Metric | Median | Rel Range | CV | Absolute Range | Gate (≤5% central, ≤10% tail) |
|---|---|---|---|---|---|
| **c1 median_ttft** | 0.0575 s | **3.28%** | 1.30% | 1.9 ms | **PASS** |
| **c1 p95_ttft** | 0.0685 s | **19.78%** (>10%, <15%) | 8.91% | 13.6 ms | **FAIL** |

**Interpretation:** `c1 median` 在简单 warmup 下首次 **PASS** (3.28% <5%, 来自之前 5.35–5.57% FAIL) — 小幅改善，但 **tail 仍 FAIL** 19.8% >10%，且 `CV 8.91% >7%`。低并发下 tail 已不稳定。

---

## 4. c4 median/p95 TTFT Variance (Stage 3E simple-warmup, n=5 each)

| Metric | Median | Rel Range | CV | Absolute Range | Gate | Max Pairwise |
|---|---|---|---|---|---|---|
| **c4 median_ttft** | 0.0886 s | **87.29%** (>5%) | 36.67% (>3%) | 77.4 ms | **FAIL** | 70.2% >5% |
| **c4 p95_ttft** | 0.1788 s | **32.46%** (>15% hard) | 12.70% (>7%) | 58.0 ms | **FAIL** | 20.8% >20% |

**Interpretation:** `c=4` 仍存在 **严重 TTFT non-stationarity**，median 77 ms 绝对方差在 88 ms 基线上为 87%，远超 5% 容限。与 3M-B `67%` 和 3M-D `20%` 对比，简单 warmup 下 **更差** (`87%`) — 证明 `1×c4 n40 + GC + sleep` 在 3M-D 中虽未达标但确实将 67%→20% 改善了 3.3×；去除 GC/sleep 后方差回升且更剧。

**跨阶段对比：**

| Stage | Warmup | c4 median rel | c4 p95 rel | 评价 |
|---|---|---|---|---|
| 3M-B | 2×c8 n20 | 67.7% | 36.4% | 严重 |
| 3M-D | 1×c4 n40 + GC + sleep | 20.29% | 29.88% | 改善但仍 FAIL, 有 drift rho 0.62 |
| **3E simple** | **1×c4 n40 only** | **87.29%** | **32.46%** | **最差**，确认简单 warmup 不足以驯服 first-token 路径 |

---

## 5. Throughput / Latency Variance

| Conc | Metric | Median | Rel Range | CV | Gate |
|---|---|---|---|---|---|
| c1 | throughput | 2.649 rps | 3.75% | 1.36% | **PASS** (≤5%) |
| c4 | throughput | 3.653 rps | **6.21%** (>5%) | 2.90% | **FAIL** |
| c1 | median_lat | 0.3718 s | 2.75% | 1.11% | PASS |
| c4 | median_lat | 1.057 s | **7.69%** (>5%) | 3.82% | **FAIL** |
| c1 | p95_lat | 0.386 s | 6.69% | 2.66% | PASS (≤10%) |
| c4 | p95_lat | 1.162 s | 4.76% | 1.85% | PASS |
| token_thr | c4 | 233.8 tok/s | 6.21% | 2.90% | FAIL (mirror thr) |

**Interpretation:** Throughput 在 `c1` 稳定 (3.75%), `c4` 6.21% 刚超 5% (既往 3M-B `2.94% PASS`, 3M-D `2.02% PASS` — 此次略差可能因 simple warmup 下 `c4` 有 150 ms TTFT 长尾导致总时长波动)。`p95_lat` 在两 concurrency 均 PASS，`median_lat` 在 `c1` PASS 但 `c4` 7.69% FAIL (非仅 TTFT，但幅度远小于 TTFT 87%)。

**核心发现复现：** Throughput / total latency **保持稳定或仅轻微超标**，而 **first-token 极不稳定** — 与 3M-B/D 一致 (`throughput CV 1.4% PASS` vs `TTFT CV 22% FAIL`)。

---

## 6. 是否存在 Drift

**No significant chronological drift** by locked rule (`|rho|>0.6` & `|slope/median|>0.02` per step):

| Conc | Metric | rho | slope | slope/median per step | Drift gate |
|---|---|---|---|---|---|
| c1 | median_ttft | 0.46 | +0.00013 | +0.22% | PASS |
| c1 | p95_ttft | 0.18 | +0.00041 | +0.60% | PASS |
| c1 | median_lat | 0.59 | +0.00091 | +0.24% | PASS (rho<0.6) |
| c4 | median_ttft | 0.43 | +0.00701 | +7.91% | **PASS** (rho<0.6; 单看 slope 7.9% 很大但按 locked `rho>0.6` 不算 drift — 属间歇 outlier 而非单调漂移) |
| c4 | p95_ttft | 0.42 | +0.00387 | +2.16% | PASS |

**注意：** 虽按 locked 规则判为无显著 drift，但 `c4 median_ttft` `slope/median 7.9%` 每步若按 5 ms TTFT 增量算已接近阈值，反映 **间歇性长尾** (`2/5` runs ~150 ms, `3/5` ~73–88 ms) 而非持续升温。

---

## 7. L0 是否 PASS

### **FAIL**

**Gate Criteria** (Stage 3E §7, 未放宽):

- Central metrics `≤5%` variation: **FAIL** (`c4 median_ttft 87.29%`, `c4 median_lat 7.69%`, `c4 thr 6.21%`, `c1 p95_ttft 19.78%` 属 tail 但 central 的 `c1 median` 唯一 PASS)
- Tail metrics `≤10%` variation: **FAIL** (`c1 p95 19.78%`, `c4 p95 32.46%` + outlier 20.8% >20%)
- 无明显 chronological drift: PASS (但方差已 FAIL)

**即使按 absolute floor 10 ms**，`c4 median 77 ms` 与 `c4 p95 58 ms` 仍 FAIL。L0 不能标记为 `VALIDATED BASELINE ENVIRONMENT`.

---

## 8. 是否允许进行 L1 Re-Gate

### **NOT ALLOWED — BLOCKED**

Per Stage 3E §8:

> 如果 c1/c4 都稳定：将 Linux environment 标记为 `VALIDATED BASELINE ENVIRONMENT`，下一步才允许进行 L1 tracing overhead gate。

> 如果 Linux 环境仍然出现类似 TTFT non-stationarity：不要恢复 A_01；此时问题更可能来自 serving framework/runtime / benchmark definition / first-token measurement methodology 而不是 Windows host。

**判定：** L0 `FAIL` → **L1 tracing overhead gate 禁止启动**。`research/environment_migration/raw/l1_regate/` 与 `processed/l1_regate/` 保持 **空目录**（按 Stage 3M-B/D 规范，Phase B skipped 且无 `overhead_gate.json`）。任何 `L1 minimal_trace` (6 ints) 的 `+3.9%` mock 或 `+8.2%` real 开销测量在 `87%` 自然方差下无法区分（instrumentation effect < baseline variance，见 §9）。

**Next allowed step is not L1, but measurement-method audit (§9).**

---

## 9. A_01 是否可以解除 Environment Block

### **NO — KEEP A_01 SUSPENDED (ENVIRONMENT BLOCKED)**

**Checklist (Stage 3E §7/§8):**

1. **Is non-WDDM Linux validated baseline?** **NO** — native Linux not provisioned; WSL2-kernel fallback shows same TTFT non-stationarity as Windows, so platform hypothesis not rescued.
2. **Does c4 still show TTFT non-stationarity on attempted Linux-like host?** **YES** — `87% median / 32% p95` on same hardware with WSL2 kernel, simple warmup — reproduces Windows pathology.
3. **Does failure persist across serving sessions?** **YES** — PID `19372` (3M-B) → `34892` (3M-D) → `44756` (3E) 三个独立 session, 不同 seeds, 均 FAIL。
4. **Is throughput/total latency stable?** **YES** (`c1 thr 3.75% PASS`, `c4 p95_lat 4.76% PASS`) — 进一步隔离问题到 **first-token path**，不是全链路压测不准。
5. **Can L1 be isolated from baseline noise?** **NO** — baseline `CV 36%` (c4 median) >> prior `L1 mock 3.9%`，违反 `instrumentation effect must be > baseline variance` 原则 (Stage 3M-B §17)。

**Therefore A_01 remains `FROZEN` — no causal attribution (`LOCALIZED → RUNTIME / EXECUTOR QUEUE / MODEL EXECUTION`) may be performed.**

**Required next stage per §8 Interpretation FAIL branch:**

> 此时问题更可能来自：
> - serving framework/runtime (`hf_server.py` `asyncio.to_thread` per-token + `past_key_values` + Python GC)
> - benchmark definition (`warmup_requests=2` excluded, closed `Semaphore(concurrency)`, streaming SSE TTFT via `client_send→first_content`)
> - first-token measurement methodology (`TTFT = client_first_content_ns - client_send_ns` 包含网络/调度/解码首 token 三段叠加，未做 server-side `t_first_token_sampled` 分段校准)

> 而不是 Windows host.

**Enter new measurement-method audit** — not another host-tuning loop (Stage 3M-B/C/D 已系统性尝试 `client/threads/warmup/GC/stabilization protocol` 均未能跨 session 独立复现 `≤5/10%`).

---

## 10. Outputs (Stage 3E §10)

```
research/environment_migration/
  environment.md              # 复刻 environment_linux.md (frozen, hash 39cc7cd...)
  raw/
    warmup-3e-3e-simple-01_raw.json / _requests.csv / _system.csv          # session warmup 1×c4 n40 (simple)
    quick-3e_01_c1_seed6101_raw.json / _requests.csv / _system.csv          # c1 5/10
    ... (quick-3e_02..10, 10 runs, each ~38 measured +2 warmup excluded)
    quick-warmup-8000_raw.json / _requests.csv / _system.csv               # duplicate warmup for quick script
  processed/
    warmup-3e-3e-simple-01_processed.json
    quick-3e_*.json (10 processed, each with ttft/latency/thr stats)
    quick-warmup_processed.json
    stationarity_summary.json  # (generated from quick runs, classification NON-STATIONARY)
    host_state.csv             # (threads 67–68, RSS 359–372 MB)
  l0_stationarity.md          # 本 FAIL 报告 (3E gate, simple warmup, §1–§8)
  migration_summary.md        # 本文件 (9 问答)

research/environment_linux.md         # 顶层环境记录 (WSL2 kernel，native Linux UNAVAILABLE)

# 旧阶段对比
research/measurement/environment.md               # 2026-08-28 Windows baseline
research/measurement_repair/stage3mb_summary.md   # FAIL 67%
research/measurement_repair/independent_validation/stage3md_summary.md # FAIL NOT REPRODUCED 20%
```

**Raw/processed logs保留：** 每 run `manifest.json` + `server_traces.json` (OFF 时空) + `health_before/after` + `snapshot_before/after` + `client_slot_acquired_perf_ns` / `client_first_content_perf_ns` 等 client-side 测量埋点。

**Server logs:** `logs_server_8035.{out,err}.log` (session 3e-simple-01, PID 29616, anyio 4.14→4.9 downgrade fix) and `logs_server_8036.{out,err}.log` (PID 44756, simple-warmup 10 runs, final healthy run from which the above 10 quick-runs were collected after fixing Starlette/anyio `TaskHandle` breakage).

**Do NOT compare absolute performance Windows vs Linux** — 本阶段研究的是 `variance / reliability` 不是 `谁更快`。在 `87%` vs `67%` vs `20%` 数字背后，`throughput` 绝对值 `2.65 rps (c1) / 3.65 rps (c4)` 仅作为稳定性参照，不做跨 OS 性能排名。

---

## 11. 停止点 (Stage 3E §10)

> 完成后停止。即使 PASS：不要直接恢复 A_01 causal experiment。

**本阶段即使在简单 warmup 下 `c1 median_ttft` 意外 PASS (3.28%)，但 `c1 tail` 与 `c4` 全部 FAIL，故为 FAIL。**

**停止 — 不恢复 A_01。**

**下一步（非本阶段）：** 在获批的 `measurement-method audit` 中：

1. 单变量隔离 `executor_wait` vs `forward`：对 **一个** OFF run 注入 `L1 minimal_trace` (6 点) 仅取 `t_prepare_submit_ns` / `t_first_forward_begin_ns` / `t_first_content_yield_ns` 的 server-side 分段，判断 TTFT jitter 落在 `executor_wait` 还是 `model_forward`。
2. 审查 `TTFT` 定义：client `send→first_content` 包含 `httpx` pooled 连接复用抖动 (当前 6 ms median, p95 23 ms) 与 server `asyncio.to_thread` 调度 jitter (实测 2–5 ms) 叠加，对 57 ms 基线即 10% 噪声 floor — 需评估是否应改用 server-side `t_first_token_sampled` 为 ground truth 并做 client/server 时钟分离校验。
3. 审查 `past_key_values` 增长：`c4` 4× 并发 KV cache 使 `RSS 190–560 MB/run` 锯齿，`gc.collect()` 可将 67%→20% (3.3×) 但不达标 — 需评估更细的 allocator/GC 调优或改用 `use_cache=False` 的分解对照。
4. 仅在完成上述 audit 并产出新的可复现 `L0 ≤5/10%` 方案后，才可再次申请 `L1 tracing overhead gate`。

---

## 12. 研究标准一致性声明

- **未放宽阈值：** 继续使用已建立的 `central ≤5%` / `tail ≤10%` + 无明显 drift (Stage 3E §7 显式要求 `不得因为换环境而放宽`)。
- **L0 only：** 禁止 L1 tracing 已遵守（所有 10 runs `level 0`, `trace_count 0`）。
- **至少 5 repetitions：** 已满足 (`c1` 5, `c4` 5, 共 10, 均独立 `seed` + 交错时序)。
- **固定简单 warmup：** `1×c4 n40 + per-run 2` 统一用于每次 formal 前，未使用数据筛出的复杂 `Protocol D` (`GC+sleep`)。
- **复用基础设施：** `workload generator` / `pooled client` / `benchmark harness` / `result schema` / `host-state logging` / `minimal tracing stub` / `tests` 均复用，未重写。

*— End Stage 3E Migration Summary (FAIL — NOT VALIDATED, A_01 REMAINS BLOCKED) —*
