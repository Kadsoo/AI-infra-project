# Host State Record — Stage 3M

> **Host:** LAPTOP-1PB54QSI — i7-14650HX 24T / 31.78 GB RAM / RTX 4060 Laptop 8GB WDDM driver 596.21 CUDA 13.2  
> **Engine:** `hf_transformers_naive` CPU FP32 `sshleifer/tiny-gpt2` (`n_positions=1024`)  
> **Sampler:** `SystemSampler interval 0.3 s` (psutil + pynvml) — 每个 run 均固定  
> **Reference gate session:** `a01-20260828-overhead` PID 18968, port 8017, 12 runs 12:42–12:51 UTC

---

## 1. Recorded per-run host state (required)

每个 run 的 `raw/*_manifest.json` 与 `processed/*_processed.json` 必须包含，且 `host_state.md` 的 `raw/host_state.csv` 聚合：

| Field | Source | Gate 前校验 |
|---|---|---|
| `gpu_util` (%) | `pynvml.nvmlDeviceGetUtilizationRates` | 理论上 CPU 推理应 0–10%（WDDM 桌面合成噪声），>30% 标记 invalid |
| `gpu_mem_used_mb` / `total` / `percent` | `nvmlDeviceGetMemoryInfo` | 记录绝对值，不应随 tracing ON/OFF 跳变 >10% |
| `gpu_temp` (if available) | `nvmlDeviceGetTemperature` | 若可用，>85°C 标记 thermal throttling 风险 |
| `gpu_clocks` (if available) | `nvmlDeviceGetClockInfo` | 记录 SM/Mem 时钟，骤降视为 invalid |
| `cpu_percent` (host) | `psutil.cpu_percent(interval=None)` | >90% 持续视为 background contention invalid |
| `cpu_freq_current_mhz` | `psutil.cpu_freq().current` | 基准 2200 MHz，<1500 MHz 视为 throttling invalid |
| `ram_used_gb` / `percent` | `psutil.virtual_memory()` | >90% 标记 invalid |
| `server_process_cpu_percent` | `psutil.Process(pid).cpu_percent` | 记录，突增 >400% 且与并发无关视为 invalid |
| `server_process_rss_mb` | `Process.memory_info().rss` | **关键**：单调增长 >500 MB/run 且不回收视为 leak invalid |
| `server_process_threads` / `python_threads` | `Process.num_threads()` / `threading.active_count()` | **关键**：跨 run 无界增长无效（见 §2） |
| `background_processes` (top 5 by CPU) | `psutil.process_iter` 采样 | 记录是否有 Chrome/Defender/Update 抢占 |
| `serving_pid` | `/health` `pid` | 必须 12 runs 同 PID，否则 session invalid |
| `experiment_start_time` (UTC ISO) | `time.time()` | 用于 run-order 漂移分析 |
| `torch_num_threads` / `interop` | `torch.get_num_threads()` | 固定 16/16，变化则 invalid |

失败的 run **不得静默删除**，标记 `invalid=True` 并保留 `raw/`，在 `overhead_gate_results.md` 的 matched-pair 表中高亮。

---

## 2. Observed non-stationarity in reference gate (why this file exists)

### 2.1 Thread count monotonic growth (leak-like)

| Run | `threads_before` | `threads_after` | Δ |
|---|---|---|---|
| 01 OFF c1 | 103 | 119 | +16 |
| 02 ON c1 | 119 | 120 | +1 |
| 03 ON c1 | 120 | 120 | 0 |
| 04 OFF c1 | 120 | 120 | 0 |
| 05 OFF c4 | 120 | 167 | **+47** |
| 06 ON c4 | 167 | 183 | +16 |
| 07 ON c4 | 183 | 199 | +16 |
| 08 OFF c4 | 199 | 199 | 0 |
| 09 OFF c8 | 199 | 231 | **+32** |
| 10 ON c8 | 231 | 231 | 0 |
| 11 ON c8 | 231 | 247 | +16 |
| 12 OFF c8 | 247 | 247 | 0 |

**Interpretation:** 默认 `ThreadPoolExecutor` 按需扩容且永不收缩 + torch intra/inter 线程（16+16） + uvicorn 线程。首个 c=4/c=8 run 各触发 +47/+32 突增，之后稳定。**Gate 期间 thread 稳定性未满足**：跨并发的扩容与 run order 耦合，使 c=1 的 OFF→ON 对比包含不同的线程池大小。**修复要求：** gate 前执行 **warmup 饱和**：在正式 paired runs 前，以 c=8 并发预跑 1–2 次 dummy workload（不计入），使线程池达到 240+ 稳定态后再开始第一对 OFF/ON。

### 2.2 RSS sawtooth growth

`rss_before/after` (MB): 45→261→79→80→81→82→270→274→282→96→108→658→1024→291。呈现 **锯齿 + 总体攀升**：c=1 段 45→261 MB（首 run 分配 tokenizer/model cache），c=4 段 270→282 MB，c=8 段 658→1024 MB（8 并发 past_key_values 堆积），OFF run 后偶尔回落至 96 MB（GC 回收）。1024 MB 峰值接近可用 RAM 的 6%，虽未 OOM，但与 tracing 叠加时 GC 暂停风险增高。**Gate 前校验：** 若 `rss_before` > 80% RAM 或单 run 增长 > 500 MB 且未回收，标记 invalid 并重启 server。

### 2.3 CPU frequency throttling

`cpu_freq_current_mhz` 均值：OFF c1 1934, ON c1 2077, ON c1(2) 1788?, OFF c1(2) 2160 等，基线 2200 MHz 偶尔跌至 1466 MHz。**判定：** 连续 3 个样本 <1500 MHz 标记 thermal throttling invalid。

### 2.4 GPU is noise

GPU util 0–10%（WDDM 合成器），`gpu_mem_used` 1711 MB idle，变化 <5% across runs，确认 **GPU 非本实验变量**（CPU 推理），`gpu_util` 不用于 gate，但需记录以排除意外 GPU 任务抢占。

---

## 3. Host-state capture implementation

### 3.1 At server health (per run before/after)

`GET /health` 已返回 `pid`, `rss_mb`, `process_threads`, `python_threads`, `gpu_*`, `runtime.torch_*`，无需改动。`run_a01_causal.py: health_before / health_after` 已记录。

### 3.2 At harness sampler (0.3 s throughout run)

`SystemSample` 已包含 `cpu_percent`, `ram_*`, `gpu_*`, `cpu_freq_current_mhz`, `server_process_cpu_percent`, `server_process_rss_mb`, `server_process_threads`。保留 0.3 s 固定间隔，OFF/ON 相同。

### 3.3 At experiment runner (new for Stage 3M)

新增 `research/measurement_repair/raw/host_state.csv` 与 `host_state.md` 附表，每 run 追加一行：

```
run_id,concurrency,seed,trace_level,order_in_session,start_utc,pid,threads_before,threads_after,rss_before,rss_after,cpu_mean,cpu_max,cpu_freq_mean,gpu_util_mean,invalid,invalid_reason
```

`invalid` 判定由 `gate_plan.md` 的 Stability gate 触发。

---

## 4. Invalid-run handling

- **不得删除**：invalid run 保留 `raw/`、`processed/`、`server_traces`，在 matched-pair 表中 `Valid=NO` 高亮。
- **判定后行动**：若同 concurrency 的 2 个配对中任一 invalid，**该并发点 gate 直接判 INCONCLUSIVE**，需修复环境后重跑，而非用另一 seed 替补。
- **记录原因**：`invalid_reason` 枚举：`thread_growth`, `rss_spike`, `cpu_throttle`, `pid_change`, `gpu_contention`, `background_process`.

*— End of host_state —*
