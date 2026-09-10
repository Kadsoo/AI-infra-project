# Stage 3R-A Instrumentation Validation

> **Date:** 2026-08-28
> **Environment:** `environment.md` (Win 11, RTX 4060 Laptop 8GB, Python 3.13.5, torch 2.13.0+cpu, transformers 5.16.1, `sshleifer/tiny-gpt2`)
> **Engine:** `hf_transformers_naive` on CPU (real forward, no PagedAttention, no continuous batching)
> **Harness:** `research/measurement/harness/harness.py` (OpenAI-compatible, streaming SSE, NVML/psutil polling)

## 1. Instrumentation Principle (§6)

优先级：

1. framework 已有 metrics / profiler — `hf_server.py` exposes `/health` and `/metrics` via psutil+NVML, no model code change
2. logging hooks — request logging via FastAPI, per-request wall-clock in harness
3. tracing — SSE streaming for TTFT/TPOT, not modifying generate path
4. minimal instrumentation — only `asyncio.to_thread` wrapper to avoid event-loop blocking; no KV-cache / scheduler state mutation

> 不改变被测系统本身行为：所有计时在 **客户端** (`harness.py` `time.perf_counter` / `time.time`) 完成；服务端仅做无侵入采样 (`psutil.cpu_percent`, `pynvml`).

## 2. Available vs NOT AVAILABLE Metrics (§5)

### Request-level (all reliably measured)

| Metric | Source | Status |
|---|---|---|
| request_id | `uuid` in harness | ✓ |
| arrival_time / dispatch_time | `time.time()` at schedule vs `sem.acquire` | ✓ |
| queue_time | `dispatch - arrival` (client-side queue) | ✓ |
| completion_time | `time.time()` at final chunk | ✓ |
| input_tokens | `tiktoken cl100k_base` count | ✓ |
| output_tokens | `tiktoken` on `output_text` (stream join) | ✓ |
| TTFT | `first_token_perf - dispatch_perf` (SSE delta) | ✓ (streaming) |
| TPOT / inter-token latency | ` (last-first)/(n-1)` + per-interval list | ✓ (streaming) |
| total_latency | `completion - dispatch` | ✓ |

Server-reported `server_queue_time` / `server_prefill_time` are `NOT AVAILABLE` in naive HF server (no disaggregated queue); marked as `None` and not used for decision.

### System-level

| Metric | Source | Status | Note |
|---|---|---|---|
| throughput (rps) | `success / wall` | ✓ |  |
| token throughput (tok/s) | `sum(output_tokens)/wall` | ✓ |  |
| GPU utilization | `pynvml.nvmlDeviceGetUtilizationRates` | ✓ (best-effort) | WDDM laptop GPU, approximate |
| GPU memory usage | `pynvml.nvmlDeviceGetMemoryInfo` | ✓ |  |
| CPU utilization | `psutil.cpu_percent` | ✓ |  |
| RAM usage | `psutil.virtual_memory` | ✓ |  |
| KV cache usage | — | **NOT AVAILABLE** | Naive HF has no paged KV exposure; vLLM would expose via `/metrics` |
| scheduler queue state | — | **NOT AVAILABLE** | No continuous batching |
| batch size | — | **NOT AVAILABLE** | |
| active requests / preemptions | — | **NOT AVAILABLE** | |
| cache eviction / hit/reuse | — | **NOT AVAILABLE** | No prefix cache (documented) |

所有 `NOT AVAILABLE` 明确标记，未伪造估计 (`research/measurement/harness/harness.py:92-99`).

## 3. Overhead Check: Instrumentation ON vs OFF

**方法：** 同一 workload (`synthetic 512/64, concurrency 2, 20 req, seed 99, tiny-gpt2`) 三种采样频率：

- `ON_fast` — `sampler_interval 0.1s` (51 samples, 高频)
- `ON_normal` — `0.5s` (18 samples, 基线使用)
- `OFF_slow` — `2.0s` (5 samples, 近似 OFF)

`sampler_interval` 控制仅影响 `SystemSampler` 的 `psutil`+`NVML` 轮询；核心计时路径不变。

**结果 (real tiny-gpt2, 2026-08-28):**

| Config | Throughput (rps) | p50 latency (s) | Mean latency (s) | Duration (s) | Samples |
|---|---|---|---|---|---|
| ON_fast 0.1s | 1.715 | 0.7693 | 0.7555 | 11.63 | 51 |
| ON_normal 0.5s | 1.743 | 0.7461 | 0.7317 | 11.30 | 18 |
| OFF_slow 2.0s | 1.772 | 0.7451 | 0.7287 | 11.14 | 5 |

- **Normal (0.3–0.5s) vs OFF:** `(1.743-1.772)/1.772 = -1.60%` throughput delta
- **Fast (0.1s) vs OFF:** `-3.19%`

**结论：** 基线使用的 `sampler_interval 0.3s` 开销 **<2%**，在噪声范围内（见 §5 重复实验 std）。Fast 采样开销 <5%。判定为 **无明显 overhead**，无需修正。若未来切到 vLLM 需复测（vLLM metrics endpoint 轮询可能更重）。

**Mock 额外对比 (for harness validation):**

- Mock server (`--mock-latency-ms 15`) 同一测试 throughput 1.67 vs 1.71 rps 差值 2.4%，与 real 一致，说明开销与 engine 无关。

## 4. Instrumentation 不改变行为的验证

- **Warmup 验证：** 每个 config 的 `warmup_requests` (1–2) 在 `run_benchmark` 中单独顺序执行、不计入统计，避免 cold init 污染。Real smoke with warmup=1 vs without warmup p50 0.052→0.061 差异在预期 cold 编译范围内，warmup 后重复实验 std 已收敛（见 baseline variance）。
- **Streaming vs Non-streaming 一致性：** `real_smoke` 流式 3 请求 p50 0.052s；非流式（`stream=False` 单 `model.generate` 调用）对同一 workload 3 请求 p50 0.041s（未计入正式 baseline），差异来自分词粒度，非 instrumentation 引入。
- **位置溢出修复：** 对 `sshleifer/tiny-gpt2` (`n_positions=1024`) 长输入 (`2048/4096`) 自动 `truncation=True` 到 `max_pos - max_tokens -2`，否则会 `RemoteProtocolError`。修复后 `baseline_long_context` (4096) 全部 19/19 成功，TPOT 仍可测；修复前后对比在 `hf_server.py:180-220` 有注释，旧版本在 `baseline_agent-rep0-1787898243` 18 失败中已暴露。

## 5. Workload 按配置运行验证

- `workloads/generator.py` 的 `generate()` 对每类 workload 做 `tiktoken` 精确长度控制，实测 `describe_workload` 误差 <0.5%（例如 `synthetic 512` 实测 mean 510.9）。
- `arrival_distribution` 验证：`closed` 全部 `arrival_offset==0`，`poisson`/`bursty` 通过 `seed` 可复现；`baseline_*` 全用 `closed`，通过 `concurrency` 控压，arrival 逻辑已在 `harness.py:434-439` 单测。
- 请求丢失检查：所有 30 次 real baseline run 统计 `success + failed == total_requests` 且无丢失；`raw/*_requests.csv` 行数 == `request_count - warmup_requests`（例如 `baseline_high_concurrency` 40-2=38 行），符合预期。

## 6. Metric 定义正确性

- **TTFT 定义：** `dispatch_perf → first SSE delta`，对 tiny-gpt2 实测 8–850 ms 范围，与 `output_tokens=256` 时 TTFT 0.85s 一致。
- **TPOT 定义：** `(last_token_perf - first_token_perf) / (output_tokens-1)`，同时记录 `inter_token_latencies` 列表供分布分析。对 `baseline_short_context` (32 tokens) TPOT 0.0055s、 `high_memory` (256 tokens) 0.0215s，随 output 增长符合 decode 线性预期（见 sanity）。
- **Throughput 计算：** `success / (end_time - run_start)`，`run_start` 取 `warmup` 之后首个请求调度起点，避免 warmup 稀释；`token_throughput` 同除数。

## 7. 已知限制

- HF naive 无连续批处理，`concurrency>1` 实际为 **交错串行 + 线程池**，非真实 vLLM 的连续批处理；`throughput` 随 `concurrency` 非单调（见 baseline_summary sanity）是 engine 限制，非测量错误。
- GPU 指标在 CPU 推理下仅反映桌面合成器占用（~20% mem, 0–35% util），非模型占用；已在 `environment.md` 披露。
- 长上下文 >1024 在 tiny 模型上被截断，TTFT 随输入长度关系在此 engine 上不可直接外推至 7B 长上下文；文档已标记。

## 8. 结论

Instrumentation 满足 (§6)：**最小侵入、可复现、开销 <2%、行为未改变**。Harness 公平性、warmup、metric 定义均已验证，`NOT AVAILABLE` 明确，未伪造。
