# Overhead Gate Experiment Plan — Stage 3M

> **Gate objective:** 证明 **L1 Minimal** 相对于 **L0 OFF** 不显著改变系统行为，方可重新执行 A_01 causal localization  
> **Locked workload (与 A_01 相同):** `synthetic` `512 input / 64 output / no prefix reuse` `40 requests (2 warmup + 38 measured)` `stream=True` `temperature=0.0` `timeout 180` `sampler_interval 0.3 s` `model sshleifer/tiny-gpt2 CPU FP32`  
> **Fixed:** `serving PID` 单进程, `config`, `prompt set`, `arrival pattern closed`, `seed`, `warmup 2 sequential`, `hardware` LAPTOP-1PB54QSI  
> **Tested concurrencies:** **c=1, c=4**（首轮不跑 c=8，见 §7）

---

## 1. Tracing levels under test

| Condition | Server level | 含义 | 预期 overhead |
|---|---|---|---|
| **OFF (L0)** | `POST /stage3/trace/level {"level":0}` | 零 tracing，`trace is None` | baseline |
| **L1** | `{"level":1}` | 6 整数 timestamp minimal（`minimal_trace.py`） | <1% |

L2 Full 保留但不参与 gate，仅用于审计对比。Gate 判定仅 **OFF vs L1**。

---

## 2. Paired randomized / interleaved design (mandatory)

**禁止** `OFF OFF OFF / ON ON ON` 顺序。采用 **matched-seed counterbalanced**：

| Pair | Concurrency | Seed | Order | 目的 |
|---|---|---|---|---|
| A | c=1 | 3101 | **OFF → L1** | 同 workload/seed，消除 prompt 差异 |
| B | c=1 | 3102 | **L1 → OFF** | 平衡 run order |
| C | c=1 | 3103 | OFF → L1 | 重复，检测稳定性 |
| D | c=1 | 3104 | L1 → OFF | 重复 |
| E | c=4 | 3401 | OFF → L1 | 同上 |
| F | c=4 | 3402 | L1 → OFF |  |
| G | c=4 | 3403 | OFF → L1 |  |
| H | c=4 | 3404 | L1 → OFF |  |

- **总 run 数：** 16 runs（c1×8 + c4×8），8 个 matched pairs。
- **每个 matched pair 使用相同 workload/seed：** `generate(synthetic, n=40, input 512, output 64, seed)` 确定性生成，`describe_workload` 记录哈希。
- **交错执行顺序示例（全局）：** `A_OFF, A_L1, B_L1, B_OFF, E_OFF, E_L1, F_L1, F_OFF, C_OFF, C_L1, D_L1, D_OFF, G_OFF, G_L1, H_L1, H_OFF` — 使 instrumentation condition 与 run order 正交。实际由 `run_measurement_repair_gate.py` 随机化但保证每个 pair 内 OFF/L1 相邻。
- **Seed 预注册：** 3101–3104（c1）、3401–3404（c4）为 A_01 的延续（3101/3102/3401/3402 复用，3103/3104/3403/3404 新增），不可事后挑选。

---

## 3. Warmup & host stabilization

1. **Model 已加载：** `GET /health` `loaded=true` `mock=false` `device=cpu`。
2. **Kernel warmup：** gate 前执行 2 次 dummy `c=8, 20 requests` 不计入，使线程池从 103 → ~240 稳定（见 `host_state.md §2.1`）。
3. **Serving process 稳定：** `health` 的 `torch_num_threads` 固定 16/16，`process_threads` 在 warmup 后稳定 ±5。
4. **Benchmark client warmup：** 每 run 的 `warmup_requests=2` 顺序执行，排除 cold-start。
5. **Host-state gate：** 每 run 前记录 `gpu_util`, `gpu_mem`, `cpu_freq`, `ram`, `server_threads`；若 `invalid`（`host_state.md §4`）则该 pair 标记 invalid，不纳入 gate 统计。

---

## 4. Gate metrics (at least)

每个 run 的 `processed/*_processed.json` 报告：

| Metric | 定义 | 来源 |
|---|---|---|
| `median_ttft` | `p50` TTFT | `harness` client `first_token - dispatch` |
| `p95_ttft` | `p95` TTFT | 同上 |
| `median_latency` | `p50 total_latency` | `completion - dispatch` |
| `p95_latency` | `p95 total_latency` | 同上 |
| `throughput_rps` | `success / (end - run_start)` | `RunResult` |
| `token_throughput` | `sum(output_tokens)/wall` | 同上 |

同时记录 **absolute Δ** (`L1 - OFF`) 与 **relative Δ** (`L1/OFF -1`)，**不可仅看百分比**（低 latency 时 10 ms 绝对差亦重要）。

---

## 5. Gate criteria (10% principle, locked)

沿用 `LOCKED_CAUSAL_PLAN.md` 原则，**不得事后改变阈值**：

### 5.1 Per-metric overhead

对 `c=1` 与 `c=4` 分别，L1 相对 OFF 的 **median paired Δ** 需基本控制在 **10% 以内**，且结果稳定。

Formal 判定（与 `a01_causal.py:evaluate_overhead_gate` 一致，但放宽个体至 10% 与 median 至 5% 的 **双阈值**，此处 Stage 3M 采用更直观的 10% 统一阈值）：

- **Median gate：** 8 个 pairs 的 `relative Δ` median 绝对值 ≤ **5%**（`throughput`, `p95_ttft`, `p95_latency` 三指标）。
- **Individual gate：** 每个 pair 的 `relative Δ` 绝对值 ≤ **10%**。
- **任一失败即 FAIL。**

> 注：Stage 3M 优先使用 10% 作为硬阈值，median 5% 作为稳定性补充，与 `result.md` 的 “10% 以内” 及 `instrumentation_plan.md` 的 “5% median / 10% individual” 双规则等价。

### 5.2 Knee preservation

`c4/c1` 的 knee-related multiplier（以 **p95 TTFT** 为主，辅以 `p95_latency`）：

```
off_multiplier = median_OFF(c4 p95_ttft) / median_OFF(c1 p95_ttft)
on_multiplier  = median_L1(c4 p95_ttft)  / median_L1(c1 p95_ttft)
relative_change = on/off -1
```

要求 `|relative_change| ≤ 10%`。若 L1 使 `c4/c1` 从 28.6→15.1（-47% 如 reference gate），则 FAIL。

### 5.3 Stability

8 个 pairs 的 `relative Δ` 不应出现 **`+5%` 与 `+150%` 混杂** 的高度不稳定。量化：`relative Δ` 的 `std` 应 < 10% 且无单点 >3σ 远离 median。若出现，判 **FAIL — ENVIRONMENT NON-STATIONARY** 而非 instrumentation effect。

---

## 6. Matched-pair table (template)

Gate 报告必须生成下表（`overhead_gate_results.md`）：

| Seed | Order | Concurrency | OFF p95_ttft | L1 p95_ttft | Absolute Δ (s) | Relative Δ (%) | Valid |
|---|---|---|---|---|---|---|---|
| 3101 | OFF→L1 | 1 | 0.0992 | 0.101? | +0.002 | +2.0% | YES |
| 3102 | L1→OFF | 1 | 0.0820 | 0.0858 | +0.0038 | +4.6% | YES |
| ... | ... | ... | ... | ... | ... | ... | ... |

重点检查 **run-order effect**：若 **第一个 run 总快/慢** 无论 OFF/L1，则判 host drift（见 §7）。

---

## 7. Separation of concerns (must report separately)

### Instrumentation Effect

L1 是否改变系统？由 **paired Δ**（同 seed 同 order 内 OFF vs L1）回答。

### Environment Non-Stationarity

即使 tracing OFF，run-to-run 是否漂移？由 **OFF-only runs 的 CV**（`c1 seed3101 OFF` vs `seed3102 OFF` 等）与 **order effect** 回答。若 OFF 之间 CV >10% 或首 run 系统性偏差，则即使 L1 median 通过，也应判 **FAIL — ENVIRONMENT NON-STATIONARY**。

二者 **不得混合**：不可用 L1 的 variance 去掩盖 OFF 的 drift。

---

## 8. Progressive fallback if L1 FAIL

按序降低 instrumentation，每级重跑 **完整 gate（8 pairs）**：

| Level | 描述 | 采样 |
|---|---|---|
| **L1a** | 4 timestamp（receive, executor, first_token, completion） | 100% |
| **L1b** | L1 的 10–20% 确定性采样（`hash(request_id)%100<20`） | 20% |
| **L1c** | 仅聚合计数/直方图（无 timeline） | 100% |

采样必须 **预先决定、确定性、可复现**，不得根据性能结果选择。

若 L1–L1c 均 FAIL → 标记 **FINE-GRAINED TRACING NOT VIABLE**，下一阶段改用 `framework built-in metrics / external timing / aggregate counters / black-box`（见 `measurement_repair_summary.md`）。

---

## 9. Execution command (frozen)

```bash
# 1. Start server with L1 support (new port, isolated)
python research/measurement/harness/hf_server.py \
  --model sshleifer/tiny-gpt2 --device cpu --port 8020

# 2. Run gate (c1+c4, 16 runs, interleaved)
python research/measurement/harness/run_measurement_repair_gate.py \
  --base-url http://127.0.0.1:8020 \
  --phase gate --out research/measurement_repair --concurrency 1,4
```

输出：`raw/*_requests.csv`, `raw/*_system.csv`, `raw/*_raw.json`, `raw/*_server_traces.json`（L0 为空，L1 为 6-point），`processed/*_processed.json`, `processed/overhead_gate.json`, `logs/*_events.jsonl`, `host_state.csv`。

---

## 10. Output contract

`research/measurement_repair/` 至少包含：

- `tracing_audit.md` ✓
- `host_state.md` ✓
- `implementation_notes.md` ✓
- `gate_plan.md` ✓（本文件）
- `raw/`（16 runs）
- `processed/`（`overhead_gate.json`）
- `overhead_gate_results.md`（matched-pair 表 + 判定）
- `measurement_repair_summary.md`（最终分类）

*— End of gate plan —*
