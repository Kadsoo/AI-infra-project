# Tracing Hot Path Audit — Stage 3M

> **Audit date:** 2026-08-28  
> **Audited code:** `research/measurement/harness/hf_server.py:252-502`, `research/measurement/harness/causal_trace.py:44-116`  
> **Workload:** synthetic 512/64, `stream=True`, `max_tokens=64`, `concurrency=1/4/8`, `request_count=40` (38 measured)  
> **Engine:** `hf_transformers_naive` CPU FP32 `sshleifer/tiny-gpt2` (prior gate PID 18968)  
> **Gate evidence:** `research/stage3/A_01_causal/processed/overhead_gate.json` FAIL, `raw/*_server_traces.json` (38 traces/run, 15 timestamps + 64 steps per trace)

---

## 0. Executive summary

现有 tracing 在 **热路径频率 = per-token / per-step（64 次/请求）** 记录了与 A_01 因果定位所需的 6 个边界事件相比 **~10× 的冗余数据**。单请求热路径涉及 **~394 次 `perf_counter_ns` + `threading.get_ident` 调用 + 64 个 dict 分配 + 15 个 timestamp dict 写入 + 64 次 `steps.append(dict(step))` + 结束时的 3 次排序统计**。微基准测得单请求 CPU 额外 0.14 ms（不含锁/ GC），占 300 ms 端到端延迟的 <0.05%，但在 **高并发 + 默认线程池 + Python GC** 叠加下，观测到 **间歇性长尾暂停（0.09 s → 0.94 s TTFT，+163% p95）**，与 host 线程数从 103 → 247 及 RSS 45 MB → 1024 MB 的漂移并发。判定为 **measurement-induced non-stationarity**：审计判定当前 tracing **不满足 Stage 3M 的 10% / multiplier 10% gate**，需降至 L1 极简。

---

## 1. Overhead source inventory

### 1. Synchronous file I/O

**Location:** `hf_server.py:500-501` / `causal_trace.py:103-116` 的 `TraceStore.add` 仅在请求完成后的 `gen_real` finally 内执行 `trace.to_dict()` 并 `records.setdefault(...)`，真正的文件 `json.dump` 发生在 **run 结束后** 的 `/stage3/trace/{run_id}` 拉取（`run_a01_causal.py:95-108`）。热路径无 `open`/`write`/`flush`。

**Behavior:** 内存写入 + 列表 append。

**Frequency:** per request（1 次）

**Expected Overhead Risk:** **Low** — 无同步 I/O，但见 §7 锁。

**Proposed Fix:** 保留“仅内存、run 后序列化”原则，L1 亦遵循。无需修改。

---

### 2. JSON serialization

**Location:** `hf_server.py:363` 的 `_sse_format` 是 serving 必要路径（所有请求无论 tracing ON/OFF 均执行），不属于 tracing 引入。Tracing 引入的 JSON 仅在 `TraceStore.add → trace.to_dict()` 构建 dict 后，**run 后**在 `take_trace` 返回时由 FastAPI 自动序列化。

**Frequency:** per request（构建 dict）+ per run（序列化 38 traces）

**Risk:** **Medium** — `to_dict` 构建 17 KB dict/请求（64 steps × 8 fields），38 请求 ≈ 650 KB/run，排序与 dict 复制在热路径结束前。

**Fix (L1):** 移除 `steps`，移除 `step_aggregates` 的 3 次排序（`statistics.mean` + `sorted`），L1 仅存储 6 个整数 timestamp（见 §3），`to_dict` 仅返回 6 字段扁平 dict，run 后再派生统计。完全移除 per-token JSON 结构。

---

### 3. Print / logging / flush

**Location:** `hf_server.py:340` 的 `print("[hf_server] _stream_real step failed: ...")` 仅在异常路径；`load_model` 的 prints 仅在启动。

**Frequency:** other（异常）

**Risk:** **Low**

**Fix:** 保持现状，L1 热路径零 print。

---

### 4. Lock contention

**Location:** `causal_trace.py:103-108` `TraceStore._lock = threading.Lock()`，被 `TraceStore.add`（每请求 1 次）与 `take_run`（每 run 1 次）持有。

**Behavior:** `add` 持有锁期间执行 `records.setdefault(...).append(record)`，而 `record = trace.to_dict()` 已在锁外预先计算，但 `append` 仍在锁内。对 c=1 串行无竞争；对 c=4/c=8 有 4–8 并发 `add` 争用（38 请求在 42 s 内完成，平均并发度 = 吞吐×p50 ≈ 0.94×2.6 ≈ 2.4，峰值 4–8）。微基准锁持有 < 5 µs，但结合 Python GIL，高并发下与 `asyncio.to_thread` 的线程池队列竞争形成二级 contention。

**Frequency:** per request

**Risk:** **Medium** — c=1 不显著，c=4/8 叠加线程池等待会放大 p95 抖动（见 §13）。

**Fix (L1):** 缩短临界区至单条 `list.append`；或去锁：利用 CPython `list.append` 的 GIL 原子性 + `dict.setdefault` 的 GIL 保护，或采用 `collections.deque`。L1 的 payload 更小，锁持有时间 < 1 µs。若极端敏感，可采用 **per-run 预分配 list + 原子索引**（`itertools.count` + `__setitem__`）完全无锁，但当前评估单锁已足够，只需避免在锁内做排序/复制。

---

### 5. Queue contention（executor queue）

**Location:** `hf_server.py:280` `await asyncio.to_thread(_prepare)` 与 `hf_server.py:334` `await asyncio.to_thread(_one_step, ...)` 均提交至 `asyncio` 默认 `ThreadPoolExecutor`。Tracing 在 `_one_step` worker 内额外执行 `perf_counter_ns` ×4 + `threading.get_ident` + dict 赋值。

**Behavior:** 每个 token 的 forward 前后各贡献一次 `_one_step` 提交。Tracing 增加 worker 内指令，但不增加 queue 入队次数（仍为 1 次 prepare + 64 次 steps = 65 次 `to_thread`/请求）。默认 executor 最大线程数 `min(32, cpu_count+4)=28`（实测初始化后线程池从 103 → 247 OS 线程，含 torch 线程），65×38=2470 次提交/run，在 c=8 时提交速率 ≈ 2470/42≈58 提交/s，queue 深度通常 < 4，但 tracing 的 worker 内额外指令（约 0.5 µs）会延长占用，使 queue 等待 `t_worker_start - t_submit` 的分布右尾增厚。

**Frequency:** per token (64/请求) 的 worker 内额外指令；per request 65 次提交

**Risk:** **Medium–High** — 是唯一真正 per-token 热路径，且与模型 forward（35 ms 首 token，3–5 ms 后续 token）共享同一 executor。观测到的首 token forward wall 在异常请求中从 35 ms 膨胀至 83 ms（+137%），与此相关。

**Fix (L1):** **完全移除 per-token 捕获**。L1 仅捕获 **首 token 前后** 的 2 次 `to_thread`（prepare + first step），并在 worker 内仅捕获 **3 个**整数 timestamp（worker_start, forward_begin, forward_end），不分配 `step` dict，不调用 `threading.get_ident`。将 per-request `to_thread` 内 tracing 指令从 64×7=448 次降至 2×3=6 次（98.7% 减少）。

---

### 6. String formatting / header handling

**Location:** `hf_server.py:367-375` 的 `X-Stage3-Run-Id` / `X-Stage3-Request-Id` header 提取与回显；`run_a01_causal.py:246-250` 的 `X-Stage3-*` 发送。

**Behavior:** 每请求 2 个 header 的 dict 查找 + 字符串复制。

**Frequency:** per request

**Risk:** **Low** — OFF 亦发送相同 headers（`instrumentation_plan.md` 要求），OFF/ON isolate 的是 trace object 而非 header 形状，已正确控制。

**Fix:** 保留，用于关联；不计入 L1 overhead。

---

### 7. Python object allocation / dictionary growth

**Location:** `causal_trace.py:47-59` `RequestTrace.__init__` 创建 `timestamps={}` + `steps=[]`；`hf_server.py:287-303` 每 step 创建 `step={}` 并多次 `step["..."]=...`；`causal_trace.py:58-59` `add_step` 再次 `dict(step)` 复制。

**Behavior:** 每请求分配 1 个 `RequestTrace` + 15 次 `timestamps.__setitem__` + 64 个 `step` dict（各 7–8 键）+ 64 次 `dict(step)` 复制 + `to_dict` 时 64 次 `dict(s) for s in steps` 二次复制。共 **约 130 个 dict + 500 个 int** 分配/请求。38 请求/run → ~5000 dict/run，触发 Python GC 代际回收。

**Frequency:** per request + per token

**Risk:** **High** — GC 暂停是观测到的 0.94 s TTFT 长尾（正常 0.07 s 首 token 中 server_sfc 仅 0.09 s，client 额外 0.85 s 含 queue + 客户端创建）的最可解释来源。`harness.py` 另有 `httpx.AsyncClient` 每请求新建（300 ms 创造成本），与 GC 暂停共振。

**Fix (L1):** 使用 `__slots__` + 固定长度 `list[int]` 预分配。L1 `MinimalRequestTrace` 结构：

```python
class MinimalRequestTrace:
    __slots__ = ("run_id","request_id","ts")  # ts: list[int] 长度 6，预分配
    def __init__(self, run_id, request_id):
        self.run_id = run_id
        self.request_id = request_id
        self.ts = [0]*6  # 预分配，无 dict 增长
    def mark(self, idx: int):  # idx 0..5, 无字符串 key
        self.ts[idx] = time.perf_counter_ns()
```

零 `step` dict，零 `timestamps` dict 增长，零 `dict(step)` 复制。GC 压力从每请求 ~5000 对象降至 1 对象 + 1 list。

---

### 8. Per-token / per-step logging

**Location:** `hf_server.py:286-354` 的 `for step_index in range(max_tokens): ... trace.add_step(step)` 循环。

**Frequency:** per token (64/请求)

**Risk:** **High** — 见 §5、§7。

**Fix:** L1 移除该循环内的所有 tracing；仅保留 `step_index==0` 的首 token 路径，其余 `step_index>=1` 完全无 tracing 分支（`if trace is not None and step_index==0:`）。

---

### 9. Repeated clock conversion

**Location:** `a01_causal.py:84-85` `_seconds(delta_ns: int) → delta_ns/1e9` 在 `derive_trace_components` 中对每个 component 做除法；`hf_server.py` 中所有 `perf_counter_ns / 1_000_000_000` 仅在 `harness.py` 的客户端 TTFT 计算时转换，服务端保持整数 ns。

**Frequency:** per run 后处理（不在热路径）

**Risk:** **Low**

**Fix:** 保持整数 ns 贯穿热路径，仅在 run 后的 `processed/` 阶段转换。

---

### 10. Expensive stack/context capture

**Location:** `hf_server.py:298` `threading.get_ident()` 每 step 捕获 worker thread id（用于诊断 threadpool）。

**Frequency:** per token

**Risk:** **Medium** — `get_ident` 单次 ~59 ns（微基准），64 次/请求 ≈ 3.8 µs，可忽略，但与 `perf_counter` 叠加且无因果必要性。

**Fix:** L1 移除；如需诊断，仅在首 token 捕获 1 次。

---

### 11. Network export

无。Trace 拉取仅在 run 结束后经 `/stage3/trace/{run_id}` HTTP GET，非热路径。

**Risk:** **Low**

---

### 12. Tracing callbacks / hooks

**Location:** `hf_server.py:369-377` 的 `new_request_trace(enabled= STATE["trace_enabled"] and ...)`；每个 `_stream_real` 对 `if trace is not None:` 分支 20+ 处。

**Behavior:** 分支预测友好，但当 `trace is None`（L0 OFF）时仍需执行 `if trace is not None:` 判断（~10 ns）。

**Frequency:** per request（handler）+ per token 累积

**Risk:** **Low** for OFF，但 L1 需减少分支次数。

**Fix:** L1 将分支从 per-token 20 次降至 per-request 6 次（首 token 3 个 mark + handler 2 + done 1），分支开销可忽略。且保证 **OFF 真正 zero-cost**：`trace_enabled==False` 时不分配对象、不执行任何 `mark`，`if trace is not None:` 快速返回。

---

### 13. Excessive instrumentation points

**现状：** 15 个命名 timestamp + 64×7 step 字段 = **463 个** per-request 写入点。要定位 A_01 的 knee，需 minimal causal tracing 的 6 个事件即可：`T_server_receive (handler_enter)`, `T_executor_submit`, `T_executor_start`, `T_first_model_step (forward_begin)`, `T_first_token (first_content_yield)`, `T_completion (server_done)`。其余 457 点为诊断冗余。

**Risk:** **High** — 是 overhead gate 失败的 **首要结构原因**。

**Fix (L1):** 精确 6 点，使用整数索引而非字符串 key，写入预分配 `ts[6]`。

可选的 **L1a** 进一步缩至 4 点：仅 `receive`, `executor_submit/start`, `first_token`, `completion`（合并 model-step 区间）。

---

## 2. 按频率分层的风险矩阵

| 频率 | 当前写入点数/请求 | 实测额外 CPU (微基准) | 观测到的尾延迟影响 | 修复优先级 |
|---|---|---|---|---|
| per token (64×) — step dict + 4 clocks + get_ident | 448 | ~0.13 ms | 首 token forward 35→83 ms (+137%) 在异常请求 | **P0 必须移除** |
| per request — 15 timestamps + to_dict 排序 | 15 + 3 sorts | ~0.14 ms | p95 TTFT 0.099→0.262 (+163%) 在 c=1 seed3101 | **P0** |
| per scheduler iteration (prepare) | 4 clocks | ~0.001 ms | 可忽略 | P1 合并 |
| shared locks | 1/contended | <0.005 ms | c=4/8 抖动 +5% | P1 缩短临界区 |

---

## 3. 与观测到的 Gate 失败的关联

Gate 失败细节（`processed/overhead_gate.json`）：

- **c1 seed3101 OFF→ON p95 TTFT +163.72%**（0.099→0.262 s），**p95 total +85.99%**，远超 10% 个体阈值；同 c1 seed3102 ON→OFF 仅 +4.64% / +9.94%，显示 **非对称、非稳定**：符合 GC 暂停的随机长尾（38 样本中 2 个请求 TTFT 0.94 s/0.55 s，正常尾 0.09–0.10 s）。
- **c4/c1 TTFT multiplier OFF 28.68 → ON 15.17，变化 -47.1%**（阈值 10%），说明 tracing 改变了 knee 的关键度量，使 A_01 归因失真。
- 线程数从 103→247、RSS 45→1024 MB 的单调漂移，以及 `dispatch→slot` gap 0.73 s→0.92 s 的增加，表明 **host non-stationarity 与 instrumentation 交互**，而非单纯 per-request 0.14 ms 可解释，需同时通过 **预热 + 固定 buffer + 有界内存** 修复。

---

## 4. Proposed Fix 分层

| Level | 记录内容 | 热路径写入 | 预期 overhead | 适用 gate |
|---|---|---|---|---|
| **L0 OFF** | 无 | 零（`trace is None` 快速返回） | 0% baseline | 对照 |
| **L1 Minimal** | 6 整数 timestamp（handler_enter, prepare_submit, prepare_worker_start, first_forward_begin, first_content_yield, server_done） | 6× `perf_counter_ns` + 6× `list.__setitem__` | <1%（微基准 0.67 µs/请求） | 主 gate |
| **L1a Ultra-minimal** | 4 点（receive, executor_submit/start, first_token, completion） | 4× | <0.5% | 若 L1 仍 FAIL |
| **L1b Sampled** | L1 的 10–20% 采样（确定性 `hash(request_id) % 100 < 20`） | 平均 1.2×/请求 | <0.2% | 若 L1a 仍 FAIL |
| **L1c Counters** | 仅原子计数器 + 直方图（无 timeline） | 1× increment | ≈0% | 最后手段 |
| **L2 Full** | 现有 15 + 64×7 完整（保留用于对比） | 463× | 已证 FAIL | 仅离线对比，不用于 gate |

L1 的实现约束（`implementation_notes.md` 详述）：单调时钟 `perf_counter_ns`、整数 ns、固定大小 `__slots__` + 预分配 `list[6]`、append-only、无共享重锁、格式化/聚合/JSON/CSV 均延后至 run 后、`buffer 上限 50000 事件` 环形覆盖。

---

## 5. 检查清单（Audit completeness）

- [x] synchronous file I/O — 无热路径
- [x] JSON serialization — 热路径外，但 `to_dict` 含排序，已标记
- [x] print/logging/flush — 无热路径
- [x] lock/queue contention — 已定位
- [x] string formatting — 无热路径
- [x] Python allocation/dict growth — 高风险，已量化
- [x] per-token/step logging — 首要原因
- [x] repeated clock conversion — 低
- [x] stack/context capture — 已定位
- [x] network export — 无
- [x] tracing callbacks — 已定位
- [x] excessive points — 首要

*— End of audit —*
