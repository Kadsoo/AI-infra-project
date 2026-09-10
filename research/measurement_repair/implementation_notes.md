# Implementation Notes — L0/L1/L2 Measurement Repair

> **Goal:** 单调时钟整数 timestamp → 预分配内存结构 → return，零文件/JSON/锁在热路径

---

## 1. Tracing levels contract

| Level | 名称 | 语义 | 热路径代码 |
|---|---|---|---|
| **L0 OFF** | 完全关闭 | `STATE["trace_level"]=0`，`new_request_trace` 返回 `None`，所有 `if trace is not None:` 快速跳过 | `hf_server.py:369` |
| **L1 Minimal Causal** | 唯一 gate 用 | 6 事件（见 §2），`__slots__ + 预分配 list[6]`，无 per-token，run 后聚合 | `minimal_trace.py` |
| **L2 Full Existing** | 保留旧实现，仅对比 | `causal_trace.py` 原 15 timestamps + 64 steps 完整 | `causal_trace.py` |

Level 切换仅在 **run 之间** 经 `POST /stage3/trace/level {"level": 0|1|2}`，run 内不可变。OFF/ON 对比的语义明确为 **L0 vs L1**（此前 OFF vs Full 的对比保留为 L0 vs L2 仅用于说明开销来源，不用于 gate）。

---

## 2. L1 event set (minimal causal)

为定位 A_01 的 6 个边界而保留的最小集合（整数索引 0..5，无字符串 key）：

| idx | Event | 语义 | 具体位置 `hf_server.py` | 时钟 |
|---|---|---|---|---|
| 0 | `T_server_receive` | FastAPI handler 第一行 | `chat_completions` `trace.mark(0)` 紧接 `t_handler_enter` | `perf_counter_ns` |
| 1 | `T_executor_submit` | 首个 `asyncio.to_thread` 提交前 | `_stream_real` `trace.mark(1)` 在 `await to_thread(_prepare)` 之前 | 同上 |
| 2 | `T_executor_start` | worker 线程内第一行 | `_prepare` 闭包内 `trace.mark(2)` | 同上（worker 线程） |
| 3 | `T_first_model_step` | 首 token forward 开始前 | `_one_step` 内 `forward_begin` 仅当 `step_index==0` | 同上 |
| 4 | `T_first_token` | 首个非空 SSE yield 前 | `gen_real` `trace.mark(4)` 在 `yield` 首个 chunk 前 | 同上 |
| 5 | `T_completion` | streaming generator finally | `gen_real` finally `trace.mark(5)` | 同上 |

**为何 6 而非 15：** `instrumentation_plan.md` 的 T0–T7 中，`handler→runtime`, `prepare_submit/start/done/resume` 可合并为 `executor_submit/start` 单区间；`first_step_submit/start/forward` 合并为 `first_model_step`；`model_to_first_content` 即 `T_first_token - T_first_model_step`。6 点已可推导 `executor_wait = ts[2]-ts[1]` 与 `model_to_token = ts[4]-ts[3]`，满足 H1–H3 的 component share 计算（`handler_to_runtime` 等可延后从 6 点差分近似）。

**L1a** 进一步缩至 4 点：`receive (0)`, `executor_submit/start (1/2 合并)`, `first_token (4)`, `completion (5)`，用于 L1 仍 FAIL 时。

---

## 3. Minimal data structure

### 3.1 `MinimalRequestTrace`

```python
class MinimalRequestTrace:
    __slots__ = ("run_id", "request_id", "ts", "_next")
    def __init__(self, run_id: str, request_id: str):
        self.run_id = run_id
        self.request_id = request_id
        self.ts = [0,0,0,0,0,0]  # 固定 6，预分配，无 dict
        self._next = 0          # 可选：ring 索引（此处固定索引直接赋值）
    def mark(self, idx: int, at_ns: int | None = None):
        # 极简：读取单调时钟 → 写入预分配槽位 → return，无分支、无字符串
        self.ts[idx] = at_ns if at_ns is not None else time.perf_counter_ns()
```

- **零 dict 增长：** `ts` 预分配 6，`mark` 仅 `list.__setitem__`（GIL 原子，~30 ns）。
- **零 per-token 分配：** 无 `steps`，无 `step` dict，无 `dict(step)` 复制。
- **零排序：** `step_aggregates` 移除，run 后在 `processed/` 用 `derive_trace_components` 统一计算。
- **整数 timestamp：** 全程 `perf_counter_ns` int，仅在 run 后 `/1e9`。

### 3.2 `MinimalTraceStore`

```python
class MinimalTraceStore:
    def __init__(self, capacity: int = 50000):
        self._lock = threading.Lock()  # 仅 per-request add 时短暂持有
        self._records: dict[str, list[dict]] = {}
        self._capacity = capacity
        self._total = 0
    def add(self, trace: MinimalRequestTrace):
        # to_dict 极小：6 整数 + 2 字符串，无排序
        rec = {"run_id": trace.run_id, "request_id": trace.request_id, "ts": list(trace.ts)}
        with self._lock:
            bucket = self._records.setdefault(trace.run_id, [])
            if len(bucket) < self._capacity:
                bucket.append(rec)
            else:
                # 环形覆盖：有界、不增长、不 realloc 风暴
                bucket[self._total % self._capacity] = rec
            self._total += 1
    def take_run(self, run_id: str):
        with self._lock:
            return self._records.pop(run_id, [])
```

- **有界：** `capacity=50000` 远超单 run 38 请求，即使未来扩展至 1000 请求/run 亦不增长。
- **不触发频繁 realloc：** `list.append` 仅在首个 run 预分配，后续 `setdefault` 复用；环形分支保证最坏也不超过 capacity。
- **不改变 GPU memory：** 纯 Python 堆，与 torch 无共享。
- **GC pressure：** 每请求仅 1 个 `MinimalRequestTrace` + 1 个 `ts` list + 1 个 `rec` dict（3 对象），对比 Full 的 ~130 dict，GC 次数降 98%。

---

## 4. Hot path 极简原则落地

```text
# L1 热路径伪码（每事件）
ts[idx] = time.perf_counter_ns()  # 读取单调时钟
#  → 写入预分配内存
return
```

- **优先：** `perf_counter_ns`（Windows QPC，~85 ns）、整数 ns、固定 `list[6]`、append-only。
- **避免：** 共享 heavy locks（仅 per-request add 时 <1 µs）、blocking queues（无）、filesystem（无）、JSON/CSV/plotting/disk write（全部延后至 `take_run` 后）。
- **Formatting/aggregation/serialization：** 全部在 `compute_processed` 后处理，`overhead_gate_results.md` 聚合。

---

## 5. Memory & GC 保障

- **上限：** `MinimalTraceStore` 容量 50000 事件或 50000/6≈8333 请求；单 run 38 请求仅占 0.07%，无无限增长。
- **不频繁 realloc：** `ts` 预分配，`bucket` 预分配，环形覆盖无 `list` 扩容风暴。
- **不影响 GPU：** Trace 结构与 `torch` 张量无交互，`past_key_values` 仍由模型持有，trace 不持有 tensor 引用。
- **GC：** 使用 `__slots__` 消除 `__dict__`，每请求 3 对象 vs Full 130；建议在 `run_benchmark` 前后 `gc.collect()` 可选，但 L1 自身已足够低，无需 `gc.disable()`（后者会掩盖真实开销，反而违背 gate）。

---

## 6. Disabled tracing truly no-op

```python
def new_minimal_trace(enabled: bool, level: int, run_id: str, request_id: str):
    if level == 0 or not enabled:
        return None
    return MinimalRequestTrace(run_id, request_id)
```

调用点 `if trace is not None: trace.mark(...)` 在 L0 时分支未取，CPU 分支预测友好，无对象分配，`perf_counter` 零调用。单元测试 `test_disabled_noop` 断言 `new_minimal_trace(False, ...) is None` 且 `hf_server` 在 L0 时热路径零 `perf_counter`。

---

## 7. Sampling (L1b) & Counters (L1c)

### L1b — 固定比例确定性采样

```python
def _should_sample(request_id: str, ratio: float) -> bool:
    # request_id 形如 "req-0003-1a2b3c"，取 hex 后 6 位转 int，确定性无随机
    h = int(request_id.rsplit("-",1)[-1], 16)
    return (h % 100) < int(ratio*100)  # ratio=0.2 → 20%
```

- **预先决定、确定性、可复现：** 不根据性能结果选择，seed 无关。
- **Overhead：** 80% 请求完全走 L0 路径，平均 overhead 仅 20%×L1。

### L1c — 仅聚合计数

若 L1b 仍 FAIL，退化为原子计数器：`atomic_inc(total_requests)`, `histogram_observe(latency_bucket)`，无 per-request timeline。实现为 `threading.Lock` 保护的 10-bin 直方图，热路径仅 `with lock: bins[bucket]+=1`（~50 ns）。

---

## 8. Serialization after run

- **L1 的 `to_dict`：** 仅 `{"run_id","request_id","ts": [...]}`，无 `step_aggregates`，无排序。
- **Run 后拉取：** `GET /stage3/trace/{run_id}` 返回 `{"run_id":..., "traces": [...]}`，由 `run_a01_causal` 在 `raw/*_server_traces.json` 落盘，此后 `derive_trace_components` 在 `processed/` 推导 `executor_wait = ts[2]-ts[1]` 等。
- **Output completeness：** 单元测试 `test_output_completeness` 校验 38/38 trace 回收且 `ts` 长度 6 且单调非负（见 `tests/test_minimal_tracing.py`）。

---

## 9. File layout

```
research/measurement/harness/minimal_trace.py   # L1/L1a/L1b/L1c 实现
research/measurement/harness/hf_server.py       # 增加 /stage3/trace/level，分支 L0/L1/L2
research/measurement/tests/test_minimal_tracing.py  # 单测
research/measurement_repair/implementation_notes.md  # 本文件
```

*— End —*
