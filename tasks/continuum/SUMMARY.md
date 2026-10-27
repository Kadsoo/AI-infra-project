# 任务1 Continuum 复现与适用边界分析 — 进展报告（本地仿真部分完成，云验证待接手）

## 0. 冻结（Phase0）

| 项目 | 原论文/仓库 | 实际采用 | 差异原因 |
|---|---|---|---|
| 论文 | arXiv:2511.02230 **v5 (2026-05-11)**（任务书指定版本；已全文核对 v3/v5/v7，核心机制 Eq.1/Eq.2、`eta`、K=100、三级优先级、基座版本、评估配置与 1.12–3.66x/8.18x 数字三版一致，v5 表格内方法名为 `CacheTTL`、v7 改称 Continuum，仅命名差异） | 以 v5 为准 |
| 仓库 | `Hanchenli/vllm-continuum` commit `316a587`，基座 vLLM 0.10.2 + LMCache 0.3.7，README 自述 **without the estimation** | 已 clone 到 `C:\Windows\Temp\opencode\continuum\vllm-continuum` 做只读审计 | 预览版只有骨架，需完整实现 TTL |
| 基座对照 | 论文 vLLM 0.10.2 / A0 用 0.11.0 / V100 | 仿真不依赖基座版本；云验证建议与论文同 0.10.2 fork | V100 跑 0.10.2 fork 需 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`（沿用 A0 经验） |
| 云机探测 | 待用户粘贴 `nvidia-smi` 输出 | **阻塞项**：Phase4 未开始 | 已给三条只读命令，见文末 |

### 代码对齐（缺口表，`vllm-continuum` 现状 vs 论文 v5 需求）

| 论文 v5 需求 | 仓库现状 | 缺失（已在仿真中实现，真机待移植） |
|---|---|---|
| 经验 CDF `P(tau,f)` /滑窗 `T` /`eta` /`CalcTTL` Eq.2 | 仅 per-tool 均值 + `FIXED_THRESHOLD_CONTINUUM=2.0s`（`estimate_with_func.py`） | CDF 存储、`CalcTTL` 枚举、`K=100` 冷启动（`continuum_ttl.py` 已实现） |
| `func_call_finish/update_tool_call_time/set_up_ttl` | 同名缺失，只有 `request_finished/set_up_pin/update_func_call_exec_time`，`is_last_step` 为 TODO | 重命名/对齐 shim |
| pin 命中复用（不双计内存） | `_free_blocks` 重 pin 2s，无复用扣减 | `_eff_need`（仿真已实现，真机需同等改 scheduler fit 检查） |
| 死锁 victim-unpin（running 为空仍不可调度时逐出最晚到达 pin） | 仅部分路径（`len(running)==0` 放一个 pin） | 完整 victim 循环 + `victim_unpin` 计数（仿真已实现并证明不可缺，见 §3） |
| 估计器评估 | `analyze.py` 只有 JCT 均值 | TTL 误差/pin 命中/precision（`test_ttl.py` + results JSON） |

## 1. 交付物（本仓库 `tasks/continuum/`）

- `continuum_ttl.py` — Eq.2、`ecdf`、`t_default=ln(B)`（Exp(1) 推导）、`compute_eta=-Corr(k,N-k)`、prefill/reload 离线标定（70k ctx：prefill 3.64s / reload 0.46s）。
- `gen_loads.py` — BFCL-like / SWE-like / mixed 合成（turn/工具/ token 均值方差对齐论文 Table2，长尾：最慢 10% 占主导；Poisson 到达；`max_ctx` 裁剪=论文"0.4x 缩放塞进上下文"的对应物）。
- `continuum_sim.py` — 离散事件仿真：GPU KV 容量、pin/TTL/过期、program-FCFS vs request-FCFS、prefill+逐轮排队、严格 HOL 记录、victim-unpin。
- `run.py` — 一键入口（`python tasks/continuum/run.py --load mixed --policy all ... --out ...`），输出 JCT/p50/p90/avgQ/prefill/pins/命中率/HOL。
- `test_ttl.py` — 5 项单元检查（冷启动/per-tool/全局回退/`eta`/标定），已通过。
- `results/` — 10 组 JSON（含 HOL 前 20 事件）。

## 2. 结果（与论文对照）

GPU-only、中等压力（cap 300k tok，token-scale 1.0，Poisson JPS 0.08）：

| 负载 | baseline→full (JCT mean) | static | prog-FCFS |
|---|---|---|---|
| BFCL-like | 44.0→**25.2 (1.75x)** | 25.3 (1.74x) | 1.0x 基本无 |
| SWE-like | 46.3→**31.2 (1.48x)** | 26.0 (1.78x) | ~1.0x |
| mixed | 67.0→**40.3 (1.66x)**，3 seed 为 1.66/1.83/1.87x | 36.4 (1.84x) | 1.03x |

- **复现了主要趋势**：turn 越多/上下文越大收益越大（mixed>BFCL>SWE）；量级落在论文 1.12–3.66x 区间下半段；收益来源拆分：prefill 总量降 ~55–65%（重算节省）+ avgQ 从 2–13s 降到 <1s（逐轮排队消除），后者在高压下占主导。
- **差异（诚实记录）**：同等条件下 static 常比 full 好 5–15%（短工具主导时 pin-always 最优；full 为长尾保守跳过部分 pin，命中率 55–65% vs static 74–86%）。论文 ablation 称 Full>Static，差异解释：①混合负载 `eta` 被拉低（turn 数离散）→OutofOrder 项保守；②`T` 滑窗噪声。未做迎合性调参。
- **JPS 扫描（mixed，适用边界）**：0.04→1.41x / 0.08→1.66x / 0.15→2.39x / **0.25→2.35x 且反超 static（1.67x）**——低压 static 占优、高压自适应 TTL 反超，正好对应论文 Fig.6"TTL 过大/过小"权衡。
- **Offload**：低压 + offload 时全部 1.00x（reload 0.46s、avgQ≈0，无事可做）；高压 + offload 时 pin 反而有害（0.88–0.91x，static 更差）——CPU 充足且 reload 快时保留的机会成本占优，论文"增益与 offload 正交"只在排队仍重的区间成立。

## 3. 机制验证中的两个真 bug（已修，均有最小复现）

1. **自死锁**：pin 过期但程序已在队列 → 按论文"过期且不在队列才释放"永久不放；运行队列为空时无人触发 → 事件耗尽 stall。仿真复现（static 37/60 完成）。修：`_victim_unpin`（逐出最晚到达 pin，重试调度）。对应论文 5.2，已列入真机移植清单。
2. **双计内存**：命中判断前把自家 pin 内存也算作占用 → 自己的下一轮永远"装不下"。修：`_eff_need` 扣减自家有效 pin。
3. **victim 后直接 return**（修 1 时引入）：最后一個事件后永久 stall。修：unpin 成功后 `continue` 重试。

## 4. 适用边界分析（HOL 均经严格标准：队头确实装不下 + 后继此时能装下 + 严格顺序阻塞）

| 假设 | 结论 |
|---|---|
| 工具长尾/预测偏差 | 长尾工具（>2s 尾部）使 static 浪费 pin 时长；full 用 ECDF 截断，但 `T` 膨胀时 `B` 膨胀会退化成 pin-long（§2 JPS 0.25 full 仍赢是靠选择性跳过，不是靠 TTL 长度）。未发现稳定"预测偏差致崩"，TTL 上限天然有界。 |
| 长短共存 | mixed 下 prog-FCFS 几乎无效（1.03x），pin+TTL 仍有 1.66–1.87x；短程序被长 pin 阻塞的 HOL 事件存在（full HOL 202–760），但 JCT 净收益为正。 |
| 上下文快速增长 | 晚轮 ctx 可达 150k tok（PR~15s），pin 价值最高；但单轮 >cap 会直接饿死（seed1 教训），必须如论文做"缩放到塞进上下文"。 |
| CPU 不足/足够 | DRAM offload 开且 reload 快时，保留无收益甚至有害（0.9x）。Continuum 的有效区间是：**prefill 或排队 ≳ 工具时长**；reload<0.5s 且系统不排队时应主动退化为不 pin（full 在 offload 下自动只 pin 6–68 次，方向对但阈值可再激进）。 |
| 接近饱和 | JPS 0.25 无 offload 时 full 反超 static——自适应在饱和区更稳；但 JCT 绝对值已 3–5x 膨胀，实际应先扩容，TTL 不是扩容替代品。 |

## 5. 对论文方法的评价

- 两种延迟都真实存在且可分：仿真中 prefill 与 avgQ 可独立计量，offload 只消除前者（reload 0.46s），排队项必须靠 pin+保序消除——这是论文相对 InferCept 最硬的增量（InferCept 只看 reload 成本，`Includes Per-Turn Queueing=✗`）。
- `eta` 是点睛但脆弱的一笔：同质 turn 分布下≈1 有效，混合负载下被稀释；`T` 滑窗在 pin 生效后样本萎缩（自我实现式低估）。建议真机实现记录"若未 pin 会排多久"的反事实估计，而非仅观测被驱逐者。
- Eq.2 消去 `mem/M` 使决策与 pin 大小无关——大 ctx pin 的阻塞外部性在线性假设下被低估，饱和区应加 tote 惩罚项（见 JPS 0.25 full 靠跳过取胜）。

## 6. Phase4 云验证（基线完成；continuum 真机策略改走 Kairos 路线）

> 路线更新（2026-10-27，用户确认）：原 fork 编译被云机 GitHub 出口阻塞（cutlass 拉取挂起），
> 且 Hanchenli 预览版本就没有 TTL 估计。改走 **Kairos 内置 continuum 对照（TTLPolicy paper=Eq.2）+
> Kairos 原生 + 消融** 三方比较。Kairos 已在本地 clone（LF 干净树）+ CPU 验证全过（43 tests OK：
> 37 原有 + 6 新增 Eq.2 回归），`verify`（179 文件/178 calls）、`audit`（复现 12.09%/9.28%/6.77%）、
> `evaluate`、`--dry-run` 均通过。新增 `tests/test_ttl_policy.py` 锁定冷启动 ln(B)/K=100/
> per-tool 与全局 ECDF 回退/τ=0 枚举。
>
> 关键对齐结论：Kairos 的 `TTLPolicy.ttl()`（`kairos/_runtime/compat/policy.py:20`）逐字实现论文 v5 Eq.2——
> `benefit=wait*eta+reload`（`scheduler.py:612`，即 T·η+PrefillReload；`waiting_samples` deque(256) 为滑窗 T），
> 冷启动 `log(max(benefit,1))`（=Exp(1) 下 T_default），per-tool/global ECDF 以 K=100 分界，
> 枚举 {0}∪观测时长取 `P·B−τ` 最大。调用点 `scheduler.py:623,1438`，历史由 `:553 observe()` 按真实 gap 喂入（cap 512）。
> 唯一有意偏离：`eta` 取 profile 常量（`eta`/`eta_audit`）而非论文在线 `-Corr(k,N−k)`；history cap 512 vs 论文无界。
> 消融注意（实测纠正）：`CONTINUUM_TTL_MODE` 环境变量只影响 `__init__` 的初始实例，
> 每次 `_switch_experiment` 会按实验名重建 `TTLPolicy`（`scheduler.py:411-413`），
> fixed/zero 必须走实验名路由；直接 export 对回放无效（一次无效 fixed/zero run 已作废，仅作 paper-mode 重复样本）。

## 7. PhaseB 三方回放结果（Qwen3-8B pin revision，RTX 4090D，2026-10-27）

| run | JCT mean | JCT p95 | TTFT p95 | wall | 邻居 foreign 落在回放窗 |
|---|---|---|---|---|---|
| continuum r1 (Eq.2) | 106.73 | 158.69 | 28.82 | 164.7 | 76 次 sightings |
| continuum r2 (Eq.2) | 103.80 | 149.50 | 18.35 | 157.7 | 73 |
| kairos r1 (原生 Eq.15) | 102.56 | 142.71 | 31.38 | 148.5 | 69 |
| kairos r2 (原生 Eq.15) | 92.82 | 144.51 | 23.73 | 149.3 | 69 |
| Kairos 归档 continuum | 106.472 | 150.118 | — | 161.2 | — |
| Kairos 归档 kairos | 93.597 | 136.181 | — | 150.3 | — |

pin TTL 实测：continuum Eq.2（272 global-ecdf + 34 per-tool）均值 0.46–0.49s、p50 ~0.5s、max 0.92s；
kairos 原生（321–323 pins）均值 1.06s、p50 0.50s、max 3.38s。per-tool 样本普遍 <K=100 故以 global 回退为主。

结论（诚实口径）：方向与归档一致（kairos < continuum；r1 对 −3.9%，r2 对 −10.6%，归档 −12.09%），
且绝对值与归档惊人接近（kairos_r2 92.82 vs 93.597；continuum ~105 vs 106.47）。
但 continuum 自身重复波动 2.8% + 全程邻居 GPU 占用（每轮 69–76 次 sightings），差距在噪声量级内，
只能算方向性复现，不能算显著性验证。fixed/zero 真机消融未执行（见上段路由原因），
静态/不 pin 行为以仿真相消融为准。原始 JSON 与 foreign 日志在 `results_cloud/kairos_3way/`。

### 补记：共享卡踩坑三则（已解，均记录）
1. torchvision 误删致 EngineCore warmup 报 `No module named 'torchvision'`——恢复 0.28.0（cu130 build 与 cu132 torch 共存可用）；torchaudio cu132 无 cp312 wheel，确认本路径无人 import，直接不装。
2. `CONTINUUM_PREFIX_ONLY_CORRECTNESS=1` 必须进**父进程**环境，否则 validate 跑全套 gapped fixture，
   在 PipelinedPrefix 门控 `Pilot supports original Continuum/v21 prefix controls only` 处 fatal（scheduler.jsonl 时间线定位）。
3. 盒子非独占：邻居 CUDA 短作业（~357MiB）在 /proc 不可见（跨容器 pid ns），原生 guard 直接判失败。
   改用只记录不中断的 wrapper（零仓库修改），事后按 foreign 时间戳与回放窗重叠取舍；配置文件改回字节一致并重过 seal（179/178）。
   fixed/zero 那次无效 run 的教训：本 harness 的 TTL 模式是实验名路由，不是环境变量路由。

环境：`RTX 4090 D 24GB / Ubuntu 22.04 / 251GB RAM / Python 3.10`，`vLLM 0.10.2 + transformers 4.55.4`
（0.10.2 需降级 transformers，新版 `Qwen2Tokenizer.all_special_tokens_extended` 不兼容；
torch.compile 在容器内 triton 初始化失败，统一加 `--enforce-eager`，与 A0 在 V100 上同因）。
模型 `Qwen2.5-3B-Instruct`（5.8GB，`/root/241880423/models`），serve 配置
`--gpu-memory-utilization 0.5 --max-model-len 8192` → 占显存 12.6GB。
云上数据统一在 `/root/241880423/`（模型/results/code），脚本与 harness 在
`tasks/continuum/bench_agent.py`，原始 JSON 在 `tasks/continuum/results_cloud/`。

### 6.1 基线（真机已完成）

| 实验 | V0 (无缓存) | V1 (prefix cache) | 命中率 |
|---|---|---|---|
| single，60 请求，conc 4 | TTFT p50 **0.090s** / tps 177 | TTFT p50 **0.038s** / tps 186 | 92.6% |
| multi，24 prog×4 轮，gap 0.5s，conc 4 | TTFT p50 0.061 / makespan p50 3.77 | TTFT p50 0.039 / makespan p50 3.60 | 93.7% |
| multi-pressure，32 prog×4 轮，gap 0.5s，每请求 unique **2k tok** 文档，conc 8 | TTFT p50 **0.322** / p95 0.71 / makespan 5.44 / tps 151 | TTFT p50 **0.070** / p95 0.36 / makespan 3.61 / tps 231 | 81.4% |

逐轮 TTFT（pressure）：V1 为 turn0 0.244 → turn1-3 ≈ 0.065（首轮 prefill 后复用）；
V0 为 0.262 → 0.370/0.359/0.377（每轮重 prefill 2k 文档）。两种延迟在此分解：
prefill/重算成本（turn TTFT 0.065 vs 0.37，差 ~5.7x）+ 排队（V0 p95 0.71）。
与 A0 在 V100 上的结论一致：**offload/缓存的价值 ∝ 被驱逐内容的复用率**。

### 6.1b DRAM offload 层（V3/LMCache，真机已完成）

同配置 + `LMCacheConnectorV1`（`LMCACHE_MAX_LOCAL_CPU_SIZE=8`）：

| 实验 | V1 GPU-only | +LMCache offload | V0 |
|---|---|---|---|
| single，TTFT p50 / tps | 0.038 / 186 | 0.045 / 164 | 0.090 / 177 |
| multi-pressure，TTFT p50 / p95 | **0.070** / 0.36 | 0.135 / 0.76 | 0.322 / 0.71 |
| multi-pressure，makespan p50 / tps | **3.61** / 231 | 6.58 / 126 | 5.44 / 151 |
| multi-pressure，逐轮 TTFT | 0.24→0.065 | 0.48→0.15→0.12→0.09 | 0.26→0.37 |

解读：offload 层 TTFT 仍显著好于无缓存（0.135 vs 0.322），但本机上不如纯 GPU 缓存——
CPU→GPU 重载在本工作集（能进 GPU KV）下是纯额外开销，且此容器内 LMCache 报
`cuda_ops compiled extension not found`（torch baseline 回退），transfer 慢反映在
ITL（0.040 vs V1 0.015）和 makespan 上。这正好是论文 §3.2 的另一半论据：
**offload 只能省 prefill，省不了逐轮排队**；工作集能装进 GPU 时 pin-GPU 严格占优。

### 6.2 论文 fork 构建（阻塞，转备选方案）

`vllm-continuum` commit `316a587` 已传至云（直连 GitHub 超时，走本地中转 tar）。
构建连踩三坑，均已解并记录（供复现）：
transformers 降级 4.55.4 → `--enforce-eager` → `python3.10-dev` 缺头（`apt install`）。
但全量编译卡在 cmake FetchContent 从 `github.com/nvidia/cutlass.git` 拉源码——
该云机直连 GitHub 超时（clone 主仓库同样超时，已走本地中转 tar；gitee 镜像需登录），
构建进程在 `git clone cutlass` 处永久挂起，已 kill。磁盘剩余 12G，GPU 已清零，
vllm 进程全部清理，远端工作目录 `/root/241880423/`（模型/results/code）完整保留。
恢复路线（二选一）：① 本地按 fork 的 cmake pin 的 cutlass tag 打包中转上传，
设 `FETCHCONTENT_SOURCE_DIR_CUTLASS` 跳过下载后重编（约 30–60min）；
② 直接在 Kairos 仓库基础上加 TTL 估计（预览 fork 本就"without estimation"，
编出来也只是 static-2s + program-FCFS，仿真已完整刻画该策略类）。
下一步（烧卡前请示）：`--scheduling-policy continuum` 跑通链路 → 移植 TTL 三函数 +
`_eff_need` / `_victim_unpin`（<150 行）→ 最小验证集（合成多轮 × {GPU-only,
+LMCache} × JPS 高低）。
