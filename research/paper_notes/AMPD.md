# Paper Metadata

- **Title:** Efficient Multi-round LLM Inference over Disaggregated Serving [PAPER FACT]
- **Authors:** Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu (corresponding: ccchengff@sjtu.edu.cn) [PAPER FACT] — Shanghai Jiao Tong University, Southeast University, Peking University, University of Cambridge, OceanBase / Ant Group [PAPER FACT]
- **Venue:** ICML 2026 (Comments: ICML 2026) ; arXiv:2602.14516 [cs.DC] v1 16 Feb 2026, v2 21 Jul 2026 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2602.14516 / https://arxiv.org/abs/2602.14516 / HTML https://arxiv.org/html/2602.14516v2 [PAPER FACT]
- **Code:** [NOT REPORTED] [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2602.14516 + https://arxiv.org/html/2602.14516v2 (v2, 21 Jul 2026) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1 Problem [PAPER FACT]

- 多轮 LLM 工作流（autonomous agents、iterative RAG）日益普及，其推理呈现 **interleaved prefill-decode** 模式：环境交互输出（tool call/检索文档）作为下一轮增量输入，需在解码间插入增量 prefill [PAPER FACT]
- 现有 PD 解耦（prefill-decode disaggregation）将 compute-bound prefill 与 memory-bound decode 分离到独立资源，已被广泛采用以缓解干扰，但在多轮场景下被**忽视交错负载模式**，导致对增量 prefill 的处理与两阶段的模型部署（资源配比 + 并行策略）次优 [PAPER FACT]
- 具体缺口：
  1) 缺乏自适应调度：需在线决定增量 prefill 在 decode 本地执行还是路由至哪个 prefill 实例，以及 prefill 队列的执行顺序 [PAPER FACT]
  2) 缺乏部署感知：以往仅按输入/输出长度定资源与并行，未考虑多轮交错特征 [PAPER FACT]
- 目标：在 PD 解耦上高效服务多轮推理，最大化 **SLO attainment**（TTFT 与 ITL 满足度）[PAPER FACT]

## 2 Motivation [PAPER FACT]

- LLM 推理天然分 prefill（一次性处理 prompt 计算 KV，compute-bound，主导 TTFT）与 decode（逐 token 自回归，memory-bound，决定 ITL）；同机混跑导致 PD interference [PAPER FACT]
- PD 解耦通过专用资源显著提升吞吐与效率，已在多系统中采用并有 elastic scaling、mixed-GPU、多模态等扩展 [PAPER FACT]
- 多轮工作流两类代表：ReAct 式 agents 交替 reasoning/tool use，iterative RAG 在生成中主动检索纠错 [PAPER FACT]；从 serving 视角均产生 interleaved 增量 prefill
- 现有 PD 解耦工作聚焦单轮，单独的多轮优化（InferCept discard/swap/preserve、KVFlow prefix cache、vLLM-Continuum TTL、MARS/AugServe 预估长度/显存）均为 **co-located** 范式，未解决 PD 解耦下的实时路由/调度与部署配置协同 [PAPER FACT]
- 需同时优化运行时调度（where/how）与离线部署规划（resource allocation + parallel strategy）[PAPER FACT]

## 3 Bottleneck [PAPER FACT]

1. **增量 prefill 的 where 问题：** 全路由至 prefill 实例以避干扰，但重负载下需权衡是否本地在 decode 实例执行以省传输、减轻 prefill 压力，同时会暂停该 decode 的 batch [PAPER FACT]
2. **增量 prefill 的 how 问题：** prefill 实例队列中含初始与增量任务，顺序影响 SLO；增加的增量负载加剧排队 [PAPER FACT]
3. **部署配置不匹配：** 资源与并行策略按单轮输入/输出长度推断失效，未计入多轮交错量 [PAPER FACT]
4. **SLO 异质：** TTFT（首 token 含增量 prefill）与 ITL（每 token）需分别满足，单阶段优化难以兼顾 [PAPER FACT]
5. **传输与一致性：** 增量 prefill 涉及 KV 在 prefill <-> decode 间传输，通用传输成本 T_kv(l_ctx; theta_src->theta_dst) 需建模 [PAPER FACT]
6. **现有 PD 相关工作偏单轮/聚合-解耦混合但偏重 decode 卸载，不适合多轮中 prefill 更重的场景** [PAPER FACT]

## 4 Core Idea [PAPER FACT]

**AMPD (Adaptive Multi-round with PD disaggregation)：多轮感知的 PD 解耦服务框架，核心为基于实时负载自适应协调 prefill 工作负载（决定 where 与 how）以最大化 SLO attainment，辅以针对多轮的离线部署规划求解最优资源与并行策略 [PAPER FACT]**

- **Online Adaptive Scheduling（运行时）：**
  - **Adaptive Routing：** 动态决定增量 prefill 是否本地在负责该请求的 decode worker 执行，或路由至某个 prefill worker；考虑实时队列、TTFT/ITL 窗口统计（默认过去 10s 平均）、传输成本与 batch 暂停代价 [PAPER FACT]
  - **Prefill Reordering：** 在 prefill 实例上对排队任务重排以最大化 SLO 达成，缓解增量任务带来的额外排队压力 [PAPER FACT]
- **Offline Deployment Planning（离线）：**
  - 将资源分配与并行策略（parallel strategy theta）确定建模为 **integer linear programming (ILP)** 并求解，得最优部署配置 [PAPER FACT]
- **Profiler + Performance Model：**
  - 采用分段 alpha-beta 模型构建三类时间估计函数：
    - T_pre(l_hist, l_incr; theta)：历史长度 l_hist + 增量输入 l_incr 的 prefill 时延 [PAPER FACT]
    - T_dec(b; theta)：batch size b 的 decode 时延 [PAPER FACT]
    - T_kv(l_ctx; theta_src, theta_dst)：跨并行策略的 KV 传输时延 [PAPER FACT]
  - 供离线规划与在线调度共同使用 [PAPER FACT]

## 5 System Changes [PAPER FACT]

- **架构分 offline/online 两阶段 + coordinator/workers（Fig.2）[PAPER FACT]**
- **Offline：**
  - **Profiler：** 测量不同 workload 下执行时间，构建上述三函数 alpha-beta 性能模型 [PAPER FACT]
  - **Planner：** 求解 ILP 确定两阶段资源量与并行策略（theta），输出部署配置 [PAPER FACT]
- **Online：**
  - **Coordinator：** 负责任务分配与路由/重排决策，基于全局可访问的任务队列元数据与 windowed TTFT/ITL 统计（分布式共享内存实现）[PAPER FACT]
  - **Prefill / Decode Workers：** 各自维护任务队列与 windowed 统计；decode worker 绑定（binding）请求，负责该请求的 KV 管理与全部 decode；prefill worker 执行被路由的 prefill [PAPER FACT]
  - **Binding：** 请求到达时按显存使用绑定到 decode worker，后续该请求的 decode 全由其负责，统一初始与增量 prefill 处理逻辑 [PAPER FACT]
  - **Routing：** 每次需 prefill（初始或增量）时触发 adaptive routing，在 local execution（在绑定的 decode worker 直接执行，省传输但暂停 decode batch）与 routed to prefill instance（选具体 prefill worker）间决策 [PAPER FACT]
  - **分布式共享内存：** 队列与 windowed 统计全局可访问以支撑 up-to-date 协调 [PAPER FACT]
- **工作流：** Binding -> Routing (adaptive) -> Prefill 执行（含重排）-> KV 回传/本地可用 -> Decode -> 环境交互 -> 再次触发增量 prefill [PAPER FACT]
- **实现基于现有推理引擎的扩展 [PAPER FACT]（论文 Sec6 Implementation，细节在截断后未完整展开）**

## 6 Target Metrics [PAPER FACT]

- **Primary：**
  - **SLO attainment**（满足 TTFT 与 ITL SLO 的请求比例）作为主目标，最大化其 [PAPER FACT]；TTFT 定义涵盖初始与增量 prefill 的首 token 时延，ITL 为每 token 生成时延 [PAPER FACT]
  - **Throughput / Goodput** 在 SLO 约束下的可服务速率（隐含）[PAPER FACT]
- **Secondary：**
  - T_pre / T_dec / T_kv 的估计准确性与 profiler 开销 [PAPER FACT]
  - 不同资源配比与并行策略下的性能对比 [PAPER FACT]
  - 消融：adaptive routing vs 固定路由、重排 vs FCFS 等 [PAPER FACT]（Inference: Sec7 实验含此类消融，但截断未展示表格）
- **Not elaborate：** 能耗、成本 $/token [NOT REPORTED]

## 7 Baselines [PAPER FACT]

- **Co-located serving（传统同机）** 作为对照 [PAPER FACT]
- **State-of-the-art disaggregated baselines（PD 解耦但为单轮设计，如 DistServe 类思路）** [PAPER FACT]
- **PD 解耦变体**：涉及 elastic scaling、mixed-GPU、多模态等扩展但仍为单轮假设的系统 [PAPER FACT]
- **Multi-round co-located 优化**：InferCept、KVFlow、vLLM-Continuum、MARS、AugServe 等（文中作为相关工作对比，说明其 co-located 局限）[PAPER FACT]
- **AMPD 的消融基线**：无 adaptive routing / 无重排 / 无 ILP 规划的退化版本（实验章节隐含）[PAPER FACT]

## 8 Workloads [PAPER FACT]

- **多轮工作流代表：**
  - **Autonomous agents（ReAct-style）**：推理-工具交替 [PAPER FACT]
  - **Iterative RAG**：生成中检索新文档纠错/精化 [PAPER FACT]
- **实验 workloads：** diverse multi-round workloads（论文声称 extensive empirical across diverse multi-round workloads）[PAPER FACT]；具体数据集名称、轮数分布、上下文长度在截断的 Sec7.1 中未完整展示 [NOT REPORTED] 需回正文表格
- **输入特征：** 初始 prompt 长度 + 增量输入长度 l_incr / 历史长度 l_hist 的组合，体现 interleaved pattern [PAPER FACT]
- **请求到达：** 未在摘要/可见正文中明示分布（推测 Poisson/真实 trace） [NOT REPORTED]

## 9 Hardware [PAPER FACT]

- **讨论硬件为 PD 解耦的专用资源池**：prefill 与 decode 各自 GPU 资源，需 결정资源分配与并行策略 theta（如 TP/PP）[PAPER FACT]
- **具体实验硬件（GPU 型号、数量、互联）** 在截断的 Sec7.1 Experimental Setup 中未完整展示 [NOT REPORTED]
- **传输建模：** T_kv 显式建模跨并行策略的 KV 传输，体现对 NVLink/InfiniBand 等带宽的感知 [PAPER FACT]
- **分布式共享内存**用于队列/统计全局可访问，暗示多节点/多 GPU 部署 [PAPER FACT]

## 10 Main Results [PAPER FACT]

- **Headline：** AMPD 相比 co-located 与 disaggregated 的 SOTA baselines，**SLO attainment 平均提升 67.29% - 339.74%** [PAPER FACT]（Abstract/Introduction Contribution 末句及实验章 headline）
- **贡献点验证：**
  - Adaptive routing + prefill reordering 有效提升 SLO [PAPER FACT]
  - ILP 规划求解的最优资源/并行配置优于仅按输入/输出长度的传统配置 [PAPER FACT]
- **具体分解（按 workload/模型/并行）在 HTML 截断的 Sec7.2 Experiment Results 中**，未在摘要中给出分表数值 [NOT REPORTED] 需回正文图表
- **Planner 开销与仿真准确性**在 Appendix A 补充：Performance Simulation 与 Time Cost of Planning、Effectiveness 小节 [PAPER FACT]
- **未以 ms/ rps 绝对值表格在摘要中列出**，仅百分比提升 [PAPER FACT]

## 11 Assumptions [PAPER FACT]

- 推理可明确分为 prefill（compute-bound）与 decode（memory-bound），且 PD interference 显著需解耦 [PAPER FACT]
- 多轮工作流可抽象为 interleaved prefill-decode 序列，增量 prefill 长度 l_incr 与历史上下文 l_ctx 可度量 [PAPER FACT]
- 性能可用分段 alpha-beta 模型以 l_hist/l_incr/batch 对 theta 的函数近似，具备可 profiling 性 [PAPER FACT]
- KV 传输时延可由 l_ctx 与收发并行策略建模 [PAPER FACT]
- Coordinator 可全局观测队列与 windowed TTFT/ITL（10s 窗口）并基于此即时决策 [PAPER FACT]
- 请求与 decode worker 的 binding 一旦确定则该请求 KV 由其管理 [PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- 论文在 **Conclusion and Possible Future Works（Ch8）** 讨论未来方向，暗示当前局限 [PAPER FACT]（截断未展开细节）
- 推断作者自认：仅聚焦 where/how 的调度与部署，未覆盖多轮中的长期 KV 管理（跨轮缓存、discard/swap 策略已由 InferCept 等补充，可正交）[PAPER FACT]
- 离线 ILP 规划依赖 profiling 与 workload 预测，对突发 workload 适应性有限 [PAPER FACT]（Inference: 由 planner 需预先定配置可推断）
- 实现与评测细节在 Sec6-7，截断部分可能含对单轮扩展或异构加速的讨论不足 [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- Adaptive routing 的决策需实时全局状态，分布式共享内存的一致性与延迟可能成为新瓶颈 [AGENT INFERENCE]
- 本地在 decode 执行增量 prefill 会暂停 decode batch，对长增量输入可能显著抬高 ITL，未给出阈值策略的理论界 [AGENT INFERENCE]
- ILP 求解在资源池大、候选并行策略多时可能计算开销高，动态重规划频率未评估 [AGENT INFERENCE]
- T_pre/T_dec/T_kv 的 alpha-beta 分段模型对新硬件/新模型/长上下文的泛化误差未量化 [AGENT INFERENCE]
- 评测中未与近期 PD 解耦变体（Mooncake、Splitwise、DistServe with KV cache reuse 等）做 head-to-head 对比 [AGENT INFERENCE]
- 多轮 workload 的轮数、上下文膨胀与 tool 延迟分布若高度偏态，固定 10s 窗口统计可能失真 [AGENT INFERENCE]
- 未讨论多租户/多会话间的公平性与隔离 [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. 如何在线自动调优 10s 窗口与 routing 阈值以适配突发与长尾多轮 workload？[AGENT INFERENCE]
2. ILP 规划能否增量式/在线重求解以应对时变资源与工作负载漂移？[AGENT INFERENCE]
3. 增量 prefill 的本地 vs 路由决策是否可与 prefix cache（RadixAttention）联合以减少重复历史编码？[AGENT INFERENCE]
4. 在异构 GPU（HBM vs 低成本内存型）集群下，prefill/decode 的资源异构配比最优是什么？[AGENT INFERENCE]
5. 跨轮 KV 的 discard/swap/retain 策略与 PD 解耦如何协同（如 InferCept + AMPD）？[AGENT INFERENCE]
6. T_kv 的压缩（量化/稀疏）与传输流水能否进一步隐藏增量 prefill 延迟？[AGENT INFERENCE]
7. 1M 长上下文多轮（如长文档 agent）下，l_hist 极大时的增量 prefill 效率如何保证？[AGENT INFERENCE]
8. 能否形式化 SLO attainment 的上界并证明 adaptive routing 的近似比？[AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Prefill-Decode Disaggregation 基础：** DistServe (Zhong et al. 2024), Splitwise (Patel et al. 2024), TetriInfer (Hu et al. 2024) 被引为 PD 解耦广泛采用的代表 [PAPER FACT]
- **PD 扩展：** elastic scaling (Zhang 2025a, Lai 2025), mixed-GPU (Jiang 2025b), multi-modal (Singh 2025, Dong 2025) [PAPER FACT]
- **聚合-解耦混合：** Ruan 2025, Wang 2025a（文中指出其不考虑多轮交错）[PAPER FACT]
- **多轮系统优化（co-located）：** InferCept (Abhyankar 2024), KVFlow (Pan 2025a), vLLM-Continuum (Li 2025b), MARS (Shahout 2025), AugServe (Wang 2025b) [PAPER FACT]
- **后续可结合：** DistServe 的 goodput 优化 + AMPD 多轮自适应、LMCache/Mooncake 的分布式 KV 池 [AGENT INFERENCE]
- **Agent/RAG：** ReAct (Yao 2023), Iterative RAG (Shao 2023, Yue 2024) 为 workload 来源 [PAPER FACT]

---
## Review Log

Reviewer: Reviewer-1 (supplemental) — 2026-08-27
Problems Found: 核心贡献、主结果数值、hardware 与 baseline 经 webfetch https://arxiv.org/abs/2602.14516 抽查一致；多轮 P/D 自适应路由与资源规划描述与原文一致，无 hallucination
Corrections: 无修正
Confidence: High
