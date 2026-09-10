# Paper Metadata

- **Title:** Fast Distributed Inference Serving for Large Language Models [PAPER FACT]
- **Authors:** Bingyang Wu*, Yinmin Zhong*, Zili Zhang*, Shengyu Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xuanzhe Liu, Xin Jin (* equal contribution) [PAPER FACT] — Affiliation: Peking University [PAPER FACT]
- **Venue:** arXiv:2305.05920 [cs.LG, cs.DC] v1 10 May 2023 (319 KB), v2 21 Sep 2024 (896 KB), v3 25 Sep 2024 (896 KB) [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2305.05920 doi:10.48550/arXiv.2305.05920 [PAPER FACT]; HTML https://arxiv.org/html/2305.05920v3 , PDF https://arxiv.org/pdf/2305.05920 [PAPER FACT]
- **Code:** [NOT REPORTED] — paper states prototype with 2.9K Python + 8.1K C++/CUDA, implements OPT in C++ and custom kernels for Orca iteration-level scheduling and vLLM PagedAttention, but no public URL given [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2305.05920 + https://arxiv.org/html/2305.05920v3 (v3, 25 Sep 2024) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1. Problem [PAPER FACT]

- 交互式 LLM 应用（ChatGPT 等）要求低延迟推理，但现有服务系统对推理作业采用 run-to-completion (FCFS) 处理，遭遇 head-of-line blocking，端到端延迟高 [PAPER FACT]
- LLM 推理具有 autoregressive 模式：每轮 iteration 生成 1 token 并拼接到输入以生成下一 token，执行时间取决于 input length + output length，后者事先未知 [PAPER FACT]
- 现有 LLM 服务 (Orca iteration-level scheduling, vLLM PagedAttention) 仍用 FCFS，一旦调度即跑完；受 GPU 显存与严格 SLO 限制，处理中的 batch 无法随意加入新作业，长作业阻塞短作业 [PAPER FACT]
- 图1 量化：理想下若作业大小一致则无排队延迟 (load≈1)，但 ShareGPT/Alpaca 等真实数据集输出长度长尾分布，排队延迟占总延迟高达 90% [PAPER FACT]
- 核心问题：如何在不预知 output length 的情况下实现 token 粒度抢占式调度以最小化延迟，并解决抢占带来的 KV cache 显存爆炸问题，同时支持分布式服务 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- LLM 驱动的新一代交互式 AI 爆发，对响应即时性要求高；企业为推理配置庞大昂贵的 GPU/TPU 集群 [PAPER FACT]
- DNN 推理 (ResNet) 执行时间确定可预测，可用 Clockwork/Shepherd 式精确 profiling 调度；LLM 推理因 output 长度可变而不可预测，传统方法失效 [PAPER FACT]
- Orca 提出 iteration-level 调度可每轮迭代后动态增删作业，vLLM 引入 PagedAttention 减少 KV 碎片，但二者均为 FCFS，仍受 HOL 阻塞困扰 [PAPER FACT]
- 排队延迟已是主要矛盾：优化单次执行时间仅占小部分，必须优化排队延迟才能显著降低端到端延迟 [PAPER FACT]
- GPU 显存稀缺：除权重外，KV cache 巨大；抢占式调度需为所有已启动但被抢占的作业保留 KV，显存压力剧增，需高效显存管理 [PAPER FACT]
- 大模型 (OPT-175B 需 350GB 权重) 必须分布式，需同时支持 tensor 与 pipeline 并行并保持调度语义 [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **FCFS 导致的 HOL 阻塞：** 长作业长时间占用 batch，短作业排队；真实长尾负载下问题尤为严重， очереди 延迟占 90% [PAPER FACT]
2. **Job size 可变不可知：** SRPT 虽最优但需预知剩余处理时间，而 LLM 的 output 长度依赖语义不可预知，无法直接应用 [PAPER FACT]
3. **MLFQ 直接套用失配：** 经典 MLFQ 假设无先验信息，新作业进最高优先级队列，但 LLM 首轮 iteration (initialization/prefill) 耗时远大于后续 decode（需计算全部 input tokens 的 KV），且随 input length 增大；若 quantum 不足首轮即耗尽，抢占将丢弃中间激活重算浪费，或不抢占则违背 MLFQ 目的 [PAPER FACT]
4. **抢占式显存爆炸：** FCFS 仅需缓存运行中 batch 的 KV，MLFQ 需缓存所有已启动未完成的作业。OPT-175B 单作业 s=512,t=1 即需 2.3 GB (按 4·l·h·(s+t) 公式 l=96,h=12288)；图8 显示 OPT-2.7B 在合成负载 (Gamma rate=64, CV=4, max output 20) 下 skip-join MLFQ 的 KV 峰值是 FCFS 的 7x [PAPER FACT]
5. **朴素显存管理两难：** (a) 延迟新作业等待显存释放则退化为 FCFS 仍 HOL；(b) kill 低优先级作业释放 KV 则需重建且可能因饥饿提升导致的死锁 [PAPER FACT]
6. **分布式复杂性：** pipeline 并行下作业在不同 stage 同时处理多 batch，传统 MLFQ“跑到 demote 才调度下一作业”的语义不适用；KV 也需分区至各 GPU [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**FastServe = Skip-Join MLFQ 调度 (利用 semi-information-agnostic) + 主动式 (Proactive) KV Cache 在 GPU-Host 间换入换出 + 分布式适配的流水与分区管理 [PAPER FACT]**

- **Semi-information-agnostic 洞察：** 虽 output 长度未知，但 input length 已知且首 token 执行时间可通过轻量 profiling 精确预测；初始化阶段时间与 input length 正相关，decode 阶段每 token 时间近似常量 [PAPER FACT]
- **Skip-Join MLFQ：** 设 n 个优先级队列 Q1..Qn，quantum q1<q2<...<qn (每级 2x 上一级，q1 设为最小 iteration 时间)。新作业不进 Q1，而是预测其 t_init 选满足 qi ≥ t_init 的最高优先级队列直接 skip-join 入队，跳过更高优先级队列以减少 demotion 次数；作业耗尽 quantum 未完成则按下一轮 iteration 时间 demote 至 η 级更低队列；并设饥饿阈值 α (默认 300 ms，SLO 驱动) 周期检查将 starveTime≥α 的作业提升至 Q1 [PAPER FACT]
- **形式：** skip: p_job = min i s.t. qi ≥ init_time；demote: Q_{p} -> Q_{p+η}；promote: starve 则 Q_{p} -> Q1；Algorithm 1 给出伪码 [PAPER FACT]
- **示例效果：** 3 作业同到时，FCFS 平均延迟 4.23、朴素 MLFQ 5、skip-join 3.3、SRPT 最优 3，skip-join 显著接近最优 [PAPER FACT]
- **主动式 KV 管理：** 将 KV 缓存空间从 GPU 扩展到 host 内存；不 reactive 等抢占后再换，而是 proactive 提前换：基于 Estimated Next Scheduled Time (ENST) 决定换出/换入顺序，最大 ENST 先换出、最小 ENST 先换入；ENST = min(T_promote(i), T_execute(i))，其中 T_execute(i) = (1/B) Σ_{j 高优先级} Σ_{i.priority<k≤j.priority} qk；通过流水与异步拷贝将换运与计算重叠 [PAPER FACT]
- **Burst 预留：** 为应对高优先新作业突发，基于历史到达模式预留若干空闲 KV 槽位，避免 reactive 驱逐 [PAPER FACT]
- **分布式：** 调度器在 pipeline 下不再“跑满 quantum 才换”，而是每 stage 完成即调度挂起队列中最高优先级作业；KV 按 tensor 并行分区到对应 GPU，各 stage 独立但复用首 stage 的换运决策并与中间结果传输并行 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **整体架构 (图4)：** 作业池 -> profiler (预测 iteration 时间) -> skip-join MLFQ 调度器 -> 分布式执行引擎 + 分布式 KV cache；引擎每轮从调度器取至多 MaxBatchSize 个作业执行一轮 iteration [PAPER FACT]
- **调度器：** 实现 skip-join、demote (η 级)、starvation promotion 三机制；调度粒度为一轮 token；每轮输出新 token、检查 finish/pop、demote、promote、再按优先级选就绪作业 [PAPER FACT]
- **KV 管理器：** 维护 GPU 与 host 两级缓存；ENST 计算需跟踪各作业晋升时间与高优先级作业累计执行时间 (按 batch 均摊)；后台异步线程执行换入换出，与推理 kernel 流水重叠；预留 idle slots 数基于历史突发频率动态 [PAPER FACT]
- **分布式执行：** 支持 tensor 并行 (算子切分，需额外通信聚和) 与 pipeline 并行 (模型按 operator 划 stage，流水传输中间结果) 的混合；调度器需同时管理 pipeline 中多 batch；KV 按同一 stage 所用张量分区，确保计算只需本地 KV；首 stage 的换运指令广播至 pipeline 后续 stage，中间结果传输与 KV 换运并行 (图10) [PAPER FACT]
- **实现：** 前端与调度器 2.9K Python，分布式引擎 8.1K C++/CUDA；前端兼容 OpenAI API (max output length, temperature 等)；引擎基于 Ray actor 实现 GPU workers；用 C++ 重实现 OPT 等模型以优于 HuggingFace Python；自研 CUDA 核支持 Orca 迭代调度与 vLLM PagedAttention [PAPER FACT]
- **集成优化：** 集成 PagedAttention 按块渐进分配 KV 以减碎片；与 profiling 配合精确预测首轮时间 [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **Average per-token latency = 均值 (job end-to-end latency / output length)**，衡量交互体验 [PAPER FACT]
  - **P95 tail latency** (95 分位) [PAPER FACT]
  - **Throughput @ SLO：** 在给定平均/尾延迟 SLO 下可达到的最大吞吐 (arrival rate) [PAPER FACT]
  - **P95 goodput：** 同时满足 95% 作业在 initialization 与 decoding 两阶段各自 SLO 内的吞吐 [PAPER FACT]
  - **SLO 设定：** 10x decode 单轮迭代延迟，实验默认 0.3 seconds；goodput 实验另测 5x/10x/20x 三档 [PAPER FACT]
- **Secondary：**
  - 各类调度下的平均/P95 延迟随 arrival rate 曲线 (图11/12) [PAPER FACT]
  - 不同 input/output 比例下的归一化延迟 [PAPER FACT]
  - Proactive vs Reactive vs Recompute 的延迟与 breakdown (queuing / execution / swapping) [PAPER FACT]
  - 量化 swapping 时间占比 (<5%) [PAPER FACT]
  - ENST 等策略有效性消融 [PAPER FACT]

## 7. Baselines [PAPER FACT]

- **FasterTransformer v5.3 [NVIDIA]：** 生产级引擎，支持 tensor+pipeline 并行，但 job-level 调度，同一 batch 内长短作业相互阻塞，无 iteration-level 调度 [PAPER FACT]
- **vLLM v0.1.7 [Kwon et al. SOSP23]：** SOTA，支持 iteration-level 调度 (Orca) + PagedAttention 减碎片，但采用 FCFS run-to-completion，无抢占 [PAPER FACT]
- **FastServe-FCFS：** 同一 FastServe 分布式引擎但关闭本文提出的 skip-join MLFQ 与主动 KV 管理，采用 FCFS；用于隔离实现效率带来的加速与调度/显存管理的增益 [PAPER FACT]
- **对比维度表 (Table 2)：** 4 者在 IP (Iteration-level Preemption)、IS (Iteration-level Scheduling)、PA (PagedAttention)、PP (Pipeline Parallelism) 上的能力矩阵；仅 FastServe 四项全具备 [PAPER FACT]
- **设计选择消融基线：** FCFS / Naive MLFQ / Fixed Priority (按 input length 定优先级) / Recompute (丢弃 KV 重算) / Reactive (阻塞式换) [PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Models (Table 1)：**
  - OPT-13B (26GB, 40层, 40 heads, hidden 5120) [PAPER FACT]
  - OPT-66B (132GB, 64层, 72 heads, 9216) [PAPER FACT]
  - OPT-175B (350GB, 96层, 96 heads, 12288) [PAPER FACT]
  - 精度 FP16 [PAPER FACT]
- **Datasets：** ShareGPT (用户与 ChatGPT 的真实共享对话) 与 Alpaca (GPT-3.5 自指令生成) [PAPER FACT]；二者均呈现 input/output 长度长尾分布 [PAPER FACT]
- **到达过程：** 原数据集无时间戳，按 Poisson 过程生成 arrival time，参数化 arrival rate 扫负载 [PAPER FACT]
- **评测场景：**
  - 端到端：ShareGPT/Alpaca × 13B/66B/175B 全组合 [PAPER FACT]
  - Design choices：单 A100 上 OPT-13B 小规模验证各组件有效性 [PAPER FACT]
  - Goodput：OPT-13B 上测 P95 goodput 随 SLO (5x/10x/20x) 变化 [PAPER FACT]
  - 额外：改变 input/output 比例以模拟 LLM 上下文窗口扩大趋势 [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **端到端测试床：** 2x AWS EC2 p4d.24xlarge，各 8x NVIDIA A100 40GB 以 NVLink 互联，1152 GB host 内存，PCIe 4.0 x16 [PAPER FACT]
- **Design choices 测试床：** 单 NVIDIA A100 40GB 自建环境，用于组件有效性验证 [PAPER FACT]
- **网络/互联：** NVLink GPU 间，PCIe 4.0 x16 用于 GPU-Host 换运 (Swap 带宽) [PAPER FACT]
- **软件：** Ray actor 管理 GPU workers，C++/CUDA 引擎，Python 调度器；实现迭代调度与 PagedAttention  [PAPER FACT]
- **未报告：** 精确 CPU 型号、OS、CUDA/驱动版本、Ray 版本 [NOT REPORTED]

## 10. Main Results [PAPER FACT]

- **端到端平均延迟 (图11 ShareGPT 第一行 / Alpaca 第二行)：**
  - **vs FasterTransformer：** ShareGPT 上 31.5–74.9x 吞吐提升 @ 相同 SLO；Alpaca 上 9.5–15.8x [PAPER FACT]
  - **vs vLLM：** ShareGPT 上 2.3–18.3x；Alpaca 上 3–31.4x (论文标题的 31.4x 即 Alpaca 上最大) [PAPER FACT]；论文摘要称平均延迟下 31.4x、尾延迟下 17.9x [PAPER FACT]
  - **vs FastServe-FCFS：** ShareGPT 上 2–4x，Alpaca 上 1.6–2x，证明调度与 KV 管理而非仅 C++ 实现带来增益；FastServe-FCFS 因高效 C++ 与 kernel 融合也优于 vLLM [PAPER FACT]
- **尾延迟 P95 (图12 ShareGPT)：**
  - FastServe 在相同 P95 SLO 下 vs vLLM 达 17.9x vs FasterTransformer 59.8x；以 OPT-175B 为例 vs FastServe-FCFS 达 1.5x，13B/66B 上 2–2.8x [PAPER FACT]
  - 虽为优化平均延迟设计，skip-join 减少 HOL 后长作业排队也降低，且饥饿防止保证长作业不过度延迟，故尾延迟亦改善 [PAPER FACT]
- **Goodput (图13 OPT-13B P95 goodput，5x/10x/20x SLO)：**
  - FastServe vs vLLM 4.1–4.7x，vs FastServe-FCFS 1.46–1.64x，保持两阶段 SLO 下仍吞吐最高 [PAPER FACT]
- **组件有效性：**
  - **Skip-join vs 其他调度 (图14 OPT-13B ShareGPT，改变 input/output 比例)：** 归一化延迟上，skip-join 比 FCFS 最多 8.9x，比 naive MLFQ 最多 1.87x，比 Fixed Priority 最多 13.9x；小比例时 naive 尚可但大比例下因首轮长而劣化，Fixed 相反，skip-join 全比例稳定最优 [PAPER FACT]
  - **主动 KV 管理 (图15 OPT-13B)：** 高负载下 proactive 比 Recompute 好 2.7x (避免丢弃重算)，比 Reactive 好 1.7x (重叠换运)；低负载时三者相近 (显存充足) [PAPER FACT]
  - **开销分解 (图15b)：** swapping 时间 <5% 端到端延迟，可忽略，证明重叠有效 [PAPER FACT]
- **关键数值补充：**
  - OPT-175B 解码单 token 约 60 ms，PCIe 4.0 x16 全带宽下换 2.3 GB KV 约 36 ms，故重叠至关重要 [PAPER FACT]
  - 单作业 KV 2.3 GB (s=512,t=1)，peak 7x 于 FCFS 的实测 [PAPER FACT]
  - 示例 3 作业调度：skip-join 3.3 接近 SRPT 3，远优于 FCFS 4.23 与 MLFQ 5 [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- LLM 推理为 autoregressive 且每轮 iteration 执行时间高度可预测，可离线轻量 profiling 精确获得不同硬件/模型/input length 下的首轮与后续轮时间 [PAPER FACT]
- Input length 已知，output length 未知且不可精准预测，长尾分布 [PAPER FACT]
- GPU 显存远小于 host 内存 (如 A100 80GB vs 1152GB)，且大量被权重占据，KV 缓存成为稀缺资源 [PAPER FACT]
- 权重与计算精度 FP16 下 KV 大小公式 4·l·h·(s+t) 成立；分块粒度与 batch 均摊模型成立 [PAPER FACT]
- Tensor 并行额外通信可接受、pipeline 可通过多 batch 流水摊销 bubbles；NVLink/PCIe 带宽可预测 [PAPER FACT]
- 到达符合 Poisson，可通过历史估计突发预留量；量子每级 2x 递增在 MLFQ 文献中最优 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 讨论于 Sec 7 相关工作与 Sec 8 结论隐含：未与最新 disaggregation 方案 (Splitwise, DistServe, LoongServe 等) 定量对比，称其正交 (消除 prefill-decode 干扰 vs 优化排队) [PAPER FACT]
- 未与量化/稀疏等显存压缩技术联合，PagedAttention 仅减碎片不减量，KV 量随上下文长度线性增长仍是根本难题 [PAPER FACT]
- 评估仅限 OPT 家族与 ShareGPT/Alpaca，虽称可扩展至其他模型但未验证 [PAPER FACT]
- alpha (饥饿阈值) 需按 SLO 调优，ENST 估算假设高优先级作业不会提前完成而被 demote 到同级，估计偏悲观 [PAPER FACT]
- 系统实现为研究原型，未提供公开代码链接，生产化仍需工程化 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- **调度信息利用有限：** 仅用 input length 预测首轮时间，未利用 prompt 语义对 output length 的可学习预测 (如预测短输出优先)，可能错失进一步逼近 SRPT 的机会 [AGENT INFERENCE]
- **ENST 启发式粗糙：** 假设高优先级作业均需完整量子才降级，未考虑提前完成或实际 service 分布，低估/高估 ENST 可能导致换运抖动与 thrashing [AGENT INFERENCE]
- **预留槽位启发式：** 基于历史突发频率设 idle slots，未给出自适应在线调参或与 autoscaling 的联动，面对突发脉冲仍可能 reactive 退化 [AGENT INFERENCE]
- **评估预算受限：** 端到端 175B 仅 16 GPU 且仅 2 节点，Design choices 仅单卡，缺乏大规模多租户与长上下文 (100k) 的显存压力测试 [AGENT INFERENCE]
- **未量化成本/能耗：** 未报告 host 内存占用、PCIe 功耗、或与 vLLM 的 Perf/$ 对比；主动换运虽重叠但仍占 PCIe 带宽 [AGENT INFERENCE]
- **与 PagedAttention 交互未深究：** 频繁换入换出可能打散 PagedAttention 的块连续性，产生额外拷贝与 TLB 压力，论文未分解 [AGENT INFERENCE]
- **Pipeline 语义妥协：** 每 stage 完成即调度挂起高优先作业虽保 MLFQ 语义，但可能增加跨 stage 的乱序与同步复杂度，未评 32+ GPU 扩展性 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 能否结合轻量 output-length 预测器 (如 BERT 分类器或历史回归) 与 skip-join，实现更接近 SRPT 且避免预测错误惩罚的调度？ [AGENT INFERENCE]
2. 在 128k-1M 长上下文时代，KV 传输 36ms/2.3GB 的开销将线性增至数百 ms，proactive 重叠是否仍足够，或需与 KV 量化/逐出/稀疏协同？ [AGENT INFERENCE]
3. ENST 能否用更精细的排队论或在线学习替代固定量子均摊，动态感知实际剩余与饥饿提升的随机性？ [AGENT INFERENCE]
4. 如何与 PD 分离 (Splitwise/DistServe) 结合：让 prefill 专用资源处理首轮重任务，decode 资源专注抢占，两者调度如何统一？ [AGENT INFERENCE]
5. burst 预留能否做成 SLO 感知的弹性池，与 Kubernetes 弹性伸缩联动，按 P95 而非均值保障？ [AGENT INFERENCE]
6. 在异构集群 (H100 + A100 + 消费级 GPU) 上，quantum 与 profiling 如何异构感知并做 placement？ [AGENT INFERENCE]
7. 能否将 PagedAttention 块管理与 host 换运统一为单一虚拟内存层，实现零拷贝与按需 page fault？ [AGENT INFERENCE]
8. 如何扩展至 MoE 与多模态 LLM，其中首轮计算不均匀且专家并行带来额外显存分片？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **Orca [Yu et al. OSDI22]：** Iteration-level 调度开创，FastServe 在其上加入抢占；FCFS 基线来源 [PAPER FACT]
- **vLLM / PagedAttention [Kwon et al. SOSP23]：** 碎片化管理与高效实现，FastServe 集成并作为 SOTA 基线 [PAPER FACT]
- **Clockwork [Gujarati OSDI20]、Shepherd [Zhang NSDI23]、Pollux 等：** 确定性 DNN 推理调度，需精确 profiling，与 LLM 可变性对比 [PAPER FACT]
- **MLFQ / Tiresias [Gu et al. NSDI19]：** GPU 集群上 MLFQ 用于训练作业，FastServe 借鉴其 2x 量子递增 [PAPER FACT]
- **Shinjuku / Shenango / Caladan：** 微秒级抢占式数据中心调度，思想同源 [PAPER FACT]
- **DistServe [Zhong et al. OSDI24]、Splitwise [Patel ISCA24]、TetriInfer [Hu 2024]：** Prefill-Decode 分离消除干扰，与 FastServe 正交，未来融合点 [PAPER FACT]
- **RAGCache / CacheBlend / LMCache：** KV 复用层，关注上下文复用而非抢占，但显存管理可借鉴 [AGENT INFERENCE]
- **Sarathi-Serve [Agrawal 2024]：** Chunked-prefill 减轻 prefill-decode 干扰，另一种缓解 HOL 的策略，与抢占对比值得 [AGENT INFERENCE]
- **LoongServe [Wu 2024]、FlexGen [Sheng 2023]：** 弹性序列并行与 offloading 吞吐优化，分布式与显存管理扩展思路 [PAPER FACT]
- **SGLang / Mooncake / DeepSeek MLA：** 前缀复用与新型注意力压缩，对 KV 容量问题的互补解法 [AGENT INFERENCE]

---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
