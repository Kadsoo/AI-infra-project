# Paper Metadata

- **Title:** FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling [PAPER FACT]
- **Authors:** Weiqing Li (*), Guochao Jiang (*), Xiangyong Ding, Zhangcheng Tao, Chuzhan Hao, Chenfeng Xu, Yuewei Zhang, Hao Wang — * equal contribution; † corresponding authors liweiqing.lwq@alibaba-inc.com, anyue.jgc@alibaba-inc.com, cashenry@126.com [PAPER FACT] — Affiliation: Alibaba Cloud Computing [PAPER FACT]
- **Venue:** arXiv:2504.03775 [cs.DC] v1 3 Apr 2025 (3,533 KB) [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2504.03775 doi:10.48550/arXiv.2504.03775 [PAPER FACT]; HTML https://arxiv.org/html/2504.03775v1 , PDF https://arxiv.org/pdf/2504.03775v1 [PAPER FACT]
- **Code:** [NOT REPORTED] — no public repository linked [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2504.03775 + https://arxiv.org/html/2504.03775v1 (v1, 03 Apr 2025) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1. Problem [PAPER FACT]

- LLM 推理由 compute-bound prefill（处理全 prompt 生成首 token 与 KV）与 memory-bound decode（自回归逐 token）两阶段组成 [PAPER FACT]
- Colocated 框架（vLLM/SGLang 同实例跑两阶段）存在干扰， PD-disaggregated 框架将 P 与 D 分离至不同节点可独立优化、水平扩展并适配异构硬件，提吞吐 [PAPER FACT]
- 但 PD 分离需在 P 与 D 节点间传输 KV cache，现有传输方案被忽视且开销显著：图1中 13k 输入+100 输出在 NCCL PagedAttention 传输占端到端延迟约 1/4 [PAPER FACT]
- 现有传输分 RDMA（需 Mellanox NIC 等特定硬件）与 NCCL（兼容 RoCE/IB/Socket 但仅支持连续地址）两类 [PAPER FACT]
- NCCL 下 PagedAttention 的分页式非连续块管理导致每块多次小 kernel 调用，传输延迟高且与 GEMM 抢占 SM，阻塞计算 [PAPER FACT]
- 另固定 P:D 节点角色导致计算不均衡（imbalance）与极端过载时资源气泡 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- LLM 推理已成为产业关键，需求快速增长 [PAPER FACT]；prefill-decode 干扰在同实例下限吞吐，解耦是趋势（Splitwise, DistServe, Mooncake 等）[PAPER FACT]
- 实测揭示 KV 传输瓶颈：单请求 13k tokens 时 NCCL 传输显著；PagedAttention 以 block 为单位管理，跨 GPU 传输需同粒度多次调用，放大全局延迟 [PAPER FACT]
- NCCL 虽性能好但仅支持连续地址，离散小张量合并导致额外拷/延时；Splitwise 按 layer 粒度多次 NCCL 导致频繁调用，vLLM-disaggregated 按 buffer 合并 layer 但产生额外内存与时间 [PAPER FACT]
- 负载不均：固定 P/D 比例在请求波峰/模型阶段偏态时易使一类节点 compute-bound 而另一类空闲；极端负载需弹性伸缩 [PAPER FACT]
- 异构 GPU（L20/H20 等）间带宽/显存差异大，异构部署若能把 decode 放大显存/带宽节点可降 TPOT/E2E [PAPER FACT]
- 机会：通过 KV 形状重塑与段式分配减少 NCCL 调用次数 + Load-Aware 调度实现灵活 PD 角色切换与均衡，可近零化传输时间并峰值吞吐 [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **碎片化调用风暴：** 标准 PagedAttention KV 形状 (L,2,B,H) 使每 KV 块需 L×2 次 NCCL 调用，block-wise 调用使 kernel 次数激增 [PAPER FACT]
2. **非连续分配：** 传统块分配器随机分散，导致收发两端块 ID 不连续，无法合并为单次传输 [PAPER FACT]
3. **额外合并开销：** 将离散层张量合并为连续 buffer 需额外拷贝与内存，降吞吐 [PAPER FACT]
4. **GPU SM 竞争：** 频繁 NCCL kernel 启动与 GEMM 等计算争夺 SM，阻塞整体流水 [PAPER FACT]
5. **负载不均与固定角色：** 现有框架常固定 P/D 角色，计算不均时一池过载一池空闲；缺乏全局感知与弹性伸缩 [PAPER FACT]
6. **异构网络：** 跨机 ENI 带宽有限，传输优化需适配单机 IPC vs 跨机 NCCL vs RDMA [PAPER FACT]
7. **长上下文放大：** 输入 10k 时传输量大，vanilla 传输在 DistServe/Mooncake 等上失败或延迟 2s+（表3）[PAPER FACT]

## 4. Core Idea [PAPER FACT]

**FlowKV = 低延迟 KV 传输优化（形状重塑 + 段式管理 + 双向段对齐合并）+ Load-Aware 调度（global controller + hybrid schedulers + 三场景策略）+ 多传输管线自适应（NCCL/IPC/RDMA）[PAPER FACT]**

- **KV 形状重塑：** 将 PagedAttention 管理的 KV 从 (L,2,B,H) 转置为 (B,L,2,H)，使同 block 的所有层连续存储，NCCL 调用数降低 L×2 倍 [PAPER FACT]；并针对 PagedAttention kernel 做定向优化 [PAPER FACT]
- **段式管理（Segment Management）：** 引入 OS 段式思想，分配时尽量在同一/少量连续段内分配，维护段内块连续；用基于段的最小堆管理空闲块，分配选合适段最小浪费，释放时合并相邻空闲段提升后续连续性 [PAPER FACT]
- **双向段对齐（Bidirectional Segment Alignment）：** 触发 NCCL 前对收发两端的 KV 块 ID 列表做双向对齐，找共连续的 N 个块 ID，合并为单次 send-recv，从而把 O(n) 次调用优化至 O(1) [PAPER FACT]；图5展示理想下从多次降至 1 次 [PAPER FACT]
- **Load-Aware Scheduler：** Global controller 实时监控各节点 load 与 KV 前缀命中，本地 hybrid scheduler 每节点含 prefill 与 decode 子调度器各有 running/waiting/swapped/pending 队列但共享 block manager [PAPER FACT]；默认 prefill 优先；Global 制定最优 P/D 节点选择与请求路由以最小化 TTFT 与传输延迟；定义三场景：
  - Normal：按 KV 前缀与负载选最优 Pt/Dt [PAPER FACT]
  - Imbalanced：指令空闲节点 hybrid scheduler 切角色若干周期，缓解气泡 [PAPER FACT]
  - Extreme：评估 load score 超阈则弹性增减特定角色节点并重构集群 [PAPER FACT]
- **多管线：** 单机自动切 IPC，跨机异构优先 NCCL，亦支持 RDMA；据硬件特征选最优 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **框架五模块：** Prefill nodes, Decode nodes, Global controller, Hybrid schedulers, KV Cache transfer module（图2）[PAPER FACT]；P/D 节点解耦可任意数量/比例/架构 [PAPER FACT]
- **KV Transfer Module：** 支持 NCCL/IPC/RDMA 管线；NCCL 下实现形状重塑 + 段式分配 + 双向对齐；单机 IPC 零拷 [PAPER FACT]
- **Global Controller：** 中心调度，监控负载模式与全局前缀匹配，生成最优请求调度与弹性伸缩方案 [PAPER FACT]
- **Hybrid Scheduler（本地）：** 每 P/D 节点内两子调度器，共享 block manager；每调度周期按 global 指令优先子调度器；默认 prefill 优先，空闲时专注 prefill [PAPER FACT]
- **算法：** 附录B 给出 Load-Aware Scheduling 伪代码（Algorithm 1），含 load scenario 判定与节点状态指示 [PAPER FACT]
- **兼容：** 基于 PagedAttention，修改其 kernel 与分配器以支持段连续 [PAPER FACT]
- **实现：** 支持同构与异构部署，异构下自动选 P-L20/D-H20 等组合 [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **KV Cache transfer latency** 平均秒级（表3）[PAPER FACT]
  - **Throughput** tokens/s 或 requests/s 在不同 RPS 下最大可承载（表1-2）[PAPER FACT]
  - **E2E latency（end-to-end response latency）** 在 LongBench summarization 上 [PAPER FACT]
  - **TPOT（time per output token）** [PAPER FACT]
- **Secondary：**
  - **加速比 Speedup** vs baseline（% 或 ×）[PAPER FACT]
  - **NCCL kernel 调用次数** 每请求（23,469 →1）[PAPER FACT]
  - **不同 input/output 长度与 RPS 下的吞吐曲线** [PAPER FACT]
  - **异构下 E2E/TPOT 提升** [PAPER FACT]
  - **传输流水对比（Mooncake/vLLM-Disagg/FlowKV-layerwise/FlowKV）** [PAPER FACT]
- **Not reported：** $ cost, 能耗 [NOT REPORTED]

## 7. Baselines [PAPER FACT]

- **vLLM (Kwon et al. 2023)：** 社区基线，PagedAttention + continuous batching；含 PD-colocated 与 PD-disaggregated (vLLM-Disagg) 两种部署 [PAPER FACT]
- **DistServe (Zhong et al. 2024)：** 代表性 PD-disaggregated 系统 [PAPER FACT]
- **Mooncake (Qin et al. 2024)：** KVCache-centric PD-disaggregated，RDMA 87GB/s + Conductor 调度 [PAPER FACT]
- **FlowKV-Layerwise：** 消融版，仅层粒度传输未做段合并 [PAPER FACT]
- **其他提及未直接 benchmark：** Splitwise, MemServe 等在 Related Work 讨论 [PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Models：** LLaMA-3 系列 Meta-Llama-3.1-8B-Instruct 与 Meta-Llama-3.1-70B-Instruct [PAPER FACT]
- **Datasets：**
  - Simulated Data：预定义 input/output 长度（1K,5K,10K 输入 /256 输出，100 条）用于最大吞吐压测；以 Poisson 到达，RPS 可控 [PAPER FACT]
  - Real-World：LongBench summarization 子任务 gov_report, multi_news, qmsum 采样 [PAPER FACT]
- **Input/Output 配置：** 吞吐表覆盖 1K/256, 5K/256, 10K/256 在 RPS 0.1–2.0 [PAPER FACT]；E2E 图覆盖长摘要实际分布 [PAPER FACT]
- **其他：** 请求采样 LongBench，输入 13k 示例在图1 [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **Homogeneous（单节点同构）：** NVIDIA A100-SXM4-80GB server 8 GPUs via NVLink [PAPER FACT]；Llama-3.1-8B 测 2 GPUs，70B 测 8 GPUs 2 nodes intra-node TP=4 [PAPER FACT]；KV 传输自动切 IPC [PAPER FACT]
- **Heterogeneous（多节点异构）：** L20 server 4× 48GB + H20 server 8× 96GB，经 ENI 连接提供基础网络带宽 [PAPER FACT]；对比 4P4D (P-L20/D-H20) vs 4P4D (P-H20/D-L20) 等配置 [PAPER FACT]
- **网络：** NVLink 600GB/s intra-node，ENI 跨机；NCCL 兼容 RoCE/IB/Socket；RDMA 需 Mellanox NIC [PAPER FACT]
- **其他平台：** 单 L20 server 也用于传输延迟对比（表3）[PAPER FACT]
- **软件：** 基于 vLLM/PagedAttention 改动；未披露精度，LLaMA-3.1 默认可推断但 [NOT REPORTED] [PAPER FACT]
- **带宽：** 测试床跨节点 25-50 Gbps 提示 [PAPER FACT]（DistServe 时 25 Gbps）

## 10. Main Results [PAPER FACT]

- **KV 传输延迟（表3，Llama-3.1-8B 1P1D，单位 s）：**
  - Single Machine：500/100 时 Mooncake 0.3010, vLLM-Disagg 0.1179, FlowKV-Layerwise 0.0678, FlowKV 0.0044 [PAPER FACT]；1000/100 时 0.5416/0.2314/0.1309/0.0075 [PAPER FACT]；4000/100 时 1.3473/0.6670/0.5338/0.0236 [PAPER FACT]；8000/100 时 2.0289/1.3382/1.1173/0.0447 [PAPER FACT]；10000/100 时 Failure/1.7373/1.4121/0.0555 [PAPER FACT]
  - Multiple Heterogeneous Machines：500/100 时 0.3418/0.1197/0.1176/0.0080 [PAPER FACT]；8000/100 时 2.1250/1.3462/1.6711/0.0993 [PAPER FACT]；平均：FlowKV NCCL 流水 vs vLLM-Disagg 降 96.8% 单机（31.5×）与 92% 跨机（12.6×），vs Mooncake RDMA 降 98.2%（55.2×）与 96.3%（55.3×）[PAPER FACT]；vs baseline layerwise 调用次数 23,469→1，单机加速 24×，多机 15× [PAPER FACT]；抽象中平均 0.944s→0.053s 降 96% [PAPER FACT]

- **同构吞吐（表1-2，图3）：**
  - Llama-3.1-8B (2 GPUs)：vs vLLM PD-colocated 平均 throughput 提 25%（1K/5K/10K 综合）[PAPER FACT]；vs DistServe/Mooncake/vLLM-Disagg (1P1D) 分别提 95%/40%/35% [PAPER FACT]；例如 5K/256 RPS1.0 时 FlowKV 264.22 vs DistServe 115.63 vs Mooncake 204.65 vs vLLM 203.04 [PAPER FACT]；10K/256 RPS2.0 时 285.14 vs Failure/185.47 vs 286.97 等 [PAPER FACT]
  - Llama-3.1-70B (8 GPUs 2 nodes TP4)：vs DistServe/Mooncake/vLLM-Disagg 提 95%/39%/35% [PAPER FACT]；例如 1K/256 RPS2.0 时 FlowKV 442.12 vs 246.62/403.62/392.50 [PAPER FACT]；10K/256 时 123.x vs Failure 等 [PAPER FACT]

- **异构 E2E（图4，LongBench，4P4D）：**
  - 4P4D (P-L20/D-H20) 显著优于 (P-H20/D-L20)：gov_report 平均 E2E 快 34.67%, multi_news 40.1%, qmsum 8.8% [PAPER FACT]；这是因为 decode 放大显存/带宽节点降 TPOT [PAPER FACT]
  - vs vLLM PD-colocated：gov_report 快 48.9%, multi_news 29.4%, qmsum 15.2% [PAPER FACT]；对应 TPOT 改善 44.57%/24.2%/15% [PAPER FACT]；vLLM 在长 prefill 时 TPOT 超限失效 [PAPER FACT]
  - Overall 加速 15.2%-48.9% 在 LongBench 上 vs baseline（摘要）[PAPER FACT]

## 11. Assumptions [PAPER FACT]

- Prefill compute-bound、Decode memory-bound 的相异性使得分离有收益 [PAPER FACT]
- 现代集群具备高带宽互联（NVLink/IB/ENI）使 KV 传输可行 [PAPER FACT]
- PagedAttention 的分页管理是碎片化主因，可通过形状重塑与段式分配改善 [PAPER FACT]
- NCCL 仅支持连续地址，合并传输可显著降 kernel 开销 [PAPER FACT]
- 请求到达近似 Poisson，负载可预测以做调度与弹性伸缩 [PAPER FACT]
- 异构 GPU 特性（显存带宽/容量）与任务阶段需求匹配可优化部署 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 论文未设显式 Limitations 节；隐含：RDMA 方案依赖特定硬件（Mellanox NIC），无此硬件需选其他管线 [PAPER FACT]
- 传输优化主要针对 NCCL，对其他后端增益未详 [PAPER FACT]
- 评估以 throughput/E2E 为主，未深入讨论 $ cost/能耗与 PD 权重拷贝 double memory 的代价 [PAPER FACT]
- 弹性伸缩策略需阈值调优，极端负载下重构开销未量化 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- 形状重塑 (B,L,2,H) 可能与现有 PagedAttention 内核深度耦合，迁移至其他引擎（SGLang/TensorRT-LLM）需重写 [AGENT INFERENCE]
- 段式分配的最小堆与双向对齐在高并发随机长度下能否保持高段连续性存疑，未给出碎片率量化 [AGENT INFERENCE]
- 评估以 LLaMA-3.1-8B/70B 与 LongBench summarization 为主，code/agent 长程多轮与 MoE 模型未测 [AGENT INFERENCE]
- Load-Aware 调度依赖全局 controller 单点，可能成为瓶颈与故障单点，未评估可扩展至百节点 [AGENT INFERENCE]
- 未开源代码，可复现性受限；NCCL 调用次数 23,469 的极端值是否代表典型有待验证 [AGENT INFERENCE]
- 异构实验仅 L20/H20 两型，未覆盖更广异构（A100/H100/4090/CXL）[AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 如何将段式分配与压缩（KIVI/GEAR/ChunkKV）及驱逐（H2O/SnapKV）联合，在传输与显存间全局最优？ [AGENT INFERENCE]
2. 能否实现跨框架的通用 KV 传输抽象，使 IPC/NCCL/RDMA 在 PagedAttention/RadixAttention 间无感切换？ [AGENT INFERENCE]
3. 全局 controller 的负载预测能否用学习型/强化学习替代阈值，以更好处理突发与分布漂移？ [AGENT INFERENCE]
4. 在 PD 解耦 + RAG chunk-cache 复用（CacheBlend/Cache-Craft/KVLink）共存时，传输与复用的协同调度如何设计？ [AGENT INFERENCE]
5. 异构感知下 P/D 放置的理论最优是什么？如何形式化带宽-显存-计算的联合优化？ [AGENT INFERENCE]
6. 对于 Mamba/RetNet 等非 Transformer，KV 传输形态如何演进？ [AGENT INFERENCE]
7. 双向段对齐的最坏情况复杂度与极端碎片化下回退策略为何？ [AGENT INFERENCE]
8. 能否通过 CXL 共享内存（如 Beluga）进一步近零化传输，FlowKV 方案在 CXL 上的收益如何？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **Splitwise (Patel et al. ISCA24)：** 首个系统化刻画两阶段异构并提出物理解耦至异构池 [PAPER FACT]
- **DistServe (Zhong et al. OSDI24)：** 形式化 goodput 与独立并行放置算法 [PAPER FACT]
- **Mooncake (Qin et al. FAST25)：** Kimi 生产架构，分布式 KV 池 + RDMA 87GB/s [PAPER FACT]
- **TetriInfer (Hu et al. 2401.11181) / DéjàVu：** 解耦与容错相关 [PAPER FACT]
- **vLLM (Kwon et al. SOSP23) PagedAttention：** 基座内存管理 [PAPER FACT]
- **KVDirect (Chen et al. 2501.14743)：** 分布式解耦 LLM 推理另一传输优化 [PAPER FACT]
- **CacheBlend / RAGCache / Cache-Craft / KVLink：** RAG KV 复用可与解耦传输协同 [AGENT INFERENCE]
- **Beluga (CXL) / Dynamic Placement：** 异构内存放置理论 [AGENT INFERENCE]
- **Sarathi-Serve / Orca：** 调度与流水互补 [AGENT INFERENCE]

---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
