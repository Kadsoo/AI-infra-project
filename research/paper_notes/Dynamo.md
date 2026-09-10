# Paper Metadata

- **Title:** NVIDIA Dynamo: A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models [PAPER FACT]
- **Authors:** Amr Elmeleegy, Harry Kim, David Zier, Kyle Kranen, Neelay Shah, Ryan Olson, Omri Kahalon [PAPER FACT] — Affiliation: NVIDIA [PAPER FACT]
- **Venue:** NVIDIA Technical Blog 18 Mar 2025, GTC 2025 Announcement [PAPER FACT]; Docs https://docs.nvidia.com/dynamo/ [PAPER FACT]; GitHub https://github.com/ai-dynamo/dynamo Apache 2.0, 7.9k stars / 1.5k forks (as of 2026-08-27) [PAPER FACT]; No peer-reviewed paper, no arXiv 2504.xxxxx exists for Dynamo framework [AGENT INFERENCE] — websearch "Dynamo LLM inference arXiv" 2026-08-27 returns DynamoLLM arXiv:2408.00741 (different system, LLM inference cluster design for performance/energy) and no Dynamo framework paper [PAPER FACT]; DynamoLLM is unrelated [AGENT INFERENCE]
- **DOI/URL:** https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/ [PAPER FACT]; GitHub https://github.com/ai-dynamo/dynamo [PAPER FACT]; Docs https://docs.nvidia.com/dynamo/latest/ [PAPER FACT]; NIM integration https://developer.nvidia.com/dynamo [PAPER FACT]
- **Code:** https://github.com/ai-dynamo/dynamo (Rust + Python), containers nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.1, tensorrtllm-runtime:1.4.1, vllm-runtime:1.4.1; PyPI ai-dynamo[sglang/vllm] [PAPER FACT]; NIXL https://github.com/ai-dynamo/nixl , Grove https://github.com/ai-dynamo/grove , ModelExpress https://github.com/ai-dynamo/modelexpress [PAPER FACT]
- **Version:** Dynamo 1.0 production-ready Mar 2025; latest containers 1.4.1 (2026) [PAPER FACT]
- **Reading Source:** default.webfetch https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/ + default.webfetch https://docs.nvidia.com/dynamo/ + https://github.com/ai-dynamo/dynamo (README) + default.websearch "NVIDIA Dynamo LLM inference arXiv 2025" 10 results [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；Dynamo 非论文，以博客/文档/GitHub 为 facts。

## 1. Problem [PAPER FACT]

- 推理成本成为 MaaS 主要瓶颈：模型规模 2000x 增长（自 Triton 2018）、上下文 1M+（RAG/视频/代码）、输出长度因推理/智能体工作流大幅增加，用户+UX 要求 fast tokens = revenue [PAPER FACT]
- 传统单 GPU/单节点耦合部署将 prefill（compute-bound，处理输入生成首 token，二次方注意力）与 decode（memory-bound，逐 token 生成，受 KV cache 显存带宽限制）置于同 GPU/同节点，资源利用失衡，长输入时效率低且 SLO 难控 [PAPER FACT]
- 多节点分布式部署需跨数百 GPU 的编排与高效数据搬运：模型并行（Tensor/Pipeline/Expert）依赖低延迟高吞吐通信，PD 解耦后还需跨 prefill↔decode 的 KV 快速迁移 [PAPER FACT]
- 单机推理引擎（SGLang/TensorRT-LLM/vLLM/PyTorch）仅优化单 GPU/单节点，无法提供跨节点协调、智能路由、多级缓存与弹性伸缩 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- Triton 已被下载 >1M 次，被 Amazon/Microsoft/Oracle/DocuSign/Perplexity/Snap 等生产使用，但新一代推理工作流需多模型 agentic 协作与跨节点分布 [PAPER FACT]
- Disaggregated serving 已被 Splitwise/DistServe/TetriInfer 等验证可提升吞吐，但需框架化落地并与 KV 感知路由、层次化缓存、自动扩缩容协同 [PAPER FACT]
- 推理需求波动大：summarization 等长 ISL 短 OSL 突发会瞬间打满 prefill 而 decode 空闲，需在 disaggregated vs aggregated 间动态抉择并弹性再分配 GPU [PAPER FACT]
- KV cache 重算代价随输入二次方增长，常见系统提示、多轮对话、智能体工作流中重复计算显著；异构存储（HBM→DRAM→SSD→对象存储）可将成本降低数个量级但需统一抽象 [PAPER FACT]
- 大模型冷启动慢（权重加载与分片）影响弹性，ModelExpress 的 NIXL/NVLink GPU-GPU 权重流式可 7x 加速启动 [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **Prefill-Decode 干扰：** 同 GPU 上 compute-bound prefill 抢占 memory-bound decode 的 batch，导致 TTFT/ITL 抖动，长上下文时尤甚（Fig.1）[PAPER FACT]
2. **并行策略僵化：** prefill 适低 TP 以减通信，decode 适高 TP 以提访存；耦合部署无法分别为两阶段选最优并行度与硬件 [PAPER FACT]
3. **KV 重算：** 朴素轮询/负载路由忽略 KV 位置，导致跨请求前缀/会话 cache miss 而重算；上下文越大、GQA 越小开销越大 [PAPER FACT]
4. **跨级传输：** KV 在 GPU HBM、host DRAM、SSD、对象存储、远端节点间搬运需兼容 NVLink/IB/RoCE/Ethernet 与 GPUDirect RDMA/UCX/GDS/S3 等多后端，硬件/协议异构导致 API 碎片 [PAPER FACT]
5. **调度复杂度：** 数百 GPU 时根据队列等待、KV 传输时间、SLO（TTFT/ITL）估算 disaggregated vs aggregated 收益并动态增减两阶段 GPU 池的决策空间大 [PAPER FACT]
6. **多跳流水与故障：** 多模型 agentic 流水的流量瓶颈与单点故障会级联，缺乏请求迁移与健康检查 [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**Dynamo = 推理引擎之上的编排层（orchestration layer above inference engines），将多 GPU/多节点集群变为协调的推理系统，四大创新协同：Dynamo Planner + Smart Router + Distributed KV Cache Manager (KVBM) + NIXL [PAPER FACT]**

- **Disaggregated Prefill/Decode：** 将 prefill 与 decode 分至独立可伸缩 GPU 池，各池独立选并行度与硬件；支持在轻载/波动时将 GPU 在两池间迁移或回退至 aggregated 传统服务 [PAPER FACT]
- **KV-Aware Routing (Smart Router)：** 对入请求哈希并在 Radix Tree 中跟踪全集群 KV 位置（含 GPU/ host/SSD 层次），计算与已活跃 KV 块的重叠分数，结合负载均衡选最优 worker，最小化重算；支持 SGLang/vLLM/TensorRT-LLM [PAPER FACT]
- **Hierarchical KV Cache Management (KVBM)：** 框架无关的分布式 KV 块管理，支持 GPU→CPU (host)→SSD→远端对象存储的多级 offload，含插入/驱逐策略与 lifecycle 管理，跨节点/集群可见性（global KV events）[PAPER FACT]
- **NIXL (NVIDIA Inference Transfer Library)：** 高吞吐低延迟点对点库，统一数据移动语义，支持非阻塞/非连续传输，抽象 memory sections（HBM/DRAM/file/object），自动选 UCX/GDS/S3/custom 后端，适配 NVLink(C2C/NVSwitch)/IB/RoCE/Ethernet [PAPER FACT]
- **Planner (SLA-Based)：** 持续监控 GPU 容量指标，结合应用 SLO（TTFT/ITL）与 profiling，决定是否 disaggregate、增减各阶段 GPU、或触发 autoscaling；与 Kubernetes (Grove, K8s Inference Gateway) 集成 [PAPER FACT]
- **ModelExpress + Grove + AIConfigurator：** ModelExpress 经 NIXL/NVLink 实现权重 GPU-GPU 流式；Grove 为 K8s operator 做拓扑感知 gang 调度（NVL72）；AIConfigurator 秒级仿真 10K+ 部署配置以择优 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **架构（Fig.3）：** API Server → Smart Router → Disaggregated Serving（Prefill Worker + Decode Worker）→ NIXL 传输；Planner 旁路监控并下发扩缩容/迁移指令 [PAPER FACT]
- **前端与路由拓扑：** 支持 Dynamo-native Frontend 路由（Frontend 持有 HTTP 与 Router 决策，无需外部网关）与 Gateway API + GAIE EPP 插件（Gateway → EPP → Frontend sidecar --router-mode direct）两种 [PAPER FACT]
- **后端集成：** 适配层对接 SGLang、TensorRT-LLM、vLLM、PyTorch；各后端特性矩阵见 docs（Disaggregated Serving/KV-Aware Routing/Planner/Multimodal/Tool Calling 均 ✅，KVBM 在 SGLang 🚧）[PAPER FACT]
- **通信抽象：** NIXL Core 含 metadata/memory，Backend API 统一 KV 数据进出，支持 UCX/GDS/S3；memory sections 屏蔽差异 [PAPER FACT]
- **存储层次：** KVBM 管理 HBM/DRAM/SSD/网络存储分层，策略：热数据留 GPU，冷数据下沉至共享 host/SSD/对象存储；支持 S3/Azure blob 与 petabyte 级存储 [PAPER FACT]
- **弹性与容错：** Planner + Grove 实现 SLA 驱动 autoscaling 与故障时的 canary 健康检查 + in-flight 请求迁移 [PAPER FACT]
- **部署：** 容器/ PyPI / Kubernetes 三种；K8s 推荐路径 DynamoGraphDeploymentRequest CRD 零配置部署（model + HW + SLA），AIConfigurator 自动 profile，Planner 优化拓扑；已提供 Qwen3-32B-FP8/TRTLLM、DeepSeek-R1/SGLang、Kimi-K3/vLLM 等 recipes [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **Throughput (requests/s, tokens/s per GPU)** 与 **Goodput under SLO** [PAPER FACT]
  - **TTFT (time to first token)** 与 **ITL (inter-token latency)** [PAPER FACT]
  - **KV cache hit rate / recomputation saved**（Smart Router 重叠分数）[PAPER FACT]
  - **Scalability (GPUs/nodes)** 与 **Cost / TCO** [PAPER FACT]
  - **Cold-start time**（ModelExpress）[PAPER FACT]
- **Secondary：**
  - NIXL 传输带宽/延迟与 KV 迁移时间 [PAPER FACT]
  - SLA breach 率 [PAPER FACT]
  - 不同并行配置 (TEP/PP/DP/TP) 与 ISL/OSL 下的对比 [PAPER FACT]

## 7. Baselines [PAPER FACT]

- **Without Dynamo (aggregated/inflight batching)：** 同模型同硬件下单池聚合服务，如 TensorRT-LLM 16PP4DP4 TEP16 vs Dynamo Context EP4DP16 + Generation EP64DP3；vLLM TP8DP2 vs Dynamo TP2DP4/TP8 [PAPER FACT]；DeepSeek-R1 on B200 without Dynamo vs on GB200 NVL72 with Dynamo [PAPER FACT]
- **Load-based / Round-robin routing：** 对比 Smart Router 的 KV-aware 路由 [PAPER FACT]
- **No KVBM（仅 GPU HBM 缓存）：** 对比启用 KVBM 多级 offload 的 petabyte 扩展 [PAPER FACT]
- **No NIXL（通用 UCX/GDS 直调）：** 对比 NIXL 统一抽象的延迟 [AGENT INFERENCE]
- **Commercial/External benchmarks：** InferenceX / InferenceXv2、Baseten、Alibaba APSARA 上报的第三方对比 [PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Models：** DeepSeek-R1 671B（主标杆）、DeepSeek-V3 (H200)、Llama 70B (Hopper)、Qwen3-Coder 480B、Qwen3-32B-FP8、Mistral Large 3、Kimi K2/K3、Llama Distill 70B (8x DL 70B) 等 [PAPER FACT]
- **ISL/OSL：** DeepSeek-R1 32K/8K (FP4, TensorRT-LLM)，Llama 70B 3K/50 (FP8, vLLM)，Baseten Qwen3-Coder 480B 真实 R1 请求 100K 条平均 4K/800 [PAPER FACT]
- **Dataset：** 100K real R1 requests (Baseten)、InferenceX 上的 DeepSeek-R1 流量、Alibaba APSARA 生产 trace [PAPER FACT]
- **部署形态：** GB200 NVL72 (72 GPUs per rack, Blackwell)、B200、H200、HGX-H100 2 nodes 8x GPUs、GB300 NVL72 预研 [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **Rack/Node：** GB200 NVL72 (NVIDIA Blackwell, Grace CPU + Blackwell GPU, NVLink C2C/NVSwitch, 72 GPUs per NVL72 rack) 为主标杆平台 [PAPER FACT]；GB300 NVL72（下一代）用于 750x 预测 [PAPER FACT]；HGX-H100、H200、B200 为对比平台 [PAPER FACT]
- **网络：** NVLink (C2C/NVSwitch)、Quantum InfiniBand、Spectrum Ethernet；NIXL 适配 IB/RoCE/Ethernet 与 GPUDirect RDMA/GDS [PAPER FACT]
- **存储：** HBM (GPU)、DRAM (host)、local SSD、networked object storage (S3/Azure blob) 的层次化；Dell PowerScale + NIXL 集成 [PAPER FACT]
- **软件：** CUDA 12+, SGLang/TensorRT-LLM/vLLM/PyTorch；Dynamo Rust (performance) + Python (extensibility)；Kubernetes (Grove, GAIE)；NIM for enterprise [PAPER FACT]
- **精度：** FP4 (DeepSeek-R1 TRTLLM)、FP8 (Llama 70B vLLM) [PAPER FACT]
- **集群规模：** 单机多 GPU 至数千 GPU 分布式；示例 2x HGX-H100 (16 GPUs) 用于 Smart Router 加速图 [PAPER FACT]

## 10. Main Results [PAPER FACT]

- **GB200 NVL72 DeepSeek-R1 671B：** Dynamo disaggregated vs 无 Dynamo，吞吐提升 **up to 30x**（博客 Fig.2 左，TRTLLM FP4 32K/8K, TEP16PP4DP4 vs EP4DP16+EP64DP3, projected performance subject to change）[PAPER FACT]；GitHub README 同场景引 InferenceX 数据 **7x higher throughput per GPU on GB200 NVL72 w/ Dynamo vs B200 without** [PAPER FACT]；GB300 NVL72 预研 **750x higher throughput** (InferenceXv2) [PAPER FACT]
- **Hopper Llama 70B：** Dynamo **>2x throughput** vs 无 Dynamo（博客 Fig.2 右，vLLM FP8 3K/50, TP8DP2 vs TP2DP4+TP8）[PAPER FACT]
- **Smart Router：** 2x HGX-H100, 8x DeepSeek-R1-Distill-Llama-70B, vLLM FP8 TP2, 100K real R1 requests (avg 4K/800)，TTFT 加速 **2x**，平均请求延迟显著降低（Fig.5）[PAPER FACT]；Baseten 基准 Qwen3-Coder 480B **2x faster TTFT** [PAPER FACT]
- **ModelExpress：** DeepSeek-V3 on H200 权重流式 **7x faster model startup** [PAPER FACT]
- **Planner/Autoscaling：** Alibaba APSARA 2025 生产 trace 上 **80% fewer SLA breaches at 5% lower TCO** [PAPER FACT]
- **集成生态：** Dell PowerScale + NIXL **19x faster TTFT** [PAPER FACT]；Moonshot Kimi K2 on GB200 **10x inference speedup**，Mistral Large 3 on GB200 **10x faster**（MarktechPost/Quantum Zeitgeist 转述）[PAPER FACT，第三方报道，转述 Dynamo 生态结果]
- **注意：** 以上多为博客/GitHub 引用的 projected 或第三方基准，非 peer-reviewed 论文的受控实验；具体误差与复现条件 [NOT REPORTED] [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- 推理负载可分为 prefill 与 decode 两阶段且资源需求正交，可用不同并行度与硬件分别优化 [PAPER FACT]
- KV cache 可哈希追踪且跨请求/跨节点复用收益显著（系统提示/多轮/智能体共享前缀）[PAPER FACT]
- 集群具备高速互连（NVLink/IB）以使 KV 迁移开销可被 disaggregation 收益覆盖；否则 Planner 应回退至 aggregated [PAPER FACT]
- 工作负载可被 profiling 且 SLO 可量化（TTFT/ITL），Planner 可基于容量指标与 SLO 做决策 [PAPER FACT]
- 后端引擎暴露 KV 事件或可预测路由（prediction-based without KV events），以支撑 Router 的全局视图 [PAPER FACT]
- 多级存储的成本-容量权衡下，冷 KV 下沉至 DRAM/SSD/对象存储仍可在需要时快速召回 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 博文未设独立 Limitations 小节；以下为文档与博客中隐含的 caveats [PAPER FACT]
- 性能数据多为 projected subject to change，需以实际部署与新版本验证 [PAPER FACT]
- 特性矩阵显示部分后端/特性尚未 GA（如 SGLang KVBM 🚧），LoRA/request migration/speculative decoding 等存在交互限制 [PAPER FACT]
- 架构依赖高速互连与 K8s 生态，单 GPU 或无高速网环境收益有限，作者建议单卡场景直接用推理引擎 [PAPER FACT]
- 具体调度策略、阈值与 profiling 开销未公开细节，复现需依赖官方实现 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- 缺乏 peer-reviewed 评估：无统一定义的 goodput/SLO 对比方法，跨平台（GB200 vs B200）对比混杂硬件代差，难以剥离框架增益 [AGENT INFERENCE]
- 控制面潜在瓶颈：Planner/Smart Router 为中心化决策点，数千 GPU 下状态同步与 Radix Tree 扩展性未量化 [AGENT INFERENCE]
- 策略不透明：KVBM 驱逐/预取、Planner 的 disaggregated vs aggregated 切换阈值未形式化，缺乏可解释性与可调性分析 [AGENT INFERENCE]
- 故障与一致性：跨节点 KV 的一致性、版本失效、多租户安全隔离未深入讨论 [AGENT INFERENCE]
- 成本核算不完整：TCO 对比未含存储与网络成本，petabyte 缓存的实际性价比未披露 [AGENT INFERENCE]
- 与学术基线（Splitwise/DistServe/Mooncake/CacheBlend）缺乏 head-to-head 受控对比 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 如何在不依赖中心化 Router 的前提下实现可扩展的全局 KV 视图，例如去中心化 Radix Tree 或 RDMA 原语？ [AGENT INFERENCE]
2. Planner 能否引入学习式 workload 预测与强化学习，以在线调优 disaggregated/aggregated 阈值与 GPU 池比例？ [AGENT INFERENCE]
3. NIXL 的统一抽象在 RoCE vs IB vs Ethernet 上的实际带宽利用率与尾延迟差异如何，是否需要拓扑感知路径选择？ [AGENT INFERENCE]
4. KVBM 的分层策略与压缩（KIVI/GEAR/CacheGen）能否联合，压缩后 KV 如何仍保持可路由与可零拷贝？ [AGENT INFERENCE]
5. 多租户与多模型并发（agentic 流水）下，如何保证公平性与 SLO 隔离，避免热点模型饿死？ [AGENT INFERENCE]
6. 冷启动之外的弹性：能否实现秒级按需扩缩容与跨地域一致性缓存？ [AGENT INFERENCE]
7. 如何提供形式化 goodput 模型，量化不同 ISL/OSL 分布下 disaggregation 的理论上限？ [AGENT INFERENCE]
8. 安全与隐私：跨节点 KV 明文传输与存储的加密、鉴权与审计如何设计？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **Splitwise (Patel et al. 2311.18677) / DistServe (Zhong et al. 2401.09670) / TetriInfer (Hu et al. 2401.11181)**：PD 解耦与 goodput 优化，Dynamo 的学术先驱，博客/文档对比引用 [PAPER FACT]
- **Mooncake (Qin et al. 2407.00079) KVCache-centric Disaggregation**：CPU DRAM/SSD 解耦缓存池与 Conductor 调度，GB200 场景的学术对照 [PAPER FACT]
- **vLLM (Kwon et al. SOSP 23) PagedAttention / SGLang (Zheng et al. 2312.07104) RadixAttention**：Dynamo 适配的核心引擎与前缀复用思想 [PAPER FACT]
- **Sarathi-Serve / Orca**：连续批处理与 chunked prefill，与 disaggregation 正交 [PAPER FACT]
- **CacheGen (Liu et al. SIGCOMM 24) / CachedAttention (Gao et al. ATC 24) / LMCache / RAGCache / CacheBlend**：KV 编码/层次化放置/选择性重算，可与 Dynamo KVBM/NIXL 叠加 [AGENT INFERENCE]
- **NIXL / UCX / GPUDirect RDMA / GDS / S3**：传输库与存储后端，Dynamo 通信基座 [PAPER FACT]
- **DynamoLLM (Im et al. arXiv:2408.00741)**：同名不同系统，聚焦 LLM 集群性能/能效设计，需区分于 NVIDIA Dynamo [PAPER FACT]
- **LLM-d / Grove / AIConfigurator**：K8s 原生分布式推理与自动配置，与 Dynamo 生态互补 [PAPER FACT]

---
## Review Log

Reviewer: Paper Reader — Supplement (Coverage Gaps) — 2026-08-27
Scope: webfetch blog + docs + github + websearch 10 结果交叉核验
Webfetch verification: 30x GB200 NVL72 DeepSeek-R1 (TRTLLM FP4 32K/8K) vs >2x Llama70B Hopper (vLLM FP8 3K/50) 已核 Fig.2；7x throughput per GPU (InferenceX), 7x startup (ModelExpress), 2x TTFT (Baseten Qwen3-Coder 480B), 80% fewer SLA breaches (APSARA), 750x GB300 已核 README；Planner/Smart Router/KVBM/NIXL 四组件架构已核 Fig.3；无 arXiv 2504.xxxxx 已核 search
Problems Found: 性能多为 projected/第三方转述，非受控论文数据，已以 [PAPER FACT] 标注来源并注明 projected；特性矩阵 SGLang KVBM 🚧 已保留
Confidence: Medium (来源为博客/文档非论文，数值需以未来论文/复现为准)
