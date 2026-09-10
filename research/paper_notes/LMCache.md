# Paper Metadata

- **Title:** LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference [PAPER FACT]
- **Authors:** Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, Junchen Jiang [PAPER FACT] — Affiliations: 1 Tensormesh Inc., 2 University of Chicago, * equal contribution for first three [PAPER FACT]
- **Venue:** arXiv:2510.09665 [cs.LG] v1 8 Oct 2025, v2 5 Dec 2025 (379 KB) [PAPER FACT]; preprint, CC BY 4.0 [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2510.09665 doi:10.48550/arXiv.2510.09665 [PAPER FACT]; HTML https://arxiv.org/html/2510.09665v2 [PAPER FACT]
- **Code:** https://github.com/LMCache/LMCache [PAPER FACT]; integrated in vLLM Production Stack, Dynamo, llm-d, KServe, SGLang OME, AIBrix [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2510.09665v2 + https://arxiv.org/abs/2510.09665 [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1. Problem [PAPER FACT]

- 传统 KV cache 仅驻留 GPU 显存、服务单 query 的单引擎生命周期，加速 decode 已不够 [PAPER FACT]
- 行业趋势要求 KV 移出 GPU：跨 query 前缀复用（context caching）以省重复 prefill，以及跨引擎/跨 GPU 的 Prefill-Decode 解耦（PD disaggregation）以隔离计算与访存 [PAPER FACT]
- 现实统计显示需存储的 KV 总量随时间快速增长并远超 GPU 容量，复用频率亦上升，GPU-only 已无法承载 [PAPER FACT]
- 缺乏高效、通用、与快速演进的推理引擎兼容的 KV 抽取/加载/存储/传输层，成为落地瓶颈 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- **趋势1 跨 query 缓存：** 文档分析、多轮对话、固定 system prompt/长 preamble 等场景共享长前缀，复用可大幅降低 TTFT 与 GPU-hours [PAPER FACT]
- **趋势2 PD 解耦：** 将 prefill 与 decode 放不同 GPU/节点，避免 prefill 干扰延迟敏感的 decode，提升尾延迟与利用率 [PAPER FACT]
- **真实统计：**
  - 周级 KV 存储总量中超出 GPU 的部分显著增长 [PAPER FACT]；图3展示 5 周内蓝（超 GPU）快速上升 [PAPER FACT]
  - 超 GPU 部分的 reuse per token（复用 token / 全部存储 token）在 top-10 用户上持续上升 [PAPER FACT]；过去一周 >19% 用户平均每存储 token 被复用 >1.5 次 [PAPER FACT]
  - Docker pull 与用户数持续增长（图1）[PAPER FACT]
- **远端存储性能逆转：** 传统 S3 仅 ~100 MBps 认为会增加延迟，而 S3 Express 已达 ~1 GBps，实测远端S3 Express (~1 GBps) 加载在特定长上下文下可比 full prefill 快 22-32% (非普遍) [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **Paged memory 导致小 IO 低效：** vLLM/SGLang 采用 paged attention，页 16-64 KB（例 Llama-3.1-8B 每页 62.5 KB，每页 16 tokens 单层）[PAPER FACT]；分散小页导致大量小 IO，带宽无法打满 [PAPER FACT]；Table1：64 KB ~4 GBps, 256 KB ~13 GBps, 1 MB ~30 GBps, 10 MB ~46 GBps, 需 ~16 MB 才能打满 8x400 Gbps NIC (aggregate 400Gbps x8) [PAPER FACT]，PCIe 5.0 需 1-2 MB 才达 75-80% 带宽 [PAPER FACT]；原生 torch.save/load 仅 sub-1GB/s 且多余 CPU-GPU 拷贝 [PAPER FACT]
2. **引擎快速演进兼容难：** 2025 年平均每 4 天发布一个新 LLM，每周 15-20 新开放权重模型 [PAPER FACT]；新模型/注意力核（Sliding Window、MLA 等）改变 KV 布局，缓存层需频繁适配 [PAPER FACT]
3. **缺乏管理 API：** 上层 router/scheduler 需感知 KV 位置做 cache-aware 路由，业务需 pin/compress/move/clear 等显式管理；缺统一接口导致重复存储与不可预测驱逐 [PAPER FACT]
4. **跨存储/网络 heterogeneity：** 需同时支持 CPU DRAM、本地盘、远端盘、Redis/S3/NFS/WEKA 等与 Ethernet/RDMA/NVLink，且需零拷贝与流水 [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**LMCache = 位于推理引擎与异构存储/网络之间的统一高性能 KV 缓存层，提供高效数据面 + 标准化连接器 + 一等管理面 [PAPER FACT]**

- **高效数据面：** 可配置大 chunk（默认 256 tokens/chunk，远大于 16-token 页）配合 streaming GPU buffer 将分散页聚合为连续块，再用 DMA 批量搬运以打满带宽 [PAPER FACT]；支持并行多 tier 存取与 delayed decode 批量写（攒一 chunk 再写）[PAPER FACT]；层间流水与异步预取重叠计算与 IO [PAPER FACT]；零拷贝 via 引用计数与 dynamic offloading 最小化拷贝 [PAPER FACT]
- **标准化连接器：** 模块化 KV cache connector 解耦 LMCache 与 vLLM/SGLang 内部实现，适配调度器-执行器分离、前缀缓存一等、piece-wise CUDA graph 等设计 [PAPER FACT]；接口分调度器侧（get_num_new_matched_tokens, update_state_after_alloc, build_connector_meta）与执行器侧（start_load_kv, wait_load_kv, start_store_kv, wait_store_kv）[PAPER FACT]
- **一等控制面：** 中心化 Controller（manager + per-instance worker）维护全局 token pool，提供 lookup/move/clear/pin-unpin/compress-decompress、batched_admit/batched_evict、batched_p2p_lookup 等内外 API，支撑路由、迁移、P2P 共享 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **架构位置：** 介于 vLLM/SGLang 与存储/网络之间（图5），统一承载存/取/查三路径：Store 走 connector 准备元数据 -> token processor 定新 token -> storage manager 经 transfer channel 批量落盘；Retrieve 走 token processor 定命中 -> event manager 查 query ID 决定复用地址或查存储 -> GPU connector 层间异步加载；Lookup 走 Controller token pool [PAPER FACT]
- **Batched Operations：** streaming buffer + 定制 CUDA kernel 将页 coalesce 到 chunk 再 DMA；支持多源多目标并发（PCIe 全双工）及 decode 延迟批量 [PAPER FACT]
- **Compute-IO 重叠：** 双 CUDA stream 层流水：推理前载入第一层，推理第一层时异步抓第二层，仅需单层大小的 GPU buffer [PAPER FACT]；调度准入到实际推理的排队间隙做异步预取至更快 tier [PAPER FACT]
- **Zero-Copy & Dynamic Offloading：** 多目的地写时引用计数共享数据，完成后递减释放 [PAPER FACT]；对 vLLM free pages 三指针（start/current/end）仅复制子集以平衡复制率与分配 stall [PAPER FACT]
- **连接器实现：** 调度器侧将命中 token 视为 vLLM 原生 prefix cached token 影响调度；执行器侧在模型/注意力前后打 hook 支持 bulky 与 layer-wise 两模式；支持 None 返回让 vLLM 先处理他请求以重叠 IO [PAPER FACT]
- **Controller：** Manager 单进程全局协调，Worker 随实例常驻；外部 API 经 manager 分发，内部 API 由 worker 主动上报/查询 [PAPER FACT]
- **生态：** vLLM 侧 API 已上线 6 个月以上，被 Dynamo/llm-d/AIBrix/vLLM Production Stack 及多家私有连接器采用 [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **TTFT (time to first token)** 均值与 P95，反映 prefill 快慢 [PAPER FACT]
  - **ITL (inter-token latency)** 平均 token 间延迟 [PAPER FACT]
  - **Throughput / QPS** 在同 TTFT 约束下可支撑的最大请求率 [PAPER FACT]
  - **端到端延迟与传输带宽** 组件分解 [PAPER FACT]
- **Secondary：**
  - 不同场景（CPU offload、central storage 15 Gbps、PD NVLink）下 TTFT/ITL vs QPS 曲线 [PAPER FACT]
  - 组件级：CPU 加载带宽、PD 传输延迟、异步重叠收益 [PAPER FACT]
  - 敏感度：上下文长度 vs 网络带宽的 crossover、SGLang 对比 [PAPER FACT]
  - 生产洞察：reuse per token、命中率、截断影响、远端 vs 本地 [PAPER FACT]

## 7. Baselines [PAPER FACT]

- **Basic vLLM v0.10.2：** 默认前缀缓存但仅 GPU 显存，容量小 [PAPER FACT]
- **Basic vLLM CPU Offloading v0.11.0：** 官方 per-layer per-16-token 粒度搬运，无带宽优化 [PAPER FACT]
- **Commercial offerings #1 / #2：** 保留 GPU 的托管端点（闭源），9 月 10 日实测黑盒对比 [PAPER FACT]
- **vLLM Native PD Disaggregation：** 基于 NIXL 逐页拷贝的官方 PD 实现 [PAPER FACT]
- **SGLang 原生：** 无 offload 与有原生 CPU offload 对比（Qwen3-32B TP=2）[PAPER FACT]
- **存储层对比：** Mooncake、Redis、InfiniStore、3FS 等（定性讨论，未端到端同等对比）[PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Models：** meta-llama/Llama-3.1-8B-Instruct, Sao10K-L3-8B, meta-llama/Llama-3.1-70B-Instruct, Qwen/Qwen2.5-Coder-32B-Instruct, Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8, Qwen/Qwen2.5-72B-Instruct [PAPER FACT]；SGLang 评测用 Qwen3-32B [PAPER FACT]
- **Datasets：**
  - 多轮 Q&A 拟人（文档分析）：每 query 10K tokens（约 12 页 PDF）+ 唯一短问题，输出 100 tokens，起始 40 用户按 Poisson 到达 [PAPER FACT]；Llama-3.1-8B-Inst 取 20K 输入 [PAPER FACT]
  - LongBench  TriviaQA 长上下文 QA [PAPER FACT]
  - vLLM 官方 random 输入输出脚本 [PAPER FACT]
  - 真实 trace：来自 company F 与 G 的输入输出分布，拉伸至 1 小时完成，保留数日真实分布 [PAPER FACT]
- **输入输出：** PD 评测用 8K 输入 / 200 输出 random；灵敏度覆盖不同长度 [PAPER FACT]
- **容量与网络：** CPU DRAM 上限 500 GB；central storage 带宽 15 Gbps；PD 用 NVLink；带宽敏感度 32/64/128 Gbps [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **单节点：** 8x H100 服务器（GMI Cloud）[PAPER FACT]；按模型取最小可启动 GPU 数 [PAPER FACT]
- **多节点/远端：** 同 GPU 数 + 远端 CPU 内存存储后端 [PAPER FACT]；PD 场景 prefiller 与 decoder 各一套同等 GPU 数，NVLink 互联 [PAPER FACT]
- **网络：** 8x Broadcom Thor-2 400Gbps NIC 示例用于 Table1 带宽饱和分析 [PAPER FACT]；32/64/128 Gbps 与 15 Gbps 远端实验 [PAPER FACT]
- **存储后端：** 本地 CPU DRAM/盘、远端 DRAM/盘、对象存储 S3、Redis/NFS/WEKA/GDS/Mooncake Store/NIXL/InfiniStore/Valkey 等（当前支持 8+ 后端，4 类处理器：NVIDIA/AMD/Ascend/TPU）[PAPER FACT]
- **软件：** vLLM v0.10.2/v0.11.0, SGLang, LMCache v0.3.6 [PAPER FACT]；精度 [NOT REPORTED] 未披露 [NOT REPORTED]

## 10. Main Results [PAPER FACT]

- **Single-node CPU Offload（图8，多轮 10K）：**
  - 低 QPS (1) 时 LMCache TTFT 比最强基线小 1.9-8.1x [PAPER FACT]；同 TTFT 下吞吐高 2.3-14x 跨 5 模型 [PAPER FACT]；ITL 在 QPS=1 时比最强基线小 7%-92% [PAPER FACT]；basic CPU offload 在 Qwen3-Coder-480B 无法运行，商业版不支持该模型 [PAPER FACT]
  - 抽象总述：结合 vLLM 最高 15x 吞吐提升，至少 2x 延迟降低 [PAPER FACT]
- **Real-trace（图9/10，F/G）：** 跨 5 模型高 QPS 下 TTFT 至少小 3.7-6.8x（F 上 4.4-6.6x），ITL 小 19-58%（F 上 34-58%）[PAPER FACT]
- **Central Storage（图11，TriviaQA，15 Gbps）：** 同 TTFT 下吞吐 1.3-3x 优于 basic vLLM [PAPER FACT]；远端命中更高但加载可能超过 prefill，尤其短上下文/小模型需自适应选择 [PAPER FACT]
- **PD Disaggregation（图12-14，8K/200 random）：** P95 显著优；均值 TTFT 降 1.53-1.84x（摘要 1.5-1.8x），ITL 降 1.12-1.66x（摘要 1.1-1.7x）跨 4 模型 [PAPER FACT]；异步重叠使端到端降 1.46x [PAPER FACT]
- **Component-wise：**
  - CPU 加载带宽：LMCache 400 Gbps vs vLLM native 88 Gbps，因 chunk 粒度降低 per-transfer  overhead [PAPER FACT]
  - PD 传输因逐页 vs chunk，前者带宽利用低，LMCache 显著快 [PAPER FACT]
- **Sensitivity：** 32 Gbps 时仅 >256K 上下文加载优于 prefill；64/128 Gbps 时全长度均优，故需在低带宽下自适应开关 [PAPER FACT]；B200 上 prefill vs 加载 crossover 明确 [PAPER FACT]
- **SGLang（图16，Qwen3-32B TP=2）：** LMCache CPU offload 显著优于无 offload，与 SGLang 原生 CPU offload 相当，但后者缺分布式层次化后端 [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- 前缀命中可复用，且 hit token 在调度器视为原生 prefix cached 即可正确调度 [PAPER FACT]
- Chunk 化（256 tokens）与页（16 tokens）聚合可充分打满带宽，且 DMA + kernel 开销可摊薄 [PAPER FACT]
- 排队间隙存在且可用于预取；层流水仅需单层 buffer [PAPER FACT]
- 引用计数与三指针动态复制足以平衡内存压力与分配 stall [PAPER FACT]
- 中心化 Controller 的内存全局视图足够实时，batched 上报可跟上实例 admit/evict 频率 [PAPER FACT]
- 远端/CPU 容量远大于 GPU，命中率随容量单调提升 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 远端加载延迟在短上下文/小模型/低带宽下可能超过 prefill，需自适应在加载 vs 重算间抉择，当前仅提出原则未全自动化 [PAPER FACT]
- 截断等上下文变换会使命中大跌（85%->45%），提示用户应避免动态增删上下文 [PAPER FACT]
- 当前对学术侧灵活修改 attention（如 token 丢弃）的支持不足，已被 deprioritized，更偏工业高性能与稳定 [PAPER FACT]
- Python 实现虽经优化，工业界正转向 Rust/C++ 以极致效率，团队仍坚持 Python 以换取迭代速度与社区贡献 [PAPER FACT]
- 评估以 vLLM 为主，SGLang 仅单点验证；更多引擎与异构硬件组合待扩展 [PAPER FACT]
- 具体 DRAM/SSD 容量与成本、能耗未量化 [NOT REPORTED] [PAPER FACT 侧未披露]

## 13. Inferred Limitations [AGENT INFERENCE]

- 中心化 Controller 可能成为规模瓶颈与单点故障；大规模（百实例）下 batched 同步的延迟与一致性未评估 [AGENT INFERENCE]
- 500 GB CPU 上限在真实企业兆 token 规模下仍可能不足，多级分层（CPU-SSD-S3）的自动分层与驱逐策略未形式化 [AGENT INFERENCE]
- 引用计数零拷贝与动态 offloading 的正确性在并发与异常路径下缺乏形式验证 [AGENT INFERENCE]
- 安全与多租户隔离：跨用户/跨实例 P2P 共享未讨论租户隔离与访问控制 [AGENT INFERENCE]
- 评估未覆盖长输出（>200 tokens）与高并发下 ITL 稳定性，真实 chatbot 长回复可能不同 [AGENT INFERENCE]
- 未与 KV 压缩（量化/稀疏/低秩）联合，传输量仍可进一步降 [AGENT INFERENCE]
- 成本模型缺失：远端 S3 Express 1 GBps 的费用 vs 自建 DRAM 的性价比未量化 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 如何实现带宽与长度感知的自适应加载 vs 重算决策器，并在 P95 SLO 下动态切换？阈值如何在线学习？ [AGENT INFERENCE]
2. 中心化 Controller 如何水平扩展为分片或分层结构以支撑千实例规模，同时保证 lookup 一致性？ [AGENT INFERENCE]
3. 能否将 LMCache 的 chunk 抽象与 CacheBlend 的 HKVD 选择性重算结合，实现跨存储 tier 的质量-带宽联合优化？ [AGENT INFERENCE]
4. 在 PD 解耦中，如何与连续批处理及前缀感知的调度器（RadixAttention、RAGCache）协同以最大化全局 goodput？ [AGENT INFERENCE]
5. 多后端（S3/NFS/Valkey 等）间的自动 tiering 与热点迁移策略如何设计，是否可借鉴 Mooncake 的热度复制？ [AGENT INFERENCE]
6. 如何在保持 Python 生态优势下，用 Rust/C++ 重写关键数据面以进一步逼近硬件极限，同时不牺牲社区迭代速度？ [AGENT INFERENCE]
7. P2P 共享的安全与计费：如何在企业多租户环境下实现细粒度授权与审计？ [AGENT INFERENCE]
8. 结合 KV 压缩（KIVI/GEAR/ShadowKV）后，端到端收益如何叠加，是否会改变 chunk 大小的最优选择？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **vLLM (Kwon et al. SOSP 23) PagedAttention / SGLang (Zheng et al. 23) RadixAttention：** 基座引擎与 paged/前缀树复用 [PAPER FACT]
- **RAGCache (Jin et al. 2404.12457) / CacheBlend (Yao et al. EuroSys 25)：** RAG 场景的层次化复用与选择性重算融合 [PAPER FACT]
- **Mooncake (Qin et al. 2407.00079) / AttentionStore / InfiniStore：** 分布式 KV 池与 RDMA/层次化存储 [PAPER FACT]
- **Splitwise (Patel et al. ISCA 24) / DistServe (Zhong et al. OSDI 24) / TetriInfer：** PD 解耦与调度 [PAPER FACT]
- **KIVI / GEAR / SnapKV / PyramidKV：** KV 压缩与驱逐，与 chunk 传输正交互补 [AGENT INFERENCE]
- **Dynamo / llm-d / AIBrix / vLLM Production Stack：** 基于 LMCache 的生产级分布式栈 [PAPER FACT]
- **NIXL / UCCL / RCCL：** 高性能 KV 传输引擎与带宽饱和分析 [PAPER FACT]
- **Strata (Xie et al. 2508.18572)：** 层次化上下文缓存对标 [AGENT INFERENCE]

## Review Log
Reviewer: Reviewer-1
Problems Found:
- Bandwidth saturation numbers (64KB 4Gbps etc.) correct per §3 Table1, but phrasing implied 8x400Gbps per-NIC; clarified as aggregate NIC bandwidth.
- Remote S3 Express 22-32% faster than full prefill is conditional on long context + 1GBps, not universal — original note overgeneralized; fixed qualifier.
- Main results verified: Single-node CPU offload TTFT 1.9-8.1x lower at QPS=1, throughput 2.3-14x across 5 models, up to 15x combined (Fig.8); Real-trace F/G TTFT 3.7-6.8x, ITL 19-58% (§6.2 Fig9/10); Central 15Gbps 1.3-3x (Fig11); PD NVLink TTFT 1.53-1.84x (1.5-1.8x abstract), ITL 1.12-1.66x (§6.3 Fig12-14); CPU load 400Gbps vs native 88Gbps correct per §5.
- Baselines correctly pinned to vLLM v0.10.2/v0.11.0 + commercial black-box + native PD NIXL per-page; commercial anonymity preserved correctly as #1/#2.
- Hardware: 8x H100 per node (GMI Cloud) + 500GB CPU DRAM limit + 15Gbps central / NVLink PD confirmed.
Corrections:
- Tightened Table1 bandwidth phrasing.
- Added conditional qualifier to 22-32% remote claim.
- No hallucination of KV compression synergy; correctly marked as not evaluated.
Confidence: High