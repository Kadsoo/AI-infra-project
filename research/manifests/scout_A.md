# LLM Inference / Serving 优化奠基性与代表性工作清单 — Scout A

> **Role:** Paper Scout A — Foundations & Serving Systems  
> **Focus:** serving architecture, memory management, scheduling, continuous batching, prefill/decode disaggregation, distributed serving  
> **切入系统:** vLLM / SGLang / LMCache  
> **日期:** 2026-08-27  
> **工作目录:** `F:\AIinfraResearch`  
> **检索方式:** `default.websearch` 多轮检索，已执行 13 次独立搜索，全部 URL 来自搜索结果可验证链接

## 简介

本清单系统梳理 LLM inference/serving 优化领域的奠基性与代表性工作，覆盖从单机内存管理到分布式集群调度的完整链路。核心关注：

- **Serving Architecture**：如何组织推理引擎、批处理与并行策略
- **Memory Management**：KV Cache 的分页、复用、卸载、压缩与分层存储
- **Scheduling**：请求调度、抢占、连续批处理、SLO 感知
- **Prefill/Decode Disaggregation**：两阶段解耦、跨节点 KV 迁移与独立扩缩容
- **Distributed Serving**：多机扩展、流水线/张量并行、容错

所有条目均通过联网检索验证，Paper URL 与 Code URL 均来自真实搜索结果（arXiv / USENIX / ACM / ICML / GitHub），未编造。

---

## 检索策略与验证

执行的 `default.websearch` 查询（共 13 次，满足 ≥8 次要求）：

1. `vLLM PagedAttention SOSP paper 2023`
2. `SGLang paper efficient execution structured generation arXiv`
3. `LMCache paper LLM KV cache arXiv`
4. `Orca continuous batching LLM serving SOSP OSDI paper`
5. `Splitwise DistServe prefill decode disaggregation paper arXiv`
6. `FastServe Sarathi Serve LLM scheduling paper arXiv OSDI`
7. `FastServe skip-join MLFQ scheduling LLM inference arXiv`
8. `TetriInfer DejaVu LLM serving disaggregation paper`
9. `DéjàVu KV cache streaming fault tolerant LLM serving paper`
10. `Mooncake KVCache centric disaggregated architecture LLM serving paper`
11. `Splitwise phase splitting LLM inference ISCA paper Patel`
12. `InfiniGen CacheGen FlexGen LLM memory management paper`
13. `FlexGen high throughput generative inference single GPU arXiv`

每篇论文记录完整字段：Title / Authors / Year / Venue / Paper URL / Code URL / Category / Why it matters / Type / Priority。

---

## 概览表（13 篇）

| # | Title (short) | Year | Venue | Category | Type | Priority |
|---|---|---|---|---|---|---|
| 1 | Orca: Distributed Serving System for Transformer-Based Generative Models | 2022 | OSDI'22 | serving systems / continuous batching / distributed | Foundational | High |
| 2 | vLLM: PagedAttention | 2023 | SOSP'23 | memory management / serving systems | Foundational | High |
| 3 | FlexGen: High-Throughput Generative Inference with a Single GPU | 2023 | ICML'23 | memory management / offloading | Representative | Medium |
| 4 | SGLang: Efficient Execution of Structured Language Model Programs | 2023/2024 | arXiv 2312.07104 → NeurIPS'24 | serving systems / memory reuse / structured generation | Representative | High |
| 5 | FastServe: Fast Distributed Inference Serving | 2023 | arXiv 2305.05920 | scheduling / distributed | Representative | Medium |
| 6 | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | 2023 | arXiv 2311.18677 → ISCA'24 | disaggregation / distributed | Representative | High |
| 7 | TetriInfer: Inference without Interference | 2024 | arXiv 2401.11181 | disaggregation / scheduling | Representative | Medium |
| 8 | DistServe: Disaggregating Prefill and Decoding | 2024 | OSDI'24 | disaggregation / distributed | Representative | High |
| 9 | DéjàVu: KV-cache Streaming | 2024 | arXiv 2403.01876 → ICML'24 | disaggregation / memory / fault-tolerance | Representative | Medium |
| 10 | Sarathi-Serve: Taming Throughput-Latency Tradeoff | 2024 | OSDI'24 | scheduling / continuous batching | Representative | High |
| 11 | InfiniGen: Dynamic KV Cache Management | 2024 | OSDI'24 | memory management | Representative | Medium |
| 12 | Mooncake: KVCache-centric Disaggregated Architecture | 2024 | arXiv 2407.00079 → FAST'25 | disaggregation / memory / distributed | Recent | High |
| 13 | LMCache: Efficient KV Cache Layer for Enterprise-Scale LLM Inference | 2025 | arXiv 2510.09665 | memory management / disaggregation / serving | Recent | High |

> 覆盖检查：vLLM ✓ | SGLang ✓ | LMCache ✓ | continuous batching ✓ (Orca, vLLM, Sarathi-Serve) | scheduling ✓ (FastServe, Sarathi-Serve) | disaggregation ✓ (Splitwise, DistServe, TetriInfer, DéjàVu, Mooncake) | memory ✓ (PagedAttention, InfiniGen, FlexGen, LMCache) | distributed ✓ (Orca, DistServe, FastServe, Mooncake)

---

## 详细条目

### 1. Orca: A Distributed Serving System for Transformer-Based Generative Models

- **Title:** Orca: A Distributed Serving System for Transformer-Based Generative Models
- **Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
- **Year:** 2022
- **Venue / arXiv:** OSDI 2022 (16th USENIX Symposium on Operating Systems Design and Implementation, pp. 521–538)
- **Paper URL:** https://www.usenix.org/conference/osdi22/presentation/yu
- **Paper URL (PDF):** https://www.usenix.org/system/files/osdi22-yu.pdf
- **Code URL:** N/A (无官方开源仓库，社区有非官方复现)
- **Category:** serving systems / continuous batching / distributed serving
- **Why it matters:** 首次提出 iteration-level scheduling（连续批处理 / continuous batching）与 selective batching，解决 static batching 的队头阻塞与 GPU 空转，在 GPT-3 175B 上相比 FasterTransformer 实现 36.9× 同延迟吞吐提升，成为后续所有 serving 系统的调度范式基石。
- **Type:** Foundational
- **Priority:** High

---

### 2. vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Year:** 2023
- **Venue / arXiv:** SOSP 2023 (ACM SIGOPS 29th Symposium on Operating Systems Principles, pp. 611–626) / arXiv:2309.06180
- **Paper URL:** https://arxiv.org/abs/2309.06180
- **Paper URL (ACM):** https://dl.acm.org/doi/10.1145/3600006.3613165
- **Code URL:** https://github.com/vllm-project/vllm
- **Category:** memory management / serving systems
- **Why it matters:** 将操作系统虚拟内存分页思想引入 KV Cache 管理，提出 PagedAttention 允许 KV 以非连续块存储，结合 copy-on-write 与 preemptive scheduling，实现近零碎片与高效共享，在多种模型与解码策略下获得 2–4× 吞吐提升，已成为事实上的开源推理引擎底座。
- **Type:** Foundational
- **Priority:** High

---

### 3. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

- **Title:** FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **Authors:** Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang
- **Year:** 2023
- **Venue / arXiv:** ICML 2023 (Proceedings of Machine Learning Research, vol. 202) / arXiv:2303.06865
- **Paper URL:** https://arxiv.org/abs/2303.06865
- **Code URL:** https://github.com/FMInference/FlexGen
- **Category:** memory management / offloading / serving systems
- **Why it matters:** 首个系统化研究单 GPU 高吞吐离线推理，通过聚合 GPU/CPU/磁盘资源并以线性规划搜索张量存放策略，结合 4-bit 压缩，首次在单 16GB GPU 上实现 OPT-175B 的 1 token/s 生成，奠定了 offloading 推理的技术路线，被 InfiniGen 等后续工作继承。
- **Type:** Representative
- **Priority:** Medium

---

### 4. SGLang: Efficient Execution of Structured Language Model Programs

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
- **Year:** 2023 (arXiv) / 2024 (NeurIPS)
- **Venue / arXiv:** arXiv:2312.07104 → NeurIPS 2024 (38th Conference on Neural Information Processing Systems, pp. 62557–62583)
- **Paper URL:** https://arxiv.org/abs/2312.07104
- **Code URL:** https://github.com/sgl-project/sglang
- **Category:** serving systems / memory management / structured generation
- **Why it matters:** 提出前端 DSL（primitives: gen/select/fork）+ 后端运行时协同设计，核心创新 RadixAttention 以 radix tree 自动复用跨多轮调用的 KV Cache，并用压缩有限状态机加速 JSON/正则约束解码，在 agent/RAG/多轮对话等复杂程序上实现 6.4× 吞吐提升，定义了现代 structured generation 引擎范式。
- **Type:** Representative
- **Priority:** High

---

### 5. FastServe: Fast Distributed Inference Serving for Large Language Models

- **Title:** Fast Distributed Inference Serving for Large Language Models
- **Authors:** Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xuanzhe Liu, Xin Jin
- **Year:** 2023
- **Venue / arXiv:** arXiv:2305.05920 (v3 2024-09-25)
- **Paper URL:** https://arxiv.org/abs/2305.05920
- **Code URL:** N/A（论文原型未官方开源，社区复现有 https://github.com/howwyhoward/FastServe 非官方）
- **Category:** scheduling / distributed serving / memory management
- **Why it matters:** 利用 LLM 自回归特性实现 token 粒度抢占，提出 skip-join Multi-Level Feedback Queue 调度器，利用输入长度已知（半信息不可知）跳过高优先队列以减少降级，结合 proactive KV cache 在 GPU/CPU 间换入换出并流水线隐藏延迟，相比 vLLM 平均延迟 SLO 下吞吐提升 31.4×、尾延迟 SLO 下 17.9×。
- **Type:** Representative
- **Priority:** Medium

---

### 6. Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- **Title:** Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini
- **Year:** 2023 (arXiv) / 2024 (ISCA)
- **Venue / arXiv:** arXiv:2311.18677 → ISCA 2024 (51st ACM/IEEE International Symposium on Computer Architecture, pp. 118–132)
- **Paper URL:** https://arxiv.org/abs/2311.18677
- **Code URL:** N/A
- **Category:** prefill/decode disaggregation / distributed serving
- **Why it matters:** 通过大规模刻画揭示 prompt compute（计算密集）与 token generation（访存密集）在延迟/吞吐/功耗上的本质差异，提出将两阶段拆分到异构机器池并通过 InfiniBand 高速 KV 传输解耦，设计面向吞吐/成本/功耗的异构集群，在相同成本下实现 1.4× 吞吐（20% 更低成本）或同功耗 2.35× 吞吐，是 disaggregation 路线的开创性系统表征与架构探索。
- **Type:** Representative
- **Priority:** High

---

### 7. TetriInfer: Inference without Interference — Disaggregate LLM Inference for Mixed Downstream Workloads

- **Title:** Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads
- **Authors:** Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan
- **Year:** 2024
- **Venue / arXiv:** arXiv:2401.11181
- **Paper URL:** https://arxiv.org/abs/2401.11181
- **Code URL:** N/A
- **Category:** prefill/decode disaggregation / scheduling
- **Why it matters:** 三支柱设计：固定尺寸 chunked prefill 使加速器始终接近饱和、prefill/decode 实例完全解耦、以及基于小模型长度预测的双层调度避免 decode 热点，在混合工作负载下 TTFT 降低 97%、JCT 降低 47%、每美元性能提升显著，提供了 chunked prefill 与预测调度的系统化干扰分析。
- **Type:** Representative
- **Priority:** Medium

---

### 8. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

- **Title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
- **Year:** 2024
- **Venue / arXiv:** OSDI 2024 / arXiv:2401.09670
- **Paper URL:** https://arxiv.org/abs/2401.09670
- **Code URL:** N/A（论文描述为 SwiftTransformer 后端，研究原型）
- **Category:** prefill/decode disaggregation / distributed serving / scheduling
- **Why it matters:** 首次形式化 goodput（满足 TTFT 与 TPOT SLO 的最大可服务速率）为优化目标，提出 prefill/decode 分池部署与独立并行策略协同优化，并给出基于带宽感知的放置算法与 CUDA IPC 的 KV 迁移，在严格 SLO 下相比 vLLM 实现 7.4× 请求率或 12.6× 更紧 SLO，成为解耦架构的理论与系统标杆。
- **Type:** Representative
- **Priority:** High

---

### 9. DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving

- **Title:** DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
- **Authors:** Foteini Strati, Sara Mcallister, Amar Phanishayee, Jakub Tarnawski, Ana Klimovic
- **Year:** 2024
- **Venue / arXiv:** arXiv:2403.01876 → ICML 2024 (PMLR vol. 235, pp. 46745–46771)
- **Paper URL:** https://arxiv.org/abs/2403.01876
- **Code URL:** https://github.com/msr-fiddle/dejavu
- **Category:** prefill/decode disaggregation / memory management / distributed serving / fault tolerance
- **Why it matters:** 设计通用 DéjàVuLib 流式库，统一实现 prompt-token 解耦减少流水线气泡、microbatch swapping 动态换入换出减少 GPU 显存超配、以及 token 级 KV 复制实现快速故障恢复，在云端多模型部署中吞吐提升 2× 且恢复时间大幅缩短，填补了 disaggregation 在容错与显存管理上的空白。
- **Type:** Representative
- **Priority:** Medium

---

### 10. Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

- **Title:** Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
- **Year:** 2024
- **Venue / arXiv:** OSDI 2024 / arXiv:2403.02310
- **Paper URL:** https://arxiv.org/abs/2403.02310
- **Paper URL (USENIX):** https://www.usenix.org/system/files/osdi24-agrawal.pdf
- **Code URL:** https://github.com/microsoft/sarathi-serve
- **Category:** scheduling / continuous batching / distributed serving
- **Why it matters:** 提出 chunked-prefills（将长 prompt 拆为等计算量块）与 stall-free 调度，使新请求可在不暂停正在解码请求的前提下加入批次，消除 prefills 对 decodes 的干扰并抹平流水线气泡，在单 A100 上对 Mistral-7B 提升 2.6× 容量、Falcon-180B 流水线并行下 5.6× 提升，现已被 vLLM/SGLang/TensorRT-LLM 采纳为默认策略。
- **Type:** Representative
- **Priority:** High

---

### 11. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management

- **Title:** InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **Authors:** Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim
- **Year:** 2024
- **Venue / arXiv:** OSDI 2024 / arXiv:2406.19707
- **Paper URL:** https://arxiv.org/abs/2406.19707
- **Code URL:** N/A
- **Category:** memory management / offloading
- **Why it matters:** 针对 offloading 推理中 CPU↔GPU 长上下文 KV 搬运瓶颈，提出基于“最小预演”（用当前层输入与下一层部分 Q/K 权重推测重要 token）的动态预取与瞬时剪枝，仅搬运关键 KV 条目，在 OPT 系列上相比 H2O/量化基线实现 3× 加速且精度损失降低 32.6 百分点，证明了注意力稀疏性在存储层次优化中的价值。
- **Type:** Representative
- **Priority:** Medium

---

### 12. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- **Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu 等
- **Year:** 2024 (arXiv 2407.00079) / 2025 (FAST 2025 Best Paper)
- **Venue / arXiv:** arXiv:2407.00079 → FAST 2025 / ACM Transactions on Storage 2026
- **Paper URL:** https://arxiv.org/abs/2407.00079
- **Code URL:** https://github.com/kvcache-ai/Mooncake
- **Category:** prefill/decode disaggregation / memory management / distributed serving
- **Why it matters:** Kimi 生产平台架构，将 prefill/decode 分池并进一步将 KVCache 作为一等公民构建跨 CPU/DRAM/SSD 的分布式缓存池，提出 Conductor 全局调度与基于预测的过载早期拒绝策略，在模拟长上下文场景下相比基线 525% 吞吐提升、真实负载下多处理 75% 请求，代表了工业级 KV 中心解耦的成熟形态。
- **Type:** Recent
- **Priority:** High

---

### 13. LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

- **Title:** LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- **Authors:** Yihua Cheng, Yuhan Liu, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Kuntai Du, Junchen Jiang
- **Year:** 2025
- **Venue / arXiv:** arXiv:2510.09665 (2025-10-08)
- **Paper URL:** https://arxiv.org/abs/2510.09665
- **Code URL:** https://github.com/LMCache/LMCache
- **Category:** memory management / prefill/decode disaggregation / serving systems
- **Why it matters:** 首个面向企业规模的开源 KV Cache 层，将 KV 作为引擎间标准化存储与通信媒介而非内部副产物，支持跨查询前缀复用与跨引擎 P/D 解耦传输，通过批量操作、计算/IO 重叠与零拷贝实现高吞吐，兼容 vLLM/SGLang 与 8 种存储后端（NFS/WEKA/S3 等）及多硬件平台，已成为 KV 复用的 de facto 标准。
- **Type:** Recent
- **Priority:** High

---

## 交叉对比与演进脉络

| 演进阶段 | 代表系统 | 核心突破 |
|---|---|---|
| **奠基 (2022–2023)** | Orca → vLLM | 连续批处理确立调度粒度；分页内存解决碎片与共享 |
| **扩展 (2023)** | FlexGen, FastServe | 单 GPU 高吞吐 offloading；抢占式调度缓解队头阻塞 |
| **解耦 (2023–2024)** | Splitwise → TetriInfer → DistServe → DéjàVu | 量化两阶段异构性并提出物理/逻辑解耦，减少干扰与气泡，引入 goodput 目标与容错 |
| **精细调度 (2024)** | Sarathi-Serve, InfiniGen | Chunked prefill 实现 stall-free；动态 KV 预取降低搬运 |
| **生产级 KV 中心 (2024–2025)** | Mooncake, SGLang, LMCache | RadixTree 前缀树复用、分布式 KV 池、标准化 KV 层，支撑长上下文与企业规模 |

---

## 建议阅读优先级

- **High (必读奠基/核心):** Orca, vLLM, SGLang, LMCache, Splitwise, DistServe, Mooncake, Sarathi-Serve
- **Medium (深度扩展):** FastServe, TetriInfer, DéjàVu, InfiniGen, FlexGen
- 建议顺序：Orca → vLLM → Sarathi-Serve / FastServe → Splitwise → DistServe → TetriInfer/DéjàVu → SGLang → Mooncake → LMCache → InfiniGen/FlexGen

---

## 局限与下一步

- 本清单聚焦 serving 系统与调度，量化/稀疏/推测解码等推理加速正交方向未深入
- 部分系统（如 DistServe/Splitwise/TetriInfer）论文原型未完全开源，复现需参考其模拟器与论文描述的 placement 算法
- 后续 Scout 可补充：CacheGen、PagedAttention 扩展（vLLM v2）、DeepSpeed-FastGen、Llumnix、LoongServe 等长上下文专题

---

*Scout A — Foundations & Serving Systems · 已满足停止条件：≥8 次 websearch，13 篇完整字段记录，文件已写入 `research/manifests/scout_A.md`*
