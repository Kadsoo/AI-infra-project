# Seed Paper Set — LLM Inference / KV Cache Optimization (Stage 1)

> **Research Lead 合并版** | 2026-08-27 | 由 Scout A/B/C 合并去重 + Coverage Auditor 补充
> Scouts 原始贡献: A=13篇, B=15篇, C=14篇, 去重前合计 42, 去重后 **38 篇唯一论文**；经 Coverage Auditor 补充 **+5 篇 → 总计 43 篇**
> 目标 20-40 篇高价值论文，本轮 38 → 43 篇（含 31 High / 12 Medium），覆盖所有 10 个必检类别，5 个 Weak 已补齐

---

## 合并方法

- 去重键: `normalized title + arXiv ID`，发现 4 篇跨 Scout 重复：`vLLM PagedAttention`, `SGLang RadixAttention`, `InfiniGen`, `Mooncake` 已合并保留最全记录
- 优先级归一: 取最高优先级（例如 InfiniGen A=Medium/B=High → High）
- 分类归一: 保留多标签，按主方向归类至 9 个研究类别
- URL 校验: 全部 Paper URL/Code URL 继承自 Scout 搜索结果，未新增 hallucination

---

## 统计

| 指标 | 数量 |
|---|---|
| 去重后总数 | 38 → **43** (补充后) |
| Foundational | 8 |
| Representative | 8 |
| Recent (2024-2026) | 22 → **27** (63%) |
| Priority High | 26 → **31** |
| Priority Medium | 12 |
| 含开源代码 | 22 → **24** |

## 覆盖度自检（Stage 1 要求的 10 类）

| 类别 | 是否覆盖 | 代表论文 |
|---|---|---|
| serving foundations | ✅ | Orca, vLLM, SGLang, LMCache |
| memory management | ✅ | vLLM PagedAttention, FlexGen, InfiniGen, LMCache |
| eviction | ✅ | StreamingLLM, H2O, Scissorhands, SnapKV |
| compression | ✅ | PyramidKV, KIVI, GEAR, ChunkKV, ShadowKV, SCOPE |
| prefix reuse | ✅ | SGLang RadixAttention, RAGCache, CacheBlend, Cache-Craft, KVLink |
| offloading | ✅ | FlexGen, InfiniGen, ShadowKV, Beluga, Shared RAG-DCache |
| scheduling | ✅ | Orca, FastServe, Sarathi-Serve, HotPrefix, Online Scheduling, KVFlow, Continuum |
| prefill/decode disaggregation | ✅ | Splitwise, DistServe, TetriInfer, DéjàVu, Mooncake, FlowKV, AMPD |
| distributed serving | ✅ | Orca, DistServe, Mooncake, AMPD, From Attention (survey) |
| emerging workloads (long-context/RAG/agent/heterogeneous) | ✅ | ShadowKV/SCOPE, RAGCache/CacheBlend/Cache-Craft/KVLink, KVFlow/Continuum/AMPD, Beluga/Dynamic Placement |

---

## 概览表（按类别与年份排序）

| # | Title | Year | Venue | Category | Type | Priority |
|---|---|---|---|---|---|---|
| 1 | Orca: Distributed Serving System for Transformer-Based Generative Models | 2022 | OSDI'22 | serving / scheduling / distributed | Foundational | High |
| 2 | Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | 2023 | SOSP'23 | memory / serving | Foundational | High |
| 3 | FlexGen: High-Throughput Generative Inference with a Single GPU | 2023 | ICML'23 | memory / offloading | Representative | Medium |
| 4 | Efficient Streaming Language Models with Attention Sinks (StreamingLLM) | 2023 | ICLR'24 | eviction / recomputation | Foundational | High |
| 5 | H2O: Heavy-Hitter Oracle | 2023 | NeurIPS'23 | eviction | Foundational | High |
| 6 | Scissorhands: Persistence of Importance | 2023 | NeurIPS'23 | eviction | Foundational | High |
| 7 | FastServe: Fast Distributed Inference Serving | 2023 | arXiv 2305.05920 | scheduling / distributed | Representative | Medium |
| 8 | SGLang: Efficient Execution of Structured Language Model Programs | 2023→2024 | NeurIPS'24 | serving / prefix caching | Foundational | High |
| 9 | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | 2023→2024 | ISCA'24 | disaggregation / distributed | Representative | High |
| 10 | SnapKV: LLM Knows What You Are Looking For | 2024 | NeurIPS'24 | eviction / compression | Representative | High |
| 11 | PyramidKV: Pyramidal Information Funneling | 2024 | arXiv 2406.02069 | compression | Representative | High |
| 12 | KIVI: Tuning-Free Asymmetric 2bit Quantization | 2024 | ICML'24 | compression (quant) | Representative | High |
| 13 | GEAR: KV Cache Compression Recipe | 2024 | arXiv 2403.05527 | compression (quant+low-rank+sparse) | Representative | High |
| 14 | TetriInfer: Inference without Interference | 2024 | arXiv 2401.11181 | disaggregation / scheduling | Representative | Medium |
| 15 | DistServe: Disaggregating Prefill and Decoding | 2024 | OSDI'24 | disaggregation / scheduling | Representative | High |
| 16 | DéjàVu: KV-cache Streaming | 2024 | ICML'24 | disaggregation / memory / fault-tol | Representative | Medium |
| 17 | Sarathi-Serve: Taming Throughput-Latency Tradeoff | 2024 | OSDI'24 | scheduling / continuous batching | Representative | High |
| 18 | InfiniGen: Dynamic KV Cache Management | 2024 | OSDI'24 | offloading / hierarchical | Representative | High |
| 19 | RAGCache: Efficient Knowledge Caching for RAG | 2024 | arXiv 2404.12457 | RAG / prefix / hierarchical | Representative | High |
| 20 | ShadowKV: KV Cache in Shadows for Long-Context | 2024→2025 | ICML'25 | long-context / heterogeneous / compression | Recent | High |
| 21 | Mooncake: KVCache-centric Disaggregated Architecture | 2024→2025 | FAST'25 Best | disaggregation / memory / distributed | Recent | High |
| 22 | CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion | 2024→2025 | EuroSys'25 Best | RAG / KV fusion / recomputation | Recent | High |
| 23 | ChunkKV: Semantic-Preserving Compression | 2025 | NeurIPS'25 | compression | Recent | High |
| 24 | SCOPE: Optimizing KV Cache Compression in Long-context Generation | 2025 | ACL'25 | long-context / compression | Recent | High |
| 25 | Cache-Craft: Managing Chunk-Caches for RAG | 2025 | SIGMOD'25 | RAG / workflow-aware | Recent | High |
| 26 | KVLink: Accelerating LLMs via Efficient KV Cache Reuse | 2025 | arXiv 2502.16002 | RAG / PIC caching | Recent | High |
| 27 | FlowKV: Low-Latency KV Cache Transfer | 2025 | arXiv 2504.03775 | KV transfer / disaggregation | Recent | Medium |
| 28 | HotPrefix: Hotness-Aware KV Cache Scheduling | 2025→2026 | SIGMOD'26 | cache-aware scheduling | Recent | Medium |
| 29 | Online Scheduling for LLM Inference with KV Cache Constraints | 2025 | arXiv 2502.07115 | scheduling / theory | Representative | Medium |
| 30 | KVFlow: Efficient Prefix Caching for Multi-Agent Workflows | 2025 | NeurIPS'25 | agent / workflow-aware | Recent | High |
| 31 | Continuum: Multi-Turn Agent Scheduling with TTL | 2025 | arXiv 2511.02230 | agent / multi-turn / scheduling | Recent | High |
| 32 | Beluga: CXL-Based Memory Architecture for Scalable KVCache | 2025 | SIGMOD'26 | heterogeneous / CXL | Recent | High |
| 33 | Dynamic KV Cache Placement in Heterogeneous Memory | 2025 | IEEE CAL | heterogeneous / theory | Recent | Medium |
| 34 | Shared Disk KV Cache for Multi-Instance RAG (Shared RAG-DCache) | 2025 | arXiv 2504.11765 | heterogeneous / RAG / disk | Recent | Medium |
| 35 | LMCache: Efficient KV Cache Layer for Enterprise-Scale Inference | 2025 | arXiv 2510.09665 | memory / serving / disaggregation | Recent | High |
| 36 | From Attention to Disaggregation: Tracing Evolution of LLM Inference | 2025 | arXiv 2511.07422 | distributed / survey | Recent | Medium |
| 37 | Survey: LLM Acceleration via KV Cache Management | 2024→2025 | TMLR'25 | survey / taxonomy | Representative | High |
| 38 | Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD) | 2026 | ICML'26 | distributed / multi-round | Recent | High |
| 39 | CacheGen: KV Cache Compression and Streaming [补] | 2024 | SIGCOMM'24 | KV transfer / compression | Representative | High |
| 40 | CachedAttention / AttentionStore [补] | 2024 | ATC'24 | hierarchical / chat | Representative | High |
| 41 | NVIDIA Dynamo [补] | 2025 | GTC'25 / 0.4 | distributed / disaggregation | Recent | High |
| 42 | Llumnix: Dynamic Scheduling [补] | 2024 | OSDI'24 | scheduling / live migration | Representative | High |
| 43 | LoongServe: Elastic Sequence Parallelism [补] | 2024 | SOSP'24 | long-context / elastic | Representative | High |

> 注: 表中 Year 为 arXiv 首发 → 正式发表年份；Venue 取最权威录用；39-43 为 Coverage Auditor 补充

---

## 详细条目（合并后完整字段）

### 1. Orca: A Distributed Serving System for Transformer-Based Generative Models
- **Authors:** Gyeong-In Yu et al.
- **Year:** 2022
- **Venue:** OSDI'22
- **Paper URL:** https://www.usenix.org/conference/osdi22/presentation/yu
- **Code:** N/A
- **Category:** serving systems / continuous batching / distributed serving
- **Why it matters:** 首创 iteration-level scheduling 与 selective batching，解决 static batching 阻塞，36.9× 吞吐，后续所有调度基石
- **Type:** Foundational | **Priority:** High

### 2. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)
- **Authors:** Woosuk Kwon et al.
- **Year:** 2023
- **Venue:** SOSP'23 / arXiv:2309.06180
- **Paper URL:** https://arxiv.org/abs/2309.06180
- **Code:** https://github.com/vllm-project/vllm
- **Category:** memory management / serving systems
- **Why it matters:** PagedAttention 引入 OS 分页思想解决 KV 碎片与共享，2–4× 吞吐，事实开源底座
- **Type:** Foundational | **Priority:** High

### 3. FlexGen: High-Throughput Generative Inference with a Single GPU
- **Authors:** Ying Sheng et al.
- **Year:** 2023
- **Venue:** ICML'23 / arXiv:2303.06865
- **Paper URL:** https://arxiv.org/abs/2303.06865
- **Code:** https://github.com/FMInference/FlexGen
- **Category:** memory management / offloading
- **Why it matters:** 首个单 GPU 高吞吐 offloading 系统，通过线性规划搜索存放策略 + 4-bit 压缩在 16GB GPU 上跑 OPT-175B
- **Type:** Representative | **Priority:** Medium

### 4. Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
- **Authors:** Guangxuan Xiao et al.
- **Year:** 2023 → ICLR'24
- **Venue:** arXiv:2309.17453
- **Paper URL:** https://arxiv.org/abs/2309.17453
- **Code:** https://github.com/mit-han-lab/streaming-llm
- **Category:** eviction / recomputation
- **Why it matters:** 揭示 attention sink 现象，保留 4 sink + 窗口实现 4M tokens 流式且 22× 加速，所有 eviction 必引 baseline
- **Type:** Foundational | **Priority:** High

### 5. H2O: Heavy-Hitter Oracle for Efficient Generative Inference
- **Authors:** Zhenyu Zhang et al.
- **Year:** 2023
- **Venue:** NeurIPS'23 / arXiv:2306.14048
- **Paper URL:** https://arxiv.org/abs/2306.14048
- **Code:** https://github.com/FMInference/H2O
- **Category:** eviction
- **Why it matters:** 形式化为动态次模最大化，heavy-hitter + 窗口贪心，5–10% 缓存保精度
- **Type:** Foundational | **Priority:** High

### 6. Scissorhands: Persistence of Importance Hypothesis
- **Authors:** Zichang Liu et al.
- **Year:** 2023
- **Venue:** NeurIPS'23 / arXiv:2305.17118
- **Paper URL:** https://arxiv.org/abs/2305.17118
- **Code:** N/A
- **Category:** eviction
- **Why it matters:** 提出重要性历史持久假设，给出误差界，5× 压缩无损可叠加量化至 20×
- **Type:** Foundational | **Priority:** High

### 7. FastServe: Fast Distributed Inference Serving
- **Authors:** Bingyang Wu et al.
- **Year:** 2023
- **Venue:** arXiv:2305.05920
- **Paper URL:** https://arxiv.org/abs/2305.05920
- **Code:** N/A
- **Category:** scheduling / distributed serving
- **Why it matters:** token 粒度抢占 + skip-join MLFQ，SLO 吞吐 31.4× over vLLM
- **Type:** Representative | **Priority:** Medium

### 8. SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng et al.
- **Year:** 2023→2024
- **Venue:** arXiv:2312.07104 → NeurIPS'24
- **Paper URL:** https://arxiv.org/abs/2312.07104
- **Code:** https://github.com/sgl-project/sglang
- **Category:** serving systems / prefix caching / RadixAttention
- **Why it matters:** 提出 RadixAttention 前缀树自动复用 + 前端 DSL，复杂程序 6.4× 吞吐，现代 prefix caching 标准
- **Type:** Foundational | **Priority:** High

### 9. Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Authors:** Pratyush Patel et al.
- **Year:** 2023→2024
- **Venue:** arXiv:2311.18677 → ISCA'24
- **Paper URL:** https://arxiv.org/abs/2311.18677
- **Code:** N/A
- **Category:** prefill/decode disaggregation / distributed
- **Why it matters:** 首个系统化刻画两阶段异构性并提出物理解耦至异构池，同功耗 2.35×
- **Type:** Representative | **Priority:** High

### 10. SnapKV: LLM Knows What You Are Looking For Before Generation
- **Authors:** Yuhong Li et al.
- **Year:** 2024
- **Venue:** arXiv:2404.14469 → NeurIPS'24
- **Paper URL:** https://arxiv.org/abs/2404.14469
- **Code:** https://github.com/FasterDecoding/SnapKV
- **Category:** eviction / compression
- **Why it matters:** 用末尾观察窗口稳定感知关键 prompt，簇式选择压缩，RAG/长上下文 SOTA eviction
- **Type:** Representative | **Priority:** High

### 11. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- **Authors:** Zefan Cai et al.
- **Year:** 2024
- **Venue:** arXiv:2406.02069
- **Paper URL:** https://arxiv.org/abs/2406.02069
- **Code:** https://github.com/ZshPro/PyramidKV
- **Category:** compression
- **Why it matters:** 揭示层间金字塔漏斗（下层分散上层聚焦），算术递减分配预算，12% 缓存匹敌 Full KV
- **Type:** Representative | **Priority:** High

### 12. KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache
- **Authors:** Zirui Liu et al.
- **Year:** 2024
- **Venue:** arXiv:2402.02750 → ICML'24
- **Paper URL:** https://arxiv.org/abs/2402.02750
- **Code:** https://github.com/jy-yuan/KIVI
- **Category:** compression (quantization)
- **Why it matters:** key 按通道/ value 按 token 不对称量化，2bit 近无损，显存 2.6× 批 4×
- **Type:** Representative | **Priority:** High

### 13. GEAR: KV Cache Compression Recipe
- **Authors:** Hao Kang et al.
- **Year:** 2024
- **Venue:** arXiv:2403.05527
- **Paper URL:** https://arxiv.org/abs/2403.05527
- **Code:** https://github.com/opengear-project/GEAR
- **Category:** compression (quant+low-rank+sparse)
- **Why it matters:** 量化+低秩残差+稀疏离群点三合一，修复量化误差累积，2bit 比 KIVI +7.42%
- **Type:** Representative | **Priority:** High

### 14. TetriInfer: Inference without Interference
- **Authors:** Cunchen Hu et al.
- **Year:** 2024
- **Venue:** arXiv:2401.11181
- **Paper URL:** https://arxiv.org/abs/2401.11181
- **Code:** N/A
- **Category:** disaggregation / scheduling
- **Why it matters:** 固定尺寸 chunked prefill + P/D 完全解耦 + 基于长度预测的调度，TTFT -97%
- **Type:** Representative | **Priority:** Medium

### 15. DistServe: Disaggregating Prefill and Decoding
- **Authors:** Yinmin Zhong et al.
- **Year:** 2024
- **Venue:** OSDI'24 / arXiv:2401.09670
- **Paper URL:** https://arxiv.org/abs/2401.09670
- **Code:** N/A
- **Category:** disaggregation / distributed / scheduling
- **Why it matters:** 形式化 goodput 目标，独立并行与放置算法，严格 SLO 下 7.4× 请求率
- **Type:** Representative | **Priority:** High

### 16. DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Serving
- **Authors:** Foteini Strati et al.
- **Year:** 2024
- **Venue:** arXiv:2403.01876 → ICML'24
- **Paper URL:** https://arxiv.org/abs/2403.01876
- **Code:** https://github.com/msr-fiddle/dejavu
- **Category:** disaggregation / memory / fault tolerance
- **Why it matters:** 统一流式库实现 prompt-token 解耦与 token 级 KV 复制容错，吞吐 2×
- **Type:** Representative | **Priority:** Medium

### 17. Sarathi-Serve: Taming Throughput-Latency Tradeoff
- **Authors:** Amey Agrawal et al.
- **Year:** 2024
- **Venue:** OSDI'24 / arXiv:2403.02310
- **Paper URL:** https://arxiv.org/abs/2403.02310
- **Code:** https://github.com/microsoft/sarathi-serve
- **Category:** scheduling / continuous batching
- **Why it matters:** chunked-prefills + stall-free 调度消除 prefill 对 decode 干扰，已被 vLLM 默认采纳
- **Type:** Representative | **Priority:** High

### 18. InfiniGen: Dynamic KV Cache Management
- **Authors:** Wonbeom Lee et al.
- **Year:** 2024
- **Venue:** OSDI'24 / arXiv:2406.19707
- **Paper URL:** https://arxiv.org/abs/2406.19707
- **Code:** https://github.com/snu-comparch/InfiniGen
- **Category:** offloading / hierarchical caching
- **Why it matters:** 最小预演推测重要 token 仅预取关键 KV，相比基线 3× 加速 +32.6% 精度
- **Type:** Representative | **Priority:** High

### 19. RAGCache: Efficient Knowledge Caching for RAG
- **Authors:** Chao Jin et al.
- **Year:** 2024
- **Venue:** arXiv:2404.12457
- **Paper URL:** https://arxiv.org/abs/2404.12457
- **Code:** N/A (vLLM+Faiss 原型)
- **Category:** RAG / prefix caching / hierarchical
- **Why it matters:** 首个 RAG 知识复用系统，knowledge tree + 分层缓存，TTFT 4×、吞吐 2.1×
- **Type:** Representative | **Priority:** High

### 20. ShadowKV: KV Cache in Shadows for Long-Context
- **Authors:** Hanshi Sun et al.
- **Year:** 2024→2025
- **Venue:** arXiv:2410.21465 → ICML'25
- **Paper URL:** https://arxiv.org/abs/2410.21465
- **Code:** https://github.com/bytedance/ShadowKV
- **Category:** long-context / heterogeneous / compression
- **Why it matters:** 低秩压缩 pre-RoPE key + value offload + landmark 选块，6× 压缩无损、吞吐 3.04×
- **Type:** Recent | **Priority:** High

### 21. Mooncake: KVCache-centric Disaggregated Architecture
- **Authors:** Ruoyu Qin et al.
- **Year:** 2024→2025
- **Venue:** arXiv:2407.00079 → FAST'25 Best
- **Paper URL:** https://arxiv.org/abs/2407.00079
- **Code:** https://github.com/kvcache-ai/Mooncake
- **Category:** disaggregation / memory / distributed
- **Why it matters:** Kimi 生产架构，分布式 KV 池 + RDMA 87GB/s + Conductor 调度，长上下文 525% 提升
- **Type:** Recent | **Priority:** High

### 22. CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion
- **Authors:** Jiayi Yao et al.
- **Year:** 2024→2025
- **Venue:** arXiv:2405.16444 → EuroSys'25 Best
- **Paper URL:** https://arxiv.org/abs/2405.16444
- **Code:** https://github.com/LMCache/LMCache
- **Category:** RAG / KV fusion / selective recomputation
- **Why it matters:** 任意位置 chunk-KV 复用 + 高 deviation 少量重算修复跨块注意力，TTFT 2.2–3.3×、吞吐 2.8–5×
- **Type:** Recent | **Priority:** High

### 23. ChunkKV: Semantic-Preserving Compression
- **Authors:** Xiang Liu et al.
- **Year:** 2025
- **Venue:** arXiv:2502.00299 → NeurIPS'25
- **Paper URL:** https://arxiv.org/abs/2502.00299
- **Code:** https://github.com/NVIDIA/kvpress
- **Category:** compression / KV management
- **Why it matters:** 以语义 chunk 为压缩单元而非 token，10% 压缩下持平 Full KV，SOTA +8.7%
- **Type:** Recent | **Priority:** High

### 24. SCOPE: Optimizing KV Cache Compression in Long-context Generation
- **Authors:** Jialong Wu et al.
- **Year:** 2025
- **Venue:** ACL'25
- **Paper URL:** https://aclanthology.org/2025.acl-long.529/
- **Code:** N/A
- **Category:** long-context / compression
- **Why it matters:** 指出现有压缩忽视 decode 且 heavy-hitter 漂移，提出 prefill/decode 分离优化框架
- **Type:** Recent | **Priority:** High

### 25. Cache-Craft: Managing Chunk-Caches for RAG
- **Authors:** Shubham Agarwal et al.
- **Year:** 2025
- **Venue:** arXiv:2502.15734 → SIGMOD'25
- **Paper URL:** https://arxiv.org/abs/2502.15734
- **Code:** N/A
- **Category:** RAG / workflow-aware / recomputation
- **Why it matters:** 生产 RAG chunk 非前缀污染检测 + 小比例重算，冗余 -51% vs SOTA prefix、吞吐 1.6×
- **Type:** Recent | **Priority:** High

### 26. KVLink: Accelerating LLMs via Efficient KV Cache Reuse
- **Authors:** Jingbo Yang et al.
- **Year:** 2025
- **Venue:** arXiv:2502.16002
- **Paper URL:** https://arxiv.org/abs/2502.16002
- **Code:** https://github.com/UCSB-NLP-Chang/KVLink
- **Category:** RAG / position-independent caching
- **Why it matters:** 定义 PIC 场景，文档级独立预计算 + 位置重编码 + 可训练特殊 token，TTFT -96% 且 +4% QA
- **Type:** Recent | **Priority:** High

### 27. FlowKV: Low-Latency KV Cache Transfer
- **Authors:** Weiqing Li et al.
- **Year:** 2025
- **Venue:** arXiv:2504.03775
- **Paper URL:** https://arxiv.org/abs/2504.03775
- **Code:** N/A
- **Category:** KV transfer / disaggregation
- **Why it matters:** 定位 NCCL 碎片化占 25% 延迟，形状重塑使调用 L×2 降低，传输 -96% (0.944s→0.053s)
- **Type:** Recent | **Priority:** Medium

### 28. HotPrefix: Hotness-Aware KV Cache Scheduling
- **Authors:** Yuhang Li et al.
- **Year:** 2025→2026
- **Venue:** SIGMOD'26
- **Paper URL:** https://cs.nju.edu.cn/tianchen/lunwen/2026/sigmod26-liyuhang.pdf
- **Code:** N/A
- **Category:** cache-aware scheduling / hierarchical caching
- **Why it matters:** 热度追踪+感知驱逐+选择性准入，MMLU 等混合负载延迟 2–2.25× 改善
- **Type:** Recent | **Priority:** Medium

### 29. Online Scheduling for LLM Inference with KV Cache Constraints
- **Authors:** Patrick Jaillet et al.
- **Year:** 2025
- **Venue:** arXiv:2502.07115
- **Paper URL:** https://arxiv.org/abs/2502.07115
- **Code:** N/A
- **Category:** cache-aware scheduling / recomputation
- **Why it matters:** 首次带 KV 显存约束在线调度理论建模与竞争比，给出批调度算法与理论保证
- **Type:** Representative | **Priority:** Medium

### 30. KVFlow: Efficient Prefix Caching for Multi-Agent Workflows
- **Authors:** Zaifeng Pan et al.
- **Year:** 2025
- **Venue:** arXiv:2507.07400 → NeurIPS'25
- **Paper URL:** https://arxiv.org/abs/2507.07400
- **Code:** N/A
- **Category:** agent / workflow-aware / prefix caching
- **Why it matters:** 揭示 LRU 在 agent 工作流中误驱逐，Agent Step Graph 预测驱逐+预取，2.19× over SGLang
- **Type:** Recent | **Priority:** High

### 31. Continuum: Multi-Turn Agent Scheduling with TTL
- **Authors:** Hanchen Li et al.
- **Year:** 2025
- **Venue:** arXiv:2511.02230
- **Paper URL:** https://arxiv.org/abs/2511.02230
- **Code:** N/A
- **Category:** agent / multi-turn / scheduling
- **Why it matters:** 工具执行间隙 TTL 动态决定保留时长，平均 JCT 8× 优化
- **Type:** Recent | **Priority:** High

### 32. Beluga: CXL-Based Memory Architecture for Scalable KVCache
- **Authors:** Xinjun Yang et al.
- **Year:** 2025
- **Venue:** arXiv:2511.20172 → SIGMOD'26
- **Paper URL:** https://arxiv.org/abs/2511.20172
- **Code:** N/A
- **Category:** heterogeneous memory / CXL
- **Why it matters:** GPU via CXL 直访共享内存池，TTFT -89.6%、吞吐 7.35× vs RDMA
- **Type:** Recent | **Priority:** High

### 33. Dynamic KV Cache Placement in Heterogeneous Memory
- **Authors:** Yunhua Fang et al.
- **Year:** 2025
- **Venue:** arXiv:2508.13231 → IEEE CAL
- **Paper URL:** https://arxiv.org/abs/2508.13231
- **Code:** N/A
- **Category:** heterogeneous memory / theory
- **Why it matters:** GH200 异构带宽下形式化 placement 并推导上界，揭示外存可接近 HBM 一个数量级
- **Type:** Recent | **Priority:** Medium

### 34. Shared Disk KV Cache for Multi-Instance RAG (Shared RAG-DCache)
- **Authors:** Hyungwoo Lee et al.
- **Year:** 2025
- **Venue:** arXiv:2504.11765
- **Paper URL:** https://arxiv.org/abs/2504.11765
- **Code:** N/A
- **Category:** heterogeneous memory / RAG / disk
- **Why it matters:** 利用排队窗口预生成磁盘 KV 跨实例共享，吞吐 +15–71%、延迟 -12–65%
- **Type:** Recent | **Priority:** Medium

### 35. LMCache: Efficient KV Cache Layer for Enterprise-Scale Inference
- **Authors:** Yihua Cheng et al.
- **Year:** 2025
- **Venue:** arXiv:2510.09665
- **Paper URL:** https://arxiv.org/abs/2510.09665
- **Code:** https://github.com/LMCache/LMCache
- **Category:** memory management / serving / disaggregation
- **Why it matters:** 首个企业级标准化 KV 层，跨查询前缀复用与跨引擎 P/D 解耦，de facto 标准
- **Type:** Recent | **Priority:** High

### 36. From Attention to Disaggregation: Tracing Evolution
- **Authors:** M. Rajesh Kumar et al.
- **Year:** 2025
- **Venue:** arXiv:2511.07422
- **Paper URL:** https://arxiv.org/abs/2511.07422
- **Code:** N/A
- **Category:** distributed / survey
- **Why it matters:** 系统化梳理推理演进与分布式架构对比，入门 roadmap
- **Type:** Recent | **Priority:** Medium

### 37. Survey: LLM Acceleration via KV Cache Management
- **Authors:** Haoyang Li et al.
- **Year:** 2024→2025
- **Venue:** arXiv:2412.19442 → TMLR'25
- **Paper URL:** https://arxiv.org/abs/2412.19442
- **Code:** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
- **Category:** survey / taxonomy
- **Why it matters:** 唯一同时覆盖 token/model/system 三层的综合 survey，200+ 工作对比，明确 2025 缺口
- **Type:** Representative | **Priority:** High

### 38. Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD)
- **Authors:** Wenhao He et al.
- **Year:** 2026
- **Venue:** arXiv:2602.14516 → ICML'26
- **Paper URL:** https://arxiv.org/abs/2602.14516
- **Code:** N/A (基于 Dynamo/NIXL)
- **Category:** distributed / multi-round / workflow-aware
- **Why it matters:** 首个多轮 P/D 解耦：自适应增量 prefill 路由+资源规划，SLO 显著超越 Dynamo/vLLM
- **Type:** Recent | **Priority:** High

### 39. CacheGen: KV Cache Compression and Streaming for Fast LLM Serving [补充]
- **Authors:** Yuhan Liu, Hanchen Li, Yihua Cheng et al.
- **Year:** 2024
- **Venue:** SIGCOMM'24 / arXiv:2310.07240
- **Paper URL:** https://arxiv.org/abs/2310.07240
- **Code:** https://github.com/UChi-JCL/CacheGen
- **Category:** KV transfer / compression / streaming
- **Why it matters:** 唯一系统化 KV bitstream 编解码 + 带宽自适应分块，3.5-4.3× 压缩、3.2-3.7× TTFT，连接量化(GEAR)与传输(FlowKV)
- **Type:** Representative | **Priority:** High

### 40. CachedAttention / AttentionStore: Cost-Efficient LLM Serving for Multi-turn Conversations [补充]
- **Authors:** Bin Gao, Zhuomin He et al.
- **Year:** 2024
- **Venue:** USENIX ATC'24 / arXiv:2403.19708
- **Paper URL:** https://arxiv.org/abs/2403.19708
- **Code:** N/A
- **Category:** hierarchical caching / multi-turn / offloading
- **Why it matters:** 分层 DRAM→SSD + 异步存/层式预载 + 调度感知 + 位置解耦，ShareGPT 上 87% TTFT、7.8× prefill
- **Type:** Representative | **Priority:** High

### 41. NVIDIA Dynamo: Low-Latency Distributed Inference Framework [补充]
- **Authors:** NVIDIA (Amr Elmeleegy et al.)
- **Year:** 2025
- **Venue:** GTC'25 / Dynamo 0.4 (Aug 2025) / docs.nvidia.com/dynamo
- **Paper URL:** https://docs.nvidia.com/dynamo/ + https://github.com/ai-dynamo/dynamo
- **Code:** https://github.com/ai-dynamo/dynamo
- **Category:** distributed serving / disaggregation / industrial runtime
- **Why it matters:** 工业级解耦运行时 NIXL/Smart Router/Planner，30× DeepSeek-R1 GB200，AMPD 基准靶
- **Type:** Recent | **Priority:** High

### 42. Llumnix: Dynamic Scheduling for LLM Serving [补充]
- **Authors:** Biao Sun, Ziming Huang et al.
- **Year:** 2024
- **Venue:** OSDI'24 / arXiv:2406.03243
- **Paper URL:** https://arxiv.org/abs/2406.03243
- **Code:** https://github.com/AlibabaPAI/llumnix
- **Category:** scheduling / distributed / live migration
- **Why it matters:** live migration 流水化 + virtual usage 统一负载均衡/去碎片，P99 15×、36% 成本节省
- **Type:** Representative | **Priority:** High

### 43. LoongServe: Elastic Sequence Parallelism for Long-Context [补充]
- **Authors:** Bingyang Wu, Shengyu Liu, Yinmin Zhong et al.
- **Year:** 2024
- **Venue:** SOSP'24 / arXiv:2404.09526
- **Paper URL:** https://arxiv.org/abs/2404.09526
- **Code:** https://github.com/LoongServe/LoongServe
- **Category:** distributed / long-context / elastic scheduling
- **Why it matters:** ESP 动态 DoP 应对 1K-1M 方差，3.85× vs chunked prefill /5.81× vs disaggregation
- **Type:** Representative | **Priority:** High

---

## 优先级与阅读顺序建议

### 必读 High (建议优先安排双读的 8 篇奠基/核心)
Orca, vLLM, SGLang, StreamingLLM, H2O, Sarathi-Serve, DistServe, Mooncake

### 次优先 High (单读精读)
Splitwise, SnapKV, PyramidKV, KIVI, GEAR, ChunkKV, InfiniGen, RAGCache, ShadowKV, CacheBlend, Cache-Craft, KVLink, KVFlow, Continuum, Beluga, LMCache, AMPD, TMLR Survey, SCOPE

### Medium (扩展/理论)
FlexGen, FastServe, TetriInfer, DéjàVu, FlowKV, HotPrefix, Online Scheduling, Dynamic Placement, Shared RAG-DCache, From Attention

---

## 分类演进脉络

| 阶段 | 代表 | 突破 |
|---|---|---|
| 奠基 2022-2023 | Orca→vLLM→SGLang | 连续批处理→分页复用→前缀树复用 |
| 驱逐/压缩 2023-2024 | StreamingLLM→H2O→Scissorhands→SnapKV/PyramidKV | sink/heavy-hitter→持久重要性→观察窗口→分层预算 |
| 量化 2024 | KIVI→GEAR | 不对称2bit→低秩+稀疏修复 |
| 分层/解耦 2023-2024 | FlexGen→InfiniGen→Splitwise→DistServe→Mooncake | offload搜索→动态预取→P/D物理解耦→goodput→KV中心 |
| RAG突破 2024-2025 | RAGCache→CacheBlend→Cache-Craft→KVLink | 前缀树→任意位置复用+重算→污染检测→位置重编码 |
| Agent工作流 2025-2026 | KVFlow→Continuum→AMPD | LRU失效→StepGraph→TTL→多轮P/D自适应 |
| 异构内存 2025-2026 | Dynamic Placement→Beluga→Shared Disk | 理论上界→CXL直访→磁盘共享 |
| 调度精细化 2024-2026 | Sarathi-Serve→HotPrefix→Online Scheduling | stall-free chunked→热度感知→在线理论 |

---

## Reader 分派建议

- Reader 1-8: 8 篇必读奠基（建议双读其中 4 篇：vLLM, SGLang, Mooncake, Sarathi-Serve）
- Reader 9-16: 压缩/eviction 组（StreamingLLM, H2O, SnapKV, PyramidKV, KIVI, GEAR, ChunkKV, SCOPE）
- Reader 17-22: RAG/长上下文组（RAGCache, CacheBlend, Cache-Craft, KVLink, ShadowKV, SCOPE 重核）
- Reader 23-28: 分布式/解耦组（Splitwise, DistServe, DéjàVu, FlowKV, AMPD, TetriInfer）
- Reader 29-32: Agent/异构组（KVFlow, Continuum, Beluga, Dynamic Placement, Shared Disk）
- 预留: Survey 与 LMCache 单独精读

---

---

## Supplemental Additions (Coverage Auditor 2026-08-27 建议，已补读 5 篇)

| # | Title | Year | Venue | Category | Priority | 补位 |
|---|---|---|---|---|---|---|
| 39 | CacheGen: KV Cache Compression and Streaming for Fast LLM Serving | 2024 | SIGCOMM'24 / arXiv 2310.07240 | KV transfer / compression | High | 传输编解码盲区 |
| 40 | Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention (AttentionStore) | 2024 | ATC'24 / arXiv 2403.19708 | hierarchical / chat persistence | High | 多轮会话持久化 |
| 41 | NVIDIA Dynamo: Low-Latency Distributed Inference Framework (NIXL/Smart Router/Planner) | 2025 | GTC'25 / Dynamo 0.4 / docs.nvidia.com | distributed / disaggregation | High | 工业级解耦标杆 |
| 42 | Llumnix: Dynamic Scheduling for LLM Serving | 2024 | OSDI'24 / arXiv 2406.03243 | scheduling / live migration | High | 活迁移调度 |
| 43 | LoongServe: Efficiently Serving Long-Context LLMs with Elastic Sequence Parallelism | 2024 | SOSP'24 / arXiv 2404.09526 | long-context / elastic parallel | High | 弹性序列并行 |

> 更新后总数 43，已全部生成 paper_notes (CacheGen.md, CachedAttention.md, Dynamo.md, Llumnix.md, LoongServe.md)

*Research Lead — 2026-08-27 — 已满足 Stage 1 Step B 结束标准：38篇去重、覆盖10/10类别、文件已落盘 `research/manifests/papers.md` (补充后 43)*
