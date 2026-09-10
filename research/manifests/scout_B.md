# KV Cache 优化工作清单 — Scout B

> **Role:** Paper Scout B — KV Cache Optimization  
> **Focus:** KV Cache eviction / compression / prefix caching / KV reuse / offloading / hierarchical caching / KV transfer / recomputation / cache-aware scheduling  
> **切入系统:** vLLM PagedAttention / SGLang RadixAttention / InfiniGen / Mooncake  
> **日期:** 2026-08-27  
> **工作目录:** `F:\AIinfraResearch`  
> **检索方式:** `default.websearch` 多轮检索，已执行 12 次独立搜索，全部 URL 来自搜索结果可验证链接

## 简介

本清单系统梳理 LLM 推理中 KV Cache 优化的代表性工作，覆盖从单请求显存压缩到跨请求/跨节点复用与调度的完整链路。核心关注：

- **Eviction（驱逐）**：基于注意力稀疏性选择性丢弃非关键 token
- **Compression（压缩）**：量化、低秩、稀疏等近无损压缩
- **Prefix Caching / KV Reuse（前缀复用）**：跨请求/跨轮次共享前缀 KV，避免重复 prefill
- **Offloading / Hierarchical Caching（卸载与分层）**：GPU ↔ CPU ↔ SSD 协同，动态预取
- **KV Transfer / Disaggregated Inference（KV 迁移与解耦）**：prefill-decode 分离下的高效 KV 传输
- **Recomputation（重计算权衡）**：eviction 与重算的折衷、注意力汇聚点 (attention sink) 保护
- **Cache-aware Scheduling（缓存感知调度）**：基于前缀热度/亲和性的请求路由与批调度

所有条目均通过联网检索验证，Paper URL 与 Code URL 均来自真实搜索结果（arXiv / SOSP / OSDI / NeurIPS / SIGMOD / GitHub），未编造。

---

## 检索策略与验证

执行的 `default.websearch` 查询（共 12 次，满足 ≥8 次要求）：

1. `KV cache eviction H2O Scissorhands SnapKV`
2. `KV cache compression KIVI quantization Gear`
3. `prefix caching LLM prefix reuse SGLang RadixAttention`
4. `KV cache offloading hierarchical caching InfiniGen`
5. `KV cache transfer disaggregated inference Mooncake`
6. `KV cache compression PyramidKV`
7. `KV cache management ChunkKV`
8. `cache-aware scheduling LLM inference`
9. `SnapKV LLM KV cache eviction arXiv`
10. `StreamingLLM Efficient Streaming Language Models with Attention Sinks arXiv`
11. `Mooncake KVCache-centric Disaggregated Architecture arXiv`
12. `vLLM PagedAttention KV cache arXiv`

每篇论文记录完整字段：Title / Authors / Year / Venue / Paper URL / Code URL / Category / Why it matters / Type / Priority。

---

## 概览表（15 篇）

| # | Title (short) | Year | Venue | Category | Type | Priority |
|---|---|---|---|---|---|---|
| 1 | StreamingLLM: Attention Sinks | 2023 | arXiv:2309.17453 → ICLR'24 | eviction / recomputation | Foundational | High |
| 2 | H2O: Heavy-Hitter Oracle | 2023 | arXiv:2306.14048 → NeurIPS'23 | eviction | Foundational | High |
| 3 | Scissorhands: Persistence of Importance | 2023 | arXiv:2305.17118 → NeurIPS'23 | eviction | Foundational | High |
| 4 | SnapKV: LLM Knows What You Are Looking For | 2024 | arXiv:2404.14469 → NeurIPS'24 | eviction / compression | Representative | High |
| 5 | PyramidKV: Pyramidal Information Funneling | 2024 | arXiv:2406.02069 | compression | Representative | High |
| 6 | KIVI: Tuning-Free Asymmetric 2bit Quantization | 2024 | arXiv:2402.02750 → ICML'24 | compression (quantization) | Representative | High |
| 7 | GEAR: KV Cache Compression Recipe | 2024 | arXiv:2403.05527 | compression (quant+low-rank+sparse) | Representative | High |
| 8 | ChunkKV: Semantic-Preserving Compression | 2025 | arXiv:2502.00299 → NeurIPS'25 | compression / KV management | Recent | High |
| 9 | SGLang: RadixAttention | 2023/2024 | arXiv:2312.07104 → NeurIPS'24 | prefix caching / KV reuse | Foundational | High |
| 10 | vLLM: PagedAttention | 2023 | arXiv:2309.06180 → SOSP'23 | KV reuse / hierarchical caching | Foundational | High |
| 11 | InfiniGen: Dynamic KV Cache Management | 2024 | arXiv:2406.19707 → OSDI'24 | offloading / hierarchical caching | Representative | High |
| 12 | Mooncake: KVCache-centric Disaggregated Architecture | 2024 | arXiv:2407.00079 → FAST'25 Best Paper | KV transfer / disaggregation / hierarchical | Recent | High |
| 13 | FlowKV: Low-Latency KV Cache Transfer | 2025 | arXiv:2504.03775 | KV transfer / disaggregation | Recent | Medium |
| 14 | HotPrefix: Hotness-Aware KV Cache Scheduling | 2026 | SIGMOD'26 | cache-aware scheduling / hierarchical | Recent | Medium |
| 15 | Online Scheduling for LLM Inference with KV Cache Constraints | 2025 | arXiv:2502.07115 | cache-aware scheduling / recomputation | Representative | Medium |

> 覆盖检查：eviction ✓ (1,2,3,4) | compression ✓ (5,6,7,8) | prefix caching ✓ (9) | KV reuse ✓ (9,10) | offloading ✓ (11) | hierarchical caching ✓ (10,11,12,14) | KV transfer ✓ (12,13) | recomputation ✓ (1,15) | cache-aware scheduling ✓ (14,15)

---

## 详细条目

### 1. Efficient Streaming Language Models with Attention Sinks (StreamingLLM)

- **Title:** Efficient Streaming Language Models with Attention Sinks
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Year:** 2023 (ICLR 2024)
- **Venue / arXiv:** arXiv:2309.17453 → ICLR 2024
- **Paper URL:** https://arxiv.org/abs/2309.17453
- **Code URL:** https://github.com/mit-han-lab/streaming-llm
- **Category:** eviction / recomputation
- **Why it matters:** 首次揭示 attention sink 现象——LLM 在所有层均对初始 token 赋予异常高注意力，单纯滑动窗口驱逐会导致困惑度崩塌；提出仅保留 4 个初始 token + 滑动窗口即可实现无限长流式生成（4M tokens 无崩）、相比重算窗口 22× 加速，成为所有 eviction 方法必须保留 sink 的理论基础，被 H2O/SnapKV/PyramidKV 等广泛引用为 baseline。
- **Type:** Foundational
- **Priority:** High

---

### 2. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

- **Title:** H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianlong Chen, Tianyi Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen
- **Year:** 2023
- **Venue / arXiv:** arXiv:2306.14048 → NeurIPS 2023
- **Paper URL:** https://arxiv.org/abs/2306.14048
- **Code URL:** https://github.com/FMInference/H2O
- **Category:** eviction
- **Why it matters:** 将 KV 驱逐形式化为动态次模 (dynamic submodular) 最大化，给出理论保证；提出 heavy-hitter（累积注意力高）+ 最近窗口 的贪心保留策略，实证在 OPT/LLaMA/GPT-NeoX 上 5-10% 缓存即可保精度，奠定了 attention-score-based eviction 的范式；是 InfiniGen 对比的核心基线，启发了后续所有动态/静态 eviction。
- **Type:** Foundational
- **Priority:** High

---

### 3. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

- **Title:** Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time
- **Authors:** Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava
- **Year:** 2023
- **Venue / arXiv:** arXiv:2305.17118 → NeurIPS 2023
- **Paper URL:** https://arxiv.org/abs/2305.17118
- **Code URL:** N/A
- **Category:** eviction
- **Why it matters:** 提出 persistence of importance 假设——历史上重要的 pivotal token 未来仍重要，给出理论误差界；设计无需微调的固定预算 KV 管理算法，实证 5× 压缩无损，并可与 4-bit 量化叠加至 20×，为 H2O 之外的另一条 eviction 路线，验证了重要性历史累积的有效性。
- **Type:** Foundational
- **Priority:** High

---

### 4. SnapKV: LLM Knows What You Are Looking for Before Generation

- **Title:** SnapKV: LLM Knows What You Are Looking for Before Generation
- **Authors:** Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen
- **Year:** 2024
- **Venue / arXiv:** arXiv:2404.14469 → NeurIPS 2024
- **Paper URL:** https://arxiv.org/abs/2404.14469
- **Code URL:** https://github.com/FasterDecoding/SnapKV
- **Category:** eviction / compression
- **Why it matters:** 发现每个 attention head 在生成前通过末尾 observation window 即可稳定感知关键 prompt 特征，提出带池化的簇式重要 token 选择，自动压缩 prompt KV；在 LongBench/Needle-in-a-Haystack 上显著优于 H2O/StreamingLLM，且与并行解码正交，可集成到 RAG/长上下文生产系统，是当前 eviction 的 SOTA 代表。
- **Type:** Representative
- **Priority:** High

---

### 5. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- **Title:** PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- **Authors:** Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, Wen Xiao
- **Year:** 2024
- **Venue / arXiv:** arXiv:2406.02069
- **Paper URL:** https://arxiv.org/abs/2406.02069
- **Code URL:** https://github.com/ZshPro/PyramidKV
- **Category:** compression
- **Why it matters:** 揭示金字塔式注意力漏斗——下层注意力分散需大缓存、上层聚焦需小缓存，打破每层均匀预算的假设；提出算术序列递减的层间预算分配 + 基于 instruction token 注意力的选择策略，在 LongBench 上仅 12% 缓存即匹敌 Full KV、0.7% 极端压缩下仍比基线高 20.5%（TREC），LLaMA-3-70B 128 条目时 Needle 测试 100% 召回。
- **Type:** Representative
- **Priority:** High

---

### 6. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- **Title:** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- **Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
- **Year:** 2024
- **Venue / arXiv:** arXiv:2402.02750 → ICML 2024
- **Paper URL:** https://arxiv.org/abs/2402.02750
- **Code URL:** https://github.com/jy-yuan/KIVI
- **Category:** compression (quantization)
- **Why it matters:** 通过元素分布全面分析发现 key 按通道量化、value 按 token 量化的不对称最优策略，提出免调优 2bit 量化算法；硬件友好实现使 Llama/Mistral 峰值显存降低 2.6×、批大小 4×、吞吐 2.35–3.47× 且精度近无损，是首个实用化超低精度 KV 量化方案，被 GEAR 等作为 backbone。
- **Type:** Representative
- **Priority:** High

---

### 7. GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM

- **Title:** GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM
- **Authors:** Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao
- **Year:** 2024
- **Venue / arXiv:** arXiv:2403.05527
- **Paper URL:** https://arxiv.org/abs/2403.05527
- **Code URL:** https://github.com/opengear-project/GEAR
- **Category:** compression (quantization + low-rank + sparse)
- **Why it matters:** 指出单纯量化在自回归累积下误差放大导致生成漂移，提出三位一体框架：超低精度量化主体 + 低秩矩阵近似量化残差 + 稀疏矩阵修补离群点；可插拔增强任意量化（KIVI/KCVT/FlexGen），实现 4-bit 近无损、2bit 在 GSM8K 上比 KIVI 高 7.42%（LLaMA3-8B），峰值显存 2.29×、吞吐 2.38× 提升。
- **Type:** Representative
- **Priority:** High

---

### 8. ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference

- **Title:** ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
- **Authors:** Xiang Liu, Zhenheng Tang, Peijie Dong, Zeyu Li, Yue Liu, Bo Li, Xuming Hu, Xiaowen Chu
- **Year:** 2025
- **Venue / arXiv:** arXiv:2502.00299 → NeurIPS 2025
- **Paper URL:** https://arxiv.org/abs/2502.00299
- **Code URL:** https://github.com/NVIDIA/kvpress
- **Category:** compression / KV management
- **Why it matters:** 批判离散 token 驱逐破坏语义完整性，提出以语义 chunk（主谓宾等）为压缩单元，保留信息量最高的 chunk 并复用层间索引（相似度更高，吞吐 +26.5%）；在 LongBench/NIAH/GSM8K 上同压缩比下比 SOTA 高 8.7%，10% 压缩下多数任务持平 Full KV，提供了一种兼顾语言学与系统效率的新视角。
- **Type:** Recent
- **Priority:** High

---

### 9. SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention)

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
- **Year:** 2023 (arXiv) / 2024 (NeurIPS)
- **Venue / arXiv:** arXiv:2312.07104 → NeurIPS 2024
- **Paper URL:** https://arxiv.org/abs/2312.07104
- **Code URL:** https://github.com/sgl-project/sglang
- **Category:** prefix caching / KV reuse
- **Why it matters:** 后端核心创新 RadixAttention 以 radix tree（压缩前缀树）+ LRU 管理所有请求的 KV，自动检测最长共享前缀并复用（无需用户注解），配合 cache-aware 调度与前端 DSL 的 fork 语义实现并行采样共享；在 few-shot、多轮对话、agent 任务上比 vLLM 4.4–5× 吞吐提升，定义了现代 prefix caching 的标准实现，已被 Mooncake/HiCache 继承为分层存储的索引层。
- **Type:** Foundational
- **Priority:** High

---

### 10. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Year:** 2023
- **Venue / arXiv:** SOSP 2023 / arXiv:2309.06180
- **Paper URL:** https://arxiv.org/abs/2309.06180
- **Code URL:** https://github.com/vllm-project/vllm
- **Category:** KV reuse / hierarchical caching / memory management
- **Why it matters:** 将 OS 虚拟内存分页思想引入 KV 管理，PagedAttention 允许 KV 以非连续块存储并通过块表映射，结合按需分配、写时复制与抢占式调度，实现近零碎片与跨请求前缀共享；相比 Orca/FasterTransformer 2–4× 吞吐提升，已成为事实上的开源推理底座，所有后续 hierarchical/offloading/transfer 工作均在其块抽象之上扩展。
- **Type:** Foundational
- **Priority:** High

---

### 11. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management

- **Title:** InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **Authors:** Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim
- **Year:** 2024
- **Venue / arXiv:** OSDI 2024 / arXiv:2406.19707
- **Paper URL:** https://arxiv.org/abs/2406.19707
- **Code URL:** https://github.com/snu-comparch/InfiniGen
- **Category:** offloading / hierarchical caching
- **Why it matters:** 针对 offloading 推理中全量 KV 在 CPU↔GPU 间搬运瓶颈，提出利用当前层输入与下一层部分 Q/K 进行最小预演 (rehearsal) 来推测下一层重要 token，仅预取关键条目并配合权重偏斜 (skewing) 提升推测精度；在长文本生成中相比 H2O/量化基线 3× 加速、精度 +32.6%，首次证明动态稀疏预取可使分层存储在长上下文下实用化。
- **Type:** Representative
- **Priority:** High

---

### 12. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- **Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu
- **Year:** 2024 (arXiv) / 2025 (FAST Best Paper)
- **Venue / arXiv:** arXiv:2407.00079 → FAST 2025 / ACM Transactions on Storage 2026
- **Paper URL:** https://arxiv.org/abs/2407.00079
- **Code URL:** https://github.com/kvcache-ai/Mooncake
- **Category:** KV transfer / disaggregated inference / hierarchical caching
- **Why it matters:** Kimi 生产平台架构，将 prefill/decode 分池并以 KVCache 为一等公民构建跨 CPU/DRAM/SSD 的分布式缓存池与 Transfer Engine（RDMA 87GB/s），Conductor 全局调度以最大有效吞吐 + SLO 为目标并支持过载早期拒绝；真实 traces 上比基线多处理 59–498% 请求、长上下文 525% 吞吐提升，已集成至 vLLM/SGLang/TensorRT-LLM 的 KV Connector/HiCache 后端。
- **Type:** Recent
- **Priority:** High

---

### 13. FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling

- **Title:** FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling
- **Authors:** Weiqing Li, Guochao Jiang, Xiangyong Ding, Zhangcheng Tao, Chuzhan Hao, Chenfeng Xu, Yuewei Zhang, Hao Wang
- **Year:** 2025
- **Venue / arXiv:** arXiv:2504.03775
- **Paper URL:** https://arxiv.org/abs/2504.03775
- **Code URL:** N/A
- **Category:** KV transfer / disaggregated inference
- **Why it matters:** 精准定位 NCCL + PagedAttention 碎片化导致 KV 迁移占端到端 25% 延迟的瓶颈，提出张量重塑 (L,2,B,H → B,L,2,H) 使 NCCL 调用数降低 L×2 以及分段连续内存分配，使传输延迟降低 96%（0.944s→0.053s）、相比 vLLM-Disagg 31.5× 加速，并配合负载感知的弹性 PD 角色调度，在异构 GPU 下端到端 48.9% 提升，填补了 Mooncake 之后的细粒度传输优化空白。
- **Type:** Recent
- **Priority:** Medium

---

### 14. HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems

- **Title:** HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems
- **Authors:** Yuhang Li, Tianchen Zhao, Weihao Cui, Xin Fu, Quan Chen, Jingwen Leng, Chao Li, Minyi Guo
- **Year:** 2025 (SIGMOD 2026)
- **Venue / arXiv:** Proc. ACM Manag. Data (SIGMOD) 2026
- **Paper URL:** https://cs.nju.edu.cn/tianchen/lunwen/2026/sigmod26-liyuhang.pdf
- **Code URL:** N/A
- **Category:** cache-aware scheduling / hierarchical caching
- **Why it matters:** 指出静态 LRU 与无差别卸载导致高热度前缀被过早驱逐，提出实时热度追踪 + 热度感知驱逐 + 选择性准入 + 热度晋升的闭环框架，结合 GPU-Host 协同与异步回填，在 MMLU/HellaSwag 等混合负载下相比 vLLM/SGLang 延迟降低 2–2.25×、吞吐同等提升，且在打乱工作负载下仍鲁棒，是 prefix caching 从 be-st-effort 走向热度驱动调度的代表。
- **Type:** Recent
- **Priority:** Medium

---

### 15. Online Scheduling for LLM Inference with KV Cache Constraints

- **Title:** Online Scheduling for LLM Inference with KV Cache Constraints
- **Authors:** Patrick Jaillet, Jiashuo Jiang, Konstantina Mellou, Marco Molinaro, Chara Podimata, Zijie Zhou
- **Year:** 2025
- **Venue / arXiv:** arXiv:2502.07115
- **Paper URL:** https://arxiv.org/abs/2502.07115
- **Code URL:** N/A
- **Category:** cache-aware scheduling / recomputation
- **Why it matters:** 首次对带 KV 显存约束的在线调度做理论建模，给出以 hindsight 最优为基准的竞争比分析，揭示 KV 线性增长与动态特性使经典调度失效；提出新型批调度算法在最小化延迟的同时管理 KV 显存，并与 continuous batching/chunked prefill 解耦，为 llm-d 等生产调度器的 prefix-aware / 负载感知路由提供了理论支撑。
- **Type:** Representative
- **Priority:** Medium

---

## 交叉对比与演进脉络

| 演进阶段 | 代表系统/方法 | 核心突破 |
|---|---|---|
| **奠基：注意力本质 (2023)** | StreamingLLM → H2O → Scissorhands | 发现 attention sink 与 heavy-hitter 稀疏性，建立 eviction 的理论与经验基础 |
| **静态压缩 (2024)** | SnapKV → PyramidKV → ChunkKV | 从全量累积→单次观察窗口、从均匀→金字塔分层、从 token→chunk，逐步提升语义保真度 |
| **量化正交 (2024)** | KIVI → GEAR | 揭示 key/value 不对称量化最优，叠加低秩+稀疏实现近无损超低精度 |
| **系统化复用 (2023–2024)** | vLLM PagedAttention → SGLang RadixAttention | 从块级内存管理到前缀树自动复用，解决单机碎片与跨请求重复计算 |
| **分层与解耦 (2024)** | InfiniGen → Mooncake → FlowKV | 从预取式卸载到分布式 KV 池与 RDMA 传输，再到细粒度内存对齐消除迁移瓶颈 |
| **调度智能化 (2025–2026)** | HotPrefix → Online Scheduling / llm-d | 从被动 LRU 到热度/预测驱动的缓存感知路由与在线理论保证 |

---

## 建议阅读优先级

- **High (必读奠基/核心):** StreamingLLM, H2O, SnapKV, PyramidKV, KIVI, GEAR, SGLang, vLLM PagedAttention, InfiniGen, Mooncake, ChunkKV
- **Medium (深度扩展):** Scissorhands, FlowKV, HotPrefix, Online Scheduling
- **建议顺序：** StreamingLLM (理解 sink) → H2O/Scissorhands (eviction 范式) → SnapKV/PyramidKV/ChunkKV (静态压缩演进) → KIVI/GEAR (量化) → vLLM → SGLang (复用) → InfiniGen (卸载) → Mooncake/FlowKV (迁移) → HotPrefix/Online Scheduling (调度)

---

## 局限与下一步

- 本清单聚焦 KV Cache 优化，未深入推测解码 (speculative decoding)、稀疏注意力 (sparse attention) 与模型架构级 GQA/MQA 等正交压缩
- 部分系统（如 HotPrefix/FlowKV）实现未完全开源，复现需参考论文描述的热度追踪与内存对齐细节
- 后续可补充：CacheGen/CacheBlend（KV 编解码）、FlexGen（单 GPU offloading 基准）、LMCache（企业级 KV 层）、AttentionStore（多轮对话分层）、SparseX/LMCache 等新兴 prefix 统一框架

---

*Scout B — KV Cache Optimization · 已满足停止条件：≥8 次 websearch，15 篇完整字段记录，文件已写入 `research/manifests/scout_B.md`*
