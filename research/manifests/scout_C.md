# Recent & Emerging Work 2025-2026 — Scout C

> **Role:** Paper Scout C — Recent & Emerging Work  
> **Focus:** long context, RAG workload, agent workload, workflow-aware optimization, heterogeneous memory, distributed inference (disaggregation)  
> **Date:** 2026-08-27  
> **Working dir:** `F:\AIinfraResearch`  
> **Search method:** `default.websearch` 12 independent queries, all Paper/Code URLs from real search results (arXiv / ACL Anthology / ACM / GitHub / NVIDIA blog), no hallucination

## Introduction

Scout C 针对 2024-2026 最新的长上下文、RAG / Agent 工作流、workflow-aware 调度与异构存储/分布式解耦方向做增量巡检。与 Scout A（serving 奠基）与 Scout B（KV Cache 优化）互补，本清单聚焦“新兴且快速迭代”的系统与压缩方案，重点回答：长上下文如何扛 128K-1M？RAG 的 chunk 非前缀复用如何保质增效？Agent 多轮/多智能体工作流的 LRU 为何失效？异构显存-内存-SSD 如何协同？Prefill/Decode 解耦在多轮推理下如何再优化？

所有条目均经联网验证，PPaper URL / Code URL 均来自本次搜索结果页可点击链接。

---

## Search Strategy & Verification

Executed `default.websearch` queries (12 total, satisfies ≥8):

1. `long context KV cache 2025 2026`
2. `RAG workload KV cache optimization 2025`
3. `multi-agent LLM serving workflow cache 2025`
4. `heterogeneous memory GPU CPU SSD KV cache 2025`
5. `distributed LLM inference disaggregation 2025`
6. `LLM inference optimization survey 2025`
7. `prefix caching RAG agent 2025`
8. `KV cache 2025 arXiv survey`
9. `CacheBlend RAG KV cache fusion EuroSys 2025`
10. `ShadowKV high throughput long context 2025 arXiv`
11. `NVIDIA Dynamo distributed inference 2026`
12. `P2P KV cache sharing LLM serving 2025 2026`

Additional fetches via `default.webfetch` on arXiv abs pages for authors/year/venue verification.

---

## Overview Table (14 papers, 2024-2026)

| # | Title (short) | Year | Venue | Category | Type | Priority |
|---|---|---|---|---|---|---|
| 1 | ShadowKV | 2024→2025 | arXiv:2410.21465 → ICML'25 | long context / heterogeneous memory / compression | Recent | High |
| 2 | SCOPE | 2025 | ACL'25 (long) | long context / KV compression (prefill vs decode) | Recent | High |
| 3 | RAGCache | 2024 | arXiv:2404.12457 | RAG workload / prefix caching / hierarchical | Representative | High |
| 4 | CacheBlend | 2024→2025 | arXiv:2405.16444 → EuroSys'25 Best Paper | RAG / prefix caching / KV fusion | Recent | High |
| 5 | Cache-Craft | 2025 | arXiv:2502.15734 → SIGMOD'25 | RAG / chunk-cache / recomputation | Recent | High |
| 6 | KVLink | 2025 | arXiv:2502.16002 | RAG / position-independent caching | Recent | High |
| 7 | KVFlow | 2025 | arXiv:2507.07400 → NeurIPS'25 | agent workload / workflow-aware / prefix caching | Recent | High |
| 8 | Continuum | 2025 | arXiv:2511.02230 | agent workload / multi-turn / cache-aware scheduling | Recent | High |
| 9 | Beluga | 2025 | arXiv:2511.20172 → SIGMOD'26 | heterogeneous memory / CXL / KV management | Recent | High |
| 10 | Dynamic KV Cache Placement | 2025 | arXiv:2508.13231 → IEEE CAL | heterogeneous memory / theory / placement | Recent | Medium |
| 11 | AMPD (Efficient Multi-round over Disaggregated) | 2026 | arXiv:2602.14516 → ICML'26 | distributed inference / disaggregation / multi-round | Recent | High |
| 12 | Shared RAG-DCache (Shared Disk KV) | 2025 | arXiv:2504.11765 | heterogeneous memory / RAG / multi-instance | Recent | Medium |
| 13 | From Attention to Disaggregation | 2025 | arXiv:2511.07422 | distributed inference / disaggregation / survey | Recent | Medium |
| 14 | Survey: LLM Acceleration via KV Cache Management | 2024→2025 | arXiv:2412.19442 → TMLR'25 | survey / KV cache management | Representative | High |

> Coverage check: long context ✓ (1,2) | RAG workload ✓ (3,4,5,6,12) | agent workload ✓ (7,8,11) | workflow-aware optimization ✓ (5,6,7,8,11) | heterogeneous memory ✓ (1,9,10,12) | distributed inference / disaggregation ✓ (11,13) | prefix caching RAG/agent ✓ (3,4,5,6,7) | surveys ✓ (13,14)

---

## Detailed Entries

### 1. ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- **Title:** ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
- **Authors:** Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen
- **Year:** 2024 (arXiv 2024-10-28) → ICML 2025 (v3 2025-04-25)
- **Venue / arXiv:** arXiv:2410.21465 / ICML 2025
- **Paper URL:** https://arxiv.org/abs/2410.21465
- **Code URL:** https://github.com/bytedance/ShadowKV
- **Category:** long context / heterogeneous memory / KV compression / sparse attention
- **Why it matters:** 针对长上下文解码阶段“显存占用 + 逐 token 访存”双瓶颈，提出 pre-RoPE key 的低秩压缩 + value offload 到 CPU，配合 chunk-level landmark 选块与 0.3% outlier 静态驻留，实现 >6× KV 显存压缩、支持 6× 更大 batch、在 A100 上 RULER/LongBench/NIAH 无损且吞吐 +3.04×，并与 MInference 等 prefill 优化正交叠加，代表 2025 长上下文高吞吐的系统级 SOTA。
- **Type:** Recent
- **Priority:** High

---

### 2. SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation

- **Title:** SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation
- **Authors:** Jialong Wu, Zhenglin Wang, Linhai Zhang, Yilong Lai, Yulan He, Deyu Zhou
- **Year:** 2025
- **Venue / arXiv:** ACL 2025 (Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, Volume 1: Long Papers, pp. 10775–10790) / arXiv counterpart reviewed in search
- **Paper URL:** https://aclanthology.org/2025.acl-long.529/
- **Paper URL (PDF):** https://aclanthology.org/2025.acl-long.529.pdf
- **Code URL:** N/A
- **Category:** long context / KV compression / decoding-phase optimization
- **Why it matters:** 指出现有压缩多聚焦 prefill 而忽视 decode，且推理任务会发生 heavy-hitter 漂移；提出 prefill 与 decode 阶段分离优化的轻量框架，分别控制压缩强度与自适应保留，在长输出推理任务上显著降低累积误差，为“分阶段差异化压缩”提供了 2025 的范式参考。
- **Type:** Recent
- **Priority:** High

---

### 3. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation

- **Title:** RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation
- **Authors:** Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin
- **Year:** 2024
- **Venue / arXiv:** arXiv:2404.12457 (v2 2024-04-25) → ACM Transactions on Computer Systems style extended version
- **Paper URL:** https://arxiv.org/abs/2404.12457
- **HTML:** https://arxiv.org/html/2404.12457v2
- **Code URL:** N/A (vLLM + Faiss 集成原型，未独立开源)
- **Category:** RAG workload / prefix caching / hierarchical caching / scheduling
- **Why it matters:** 首个系统化剖析 RAG 长序列瓶颈并利用“知识复用”机会的工作，将检索文档的中间态以 knowledge tree 组织并跨 GPU/host 内存分层缓存，提出感知 LLM 推理与检索模式的替换策略 + 检索/推理动态重叠，相比 vLLM+Faiss TTFT 4×、吞吐 2.1×，为后续 CacheBlend/Cache-Craft 等非前缀复用奠基。
- **Type:** Representative
- **Priority:** High

---

### 4. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

- **Title:** CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion
- **Authors:** Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang
- **Year:** 2024 (arXiv) / 2025 (EuroSys)
- **Venue / arXiv:** arXiv:2405.16444 → EuroSys 2025 Best Paper (pp. 94–109, DOI:10.1145/3689031.3696098)
- **Paper URL:** https://arxiv.org/abs/2405.16444
- **HTML:** https://arxiv.org/html/2405.16444v3
- **Code URL:** https://github.com/LMCache/LMCache
- **Category:** RAG workload / KV cache fusion / selective recomputation / prefix caching
- **Why it matters:** 针对 RAG 中多 chunk 拼接但非前缀导致现有 prefix caching 失效，提出“任意位置预计算 chunk-KV 直接复用 + 仅对 高 KV deviation 的少量 token 选择性重算”修复跨块注意力缺失，并用 loading controller 流水化重算与加载，在 3 模型 × 4 数据集上 TTFT 2.2–3.3×、吞吐 2.8–5× 且质量近无损，现已集成进 LMCache/UCM，成为 RAG KV 复用的事实标准。
- **Type:** Recent
- **Priority:** High

---

### 5. Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation

- **Title:** Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation
- **Authors:** Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Mahapatra, Archit Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu, Shiv Saini
- **Year:** 2025
- **Venue / arXiv:** arXiv:2502.15734 → SIGMOD 2025 (Accepted)
- **Paper URL:** https://arxiv.org/abs/2502.15734
- **Code URL:** N/A
- **Category:** RAG workload / workflow-aware optimization / partial recomputation / cache management
- **Why it matters:** 更贴生产 RAG：高频 chunk 复用但非前缀对齐，直接拼接会“污染”KV。Cache-Craft 给出可复用性检测 + 受污染 token 的小比例重算修复 + 缓存组织/淘汰的闭环系统，在真实生产 traces 上相比 SOTA prefix-caching 冗余计算 -51%、相比全重算 -75%，连续批下吞吐 +1.6×、端到端延迟 -50%（LLaMA-3 8B/70B），且质量保持。
- **Type:** Recent
- **Priority:** High

---

### 6. KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse

- **Title:** KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
- **Authors:** Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang
- **Year:** 2025
- **Venue / arXiv:** arXiv:2502.16002 (v4 2025-11-10)
- **Paper URL:** https://arxiv.org/abs/2502.16002
- **Code URL:** https://github.com/UCSB-NLP-Chang/KVLink
- **Category:** RAG workload / position-independent caching / KV reuse / prefix caching
- **Why it matters:** 定义 Position-Independent Caching (PIC) 场景：同一文档在不同请求中位置不同导致传统 prefix 失效。通过文档级独立预计算 KV + 在线拼接时全局位置重编码 + 可训练特殊 token 恢复跨块自注意力，在 7 数据集上平均 +4% QA 准确率且 TTFT -96%，并可与 KV 量化压缩叠加，是 RAG 高复用服务的轻量高效路线。
- **Type:** Recent
- **Priority:** High

---

### 7. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows

- **Title:** KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- **Authors:** Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, Yufei Ding
- **Year:** 2025
- **Venue / arXiv:** arXiv:2507.07400 → NeurIPS 2025
- **Paper URL:** https://arxiv.org/abs/2507.07400
- **HTML:** https://arxiv.org/html/2507.07400v1
- **Paper URL (NeurIPS PDF):** https://papers.nips.cc/paper_files/paper/2025/file/b7971d31a7d5eb0f1eed2f8f6f368195-Paper-Conference.pdf
- **Code URL:** N/A
- **Category:** multi-agent LLM serving / workflow-aware optimization / prefix caching / prefetching
- **Why it matters:** 首次揭示 LRU 在 agent 工作流中频繁在重用前误驱逐的根本缺陷，抽象 Agent Step Graph 计算 steps-to-execution 估计下一次激活距离，指导 KV 节点级细粒度驱逐 + 下一步 agent 的 CPU→GPU 全重叠预取，相比 SGLang hierarchical radix cache 单工作流 1.83×、多并发 2.19×，是 workflow-aware KV 调度的里程碑。
- **Type:** Recent
- **Priority:** High

---

### 8. Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

- **Title:** Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **Authors:** Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica
- **Year:** 2025
- **Venue / arXiv:** arXiv:2511.02230 (v6 2026-05-25)
- **Paper URL:** https://arxiv.org/abs/2511.02230
- **Code URL:** N/A
- **Category:** agent workload / multi-turn scheduling / heterogeneous memory / TTL cache
- **Why it matters:** 聚焦 ReAct/工具调用型 agent：工具执行间隙若直接驱逐 KV，下一次 prefill 代价高但常驻又挤占队列。Continuum 提出 TTL 机制结合重算/重载成本与排队延迟动态决定保留时长，过期自动驱逐提升鲁棒性，并配合 program-level FCFS 保证多轮连续性，在 SWE-Bench/BFCL/OpenHand 上对 Llama-3.1 8B/70B、Gemma-3 等平均作业完成时间 8× 优化。
- **Type:** Recent
- **Priority:** High

---

### 9. Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management

- **Title:** Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management
- **Authors:** Xinjun Yang, Qingda Hu, Junru Li, Feifei Li, Yicong Zhu, Yuqi Zhou, Qiuru Lin, Jian Dai, Yang Kong, Jiayu Zhang, Guoqiang Xu, Qiang Liu
- **Year:** 2025
- **Venue / arXiv:** arXiv:2511.20172 → SIGMOD 2026 (Accepted, 13 pages)
- **Paper URL:** https://arxiv.org/abs/2511.20172
- **Code URL:** N/A
- **Category:** heterogeneous memory / CXL / distributed KV management / disaggregation
- **Why it matters:** 首次让 GPU 通过 CXL switch 以原生 load/store 语义直访大规模共享内存池，系统化刻画商用 CXL 池特性并给出设计指南，基于此构建 Beluga-KVCache：相比 RDMA 方案在 vLLM 上 TTFT -89.6%、吞吐 7.35×，为突破 CPU DRAM 通道数限制、支撑长上下文与大 KV 池提供了硬件级新路径，获 SIGMOD 录用。
- **Type:** Recent
- **Priority:** High

---

### 10. Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System

- **Title:** Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System
- **Authors:** Yunhua Fang, Rui Xie, Asad Ul Haq, Linsen Ma, Kaoutar El Maghraoui, Naigang Wang, Meng Wang, Liu Liu, Tong Zhang
- **Year:** 2025
- **Venue / arXiv:** arXiv:2508.13231 (v2 2025-09-15) → IEEE Computer Architecture Letters (IEEE CAL)
- **Paper URL:** https://arxiv.org/abs/2508.13231
- **Code URL:** N/A
- **Category:** heterogeneous memory / GPU-CPU-SSD tiering / scheduling theory
- **Why it matters:** 面向 GH200 类 HBM+LPDDR5X+NVLink-C2C 的异构带宽系统，形式化动态 KV 放置为容量约束下的带宽最大化问题并推导理论上界，首次给出 HBM/外存聚合带宽的上限与运行时优化 headroom 估计，揭示外存带宽已接近 HBM 一个数量级、智能放置可显著提升有效带宽，为异构调度提供了理论基座。
- **Type:** Recent
- **Priority:** Medium

---

### 11. Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD)

- **Title:** Efficient Multi-round LLM Inference over Disaggregated Serving
- **Authors:** Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu
- **Year:** 2026
- **Venue / arXiv:** arXiv:2602.14516 (v2 2026-07-21) → ICML 2026
- **Paper URL:** https://arxiv.org/abs/2602.14516
- **Code URL:** N/A (基于 NVIDIA Dynamo/NIXL 实现)
- **Category:** distributed inference / prefill-decode disaggregation / multi-round / workflow-aware
- **Why it matters:** 指出既有 P/D 解耦仅针对单轮，而 agent/迭代检索的多轮会产生交错的增量 prefill，提出 AMPD 自适应决定增量 prefill 落在 prefill 池还是 decode 池并联合调度，同时给出面向多轮的最优资源分配与并行策略规划算法，SLO 达成率显著超越 Dynamo/vLLM 基线，填补了“解耦×多轮”空白。
- **Type:** Recent
- **Priority:** High

---

### 12. Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs (Shared RAG-DCache)

- **Title:** Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs
- **Authors:** Hyungwoo Lee, Kihyun Kim, Jinwoo Kim, Jungmin So, Myung-Hoon Cha, Hong-Yeon Kim, James J. Kim, Youngjae Kim
- **Year:** 2025
- **Venue / arXiv:** arXiv:2504.11765
- **Paper URL:** https://arxiv.org/abs/2504.11765
- **Code URL:** N/A
- **Category:** heterogeneous memory / RAG workload / multi-instance sharing / disk tiering
- **Why it matters:** 将 RAG 的“文档局部性 + 推理排队延迟”转化为增益：利用排队窗口主动预生成磁盘 KV 并跨多实例共享，结合最优配置搜索，在单机双 GPU 上吞吐 +15~71%、延迟 -12~65%，证明“磁盘并非累赘而是 RAG 多实例的扩容利器”，与 LMCache/Mooncake 的多级存储互补。
- **Type:** Recent
- **Priority:** Medium

---

### 13. From Attention to Disaggregation: Tracing the Evolution of LLM Inference

- **Title:** From Attention to Disaggregation: Tracing the Evolution of LLM Inference
- **Authors:** Madabattula Rajesh Kumar, Srinivasa Rao Aravilli, Mustafa Saify, Shashank Srivastava
- **Year:** 2025
- **Venue / arXiv:** arXiv:2511.07422
- **Paper URL:** https://arxiv.org/abs/2511.07422
- **Code URL:** N/A
- **Category:** distributed inference / disaggregation / survey / system architecture
- **Why it matters:** 以分布式系统视角系统化梳理从 Transformer 到万亿参数的推理演进：为何 TTFT/ITL 需解耦、以服务分解/资源解聚/负载分区重构单体 GPU 集群，涵盖 DistServe/Mooncake/NVIDIA Dynamo/AIBrix 的架构对比与独立扩缩容实践，是 2025 下半年入门 distributed inference 的最佳 roadmap。
- **Type:** Recent
- **Priority:** Medium

---

### 14. A Survey on Large Language Model Acceleration based on KV Cache Management

- **Title:** A Survey on Large Language Model Acceleration based on KV Cache Management
- **Authors:** Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong, Qing Li, Lei Chen
- **Year:** 2024 (arXiv) / 2025 (TMLR)
- **Venue / arXiv:** arXiv:2412.19442 → Transactions on Machine Learning Research (TMLR) 2025
- **Paper URL:** https://arxiv.org/abs/2412.19442
- **Code URL:** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
- **Category:** survey / KV cache management / LLM inference optimization / heterogeneous memory / RAG-agent pipeline
- **Why it matters:** 唯一同时覆盖 token-level（selection/budget/merging/quant/low-rank）、model-level（GQA/MQA/attention）与 system-level（memory/scheduling/hardware）的综合 survey，横向对比 200+ 工作并梳理 benchmarks，明确指出“分层与流水化已成为系统级优化主线，而 RAG/agent 的非前缀复用与异构放置仍是 2025 最大缺口”，适合作为 Scout C 的总览地图。
- **Type:** Representative
- **Priority:** High

---

## Cross-comparison & Evolution

| Evolution Stage | Rep. Systems | Core Breakthrough | Scout C Evidence |
|---|---|---|---|
| **Long context goes throughput-first (2024-2025)** | ShadowKV → SCOPE | 从单请求显存节省转向批吞吐与 rank/稀疏协同；decode 阶段需独立优化 | ShadowKV 6× batch, SCOPE 分阶段压缩 |
| **RAG: prefix → any-position (2024-2025)** | RAGCache → CacheBlend → Cache-Craft → KVLink | 从严格前缀复收到任意位置 chunk 复用 + 轻量重算修复跨块注意力 + 位置重编码 | CacheBlend 2.8-5×, KVLink -96% TTFT |
| **Agent: LRU → workflow-aware (2025)** | KVFlow → Continuum → AMPD | Agent Step Graph 预测 + TTL 保留 + P/D 多轮自适应路由 | KVFlow 2.19×, Continuum 8× JCT |
| **Heterogeneous: offload → CXL shared pool (2025-2026)** | Dynamic Placement theory → Beluga CXL → Shared RAG-DCache | 理论上界 → 硬件级共享内存 → 磁盘多实例共享 | Beluga 7.35× vs RDMA, Dynamic 理论 headroom |
| **Distributed: single-round P/D → multi-round & surveyed (2025-2026)** | DistServe/Mooncake → AMPD → Survey | P/D 解耦从单轮走向多轮增量 prefill 与资源规划 | AMPD ICML'26, From Attention survey |

---

## Reading Priority

- **High (必读):** ShadowKV, CacheBlend, Cache-Craft, KVLink, KVFlow, Continuum, Beluga, AMPD, RAGCache, SCOPE, TMLR Survey
- **Medium (扩展):** Dynamic KV Placement, Shared RAG-DCache, From Attention to Disaggregation
- **Suggested order:** TMLR Survey (建图) → ShadowKV/SCOPE (长上下文) → RAGCache → CacheBlend → Cache-Craft/KVLink (RAG 非前缀) → KVFlow → Continuum (agent 工作流) → Beluga/Dynamic (异构) → AMPD → Shared RAG-DCache (磁盘) → From Attention (distributed 全景)

---

## Limitations & Next Steps

- 部分系统（Beluga/Continuum/KVFlow）尚未完全开源，复现需参考论文的 CXL/ TTL / Step Graph 细节；CacheBlend/KVLink/ShadowKV 已在 LMCache / GitHub 开源可直接集成。
- 本清单未深入推测解码 (speculative decoding)、量化与 P/D 解耦的正交叠加，下一步可补充：NVIDIA Dynamo 1.0 (disaggregated serving 工业实现, https://github.com/ai-dynamo/dynamo ), SGLang Mooncake Integration, SwiftCache 异构多轮会话共享。
- 推荐与 Scout A/B 联动：A 的 PagedAttention/RadixAttention 为本清单所有复用/放置的内存抽象；B 的 eviction/compression 为 SCOPE/ShadowKV 的前置理论。

---

*Scout C — Recent & Emerging Work · 已满足停止条件：12 次 websearch，14 篇完整字段记录，文件已写入 `research/manifests/scout_C.md`*
