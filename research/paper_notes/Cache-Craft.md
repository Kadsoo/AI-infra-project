# Paper Metadata

- **Title:** Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation [PAPER FACT]
- **Authors:** Shubham Agarwal (1*), Sai Sundaresan (1*), Subrata Mitra (1,2 †), Debabrata Mahapatra (1), Archit Gupta (2,3), Rounak Sharma (3), Nirmal Joshua Kapu (3), Tong Yu (1), Shiv Saini (1) — * equal contribution, † corresponding author subrata.mitra@adobe.com [PAPER FACT] — Affiliations: 1 Adobe Research, 2 IIT Bombay, 3 IIT Kanpur; Gupta/Sharma/Kapu work done at Adobe Research [PAPER FACT]
- **Venue:** arXiv:2502.15734 [cs.DC] v1 5 Feb 2025 (6,416 KB) [PAPER FACT]; Comments: Accepted at SIGMOD 2025 [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2502.15734 doi:10.48550/arXiv.2502.15734 [PAPER FACT]; HTML https://arxiv.org/html/2502.15734v1 , PDF https://arxiv.org/pdf/2502.15734v1 [PAPER FACT]
- **Code:** [NOT REPORTED] — no public repository linked on arXiv page; paper states integration into vLLM and plan to open-source [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2502.15734 + https://arxiv.org/html/2502.15734v1 (v1, 05 Feb 2025) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1. Problem [PAPER FACT]

- RAG 为 LLM 注入域知识：retriever 按 query 从知识库向量索引中抽取 k 个文本块（chunks），与用户问题拼接为 prompt 送入 LLM [PAPER FACT]
- 同一 chunk 会被不同 query 反复检索到 [PAPER FACT]
- 现有 LLM attention 层对每个 query 都对输入 chunks 全量重复计算 KV（Key-Value），SOTA 无法在 chunk 出现在任意位置、任意上下文时复用 KV-cache；naive reuse 会导致输出质量下降 [PAPER FACT]
- 结果是昂贵 GPU 上的冗余计算与延迟增加 [PAPER FACT]
- 核心问题：如何管理并复用文本 chunk 对应的预计算 KV（称为 chunk-cache），在任意位置/任意上下文的 RAG 请求中正确复用，同时通过少量重算修复质量，并高效存储/驱逐以最大化复用并掩盖开销 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- RAG 无需重训即可定制 LLM，已广泛用于企业 SaaS（Sys-X：基于用户手册的工作流配置问答；Sys-Y：大知识库概念搜索/问答）[PAPER FACT]
- Prefill 远重于 Decode：图1显示 RAG 输入中 retrieved chunks 占 60%–98% tokens [PAPER FACT]；Sys-X/Y 平均约 30k prefill tokens vs 600 decode tokens [PAPER FACT]；Sys-X prefill 占总推理时间高达 55.4%–77% 且为 decode 的 19.3× 算量，Sys-Y 76% /46×，LMSys 22%/4.4× [PAPER FACT]
- Prefill 时延随输入长度平方增长：LLaMA-3-70B + vLLM + 4×A100-80GB 在 32k tokens batch 8 时达 76 秒，并发下可超 100 秒，TTFT 恶化 [PAPER FACT]
- 新 LLM 支持 1M tokens（Claude 3, Gemini）将进一步加长上下文 [PAPER FACT]
- 复用机会：RAG 知识库有限，检索偏态明显 — Sys-X top 5% chunks 被 60% 请求命中；MuSiQue 同样 60%，2WikiMQA 40% [PAPER FACT]；Sys-X 94% 复用跨用户，55% 会话内，67% 跨会话 [PAPER FACT]；Sys-X 一个月内 75% retrieved chunks 被 reprocessed，累计 12B tokens，需 8×A100 上 9600 GPU 小时约 $50k 成本 [PAPER FACT]
- 前缀缓存有效性低：生产环境 exact prefix caching 仅适用于 8% 请求、18% prefill tokens；k-tuple 复用密度随 k 增至 5 时降至 ~5，5-tuple hit 率远低于单 chunk hit 率且不服从幂律，组合爆炸使内存不可行 [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **Cross-attention 缺失与位置敏感：** causal mask 与 positional embedding 使 H_{C_i}^L(C_{1:i}) 不等于 H_{C_i}^L(C_{(1):(i-1)}C_i) [PAPER FACT]；chunk 顺序重排或前缀改变会导致 KV 污染；实验图9/10显示 inter-attention 与 intra-attention 重叠度高时 naive 复用直接导致错误 [PAPER FACT]
2. **多源缓存混合污染：** 多数请求（>50%）的 5 个检索块分散在 3 个以上历史请求中，仅 10%（Sys-X）/8%（2WikiMQA）可在 1 个历史请求内找回全部 5 块；跨 5 个不同历史请求 naive 复用 5 块会使 F1 下降 50%（图8）[PAPER FACT]，即使校正位置 [PAPER FACT]
3. **上下文影响量化缺失：** 需区分 chunk 被外部 prefix 高度 contextualized（不可复用）vs 主要受自身 tokens 影响（可复用）[PAPER FACT]
4. **存储层级与加载延迟：** 大知识库预计算 KV 无法常驻 GPU（需存模型参数与 decode 增长的 KV），从 HBM/Host/SSD 加载延迟若不掩盖会抵消节省 [PAPER FACT]
5. **Prefill 计算瓶颈显著：** 4000 tokens 在 A40 上 LLaMA-70B 需 6s 等，消除 prefill 可使吞吐翻倍 [PAPER FACT]
6. **现有工作局限：** prefix caching（PagedAttention/CacheGen/RadixAttention/RAGCache/Gemini context caching）仅对相同前缀且同序有效，稍变问法即失效 [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**Cache-Craft = chunk 粒度 KV 复用 + 注意力感知的可复用性度量（beta/gamma/CCI/CFO）+ 选中 token 的小比例重算修复 + 相关性分层的早停 + 层间预取流水 + 分层存储与变体管理 [PAPER FACT]**

- **Chunk-cache 抽象：** C(C | C_{1:i-1}) := {(K_C^l,V_C^l)|l in [L]}，即 chunk C 在服务某请求时所在前缀 C_{1:i-1} 下的各层 KV，连同元数据存储 [PAPER FACT]
- **可复用性判定：** inter(C_i,C_j)=sum a_{k,l}，intra(C_i)=sum a_{k,l} [PAPER FACT]；定义 Prefix Overlap Score beta = sum_{j in Sold intersect Snew} inter / sum_{j in Sold} inter [PAPER FACT]；为惩罚重排序引入 Order Penalty gamma = D / C(m,2)，D 为 Kendall Tau 不一致对数，调整得 beta_prime=beta*(1-gamma) [PAPER FACT]；归一化 a(C_i)=sum inter/|C_i||C_j|，b(C_i)=intra/|C_i|^2，层均 a_bar,b_bar，比值输入 sigmoid 得 CCI=1/(1+e^{-a_bar/b_bar}) in [0,1] [PAPER FACT]；CCI 低=外部影响小易复用 [PAPER FACT]；三类场景（自上下文高可复用/外部强上下文不可复用/仅少 token 外部污染可重算修复）见图11 [PAPER FACT]
- **修复开销：** 输出偏差随 CCI 与 1-beta_prime 增大而增大，定义 Cache Fix Overhead CFO = f(CCI,1-beta_prime) 用于权衡；图13显示 CFO 随两者单调增 [PAPER FACT]
- **选择性重算：** 仅重算被外部强影响的 tokens 的 KV（受外部 attention 高的 tokens 贡献 CCI 主要部分，图14）即可使输出偏差陡降（图15）[PAPER FACT]；结合 chunk 与 query 的相关性（relevance）对低相关 chunk 减少重算层数，实现 Adaptive Early Recomputation Termination [PAPER FACT]
- **流水与层次：** Recomputation Planning 决定每 chunk 重算比例与早停层；Layer-wise Preloading 在计算当前层时预取下层 chunk KV；Partial Prefill 在 LLM 中处理混合：部分 tokens 走重算路径，其余直接拼接复用 KV [PAPER FACT]
- **变体与驱逐：** 同一逻辑 chunk 的多版本（不同 Sold）以不同 CCI/beta_prime/频率/大小优先；层次化管理（GPU HBM vs Host DRAM vs SSD）与 frequency-aware 驱逐最大化节省 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **离线/在线两阶段：** 离线计算 inter/intra、a_bar/b_bar/CCI 等元数据并随存储；在线为新请求 S_new 评估候选版本 beta_prime/CFO，选最复用版本 [PAPER FACT]
- **Chunk-Cache Store：** 存 N=100 chunks x M=5 variants 示例约 0.05 TB（LLaMA-3-8B 每 token ~0.1 MB）=50 GB，70B 每 token 0.3 MB =150 GB [PAPER FACT]
- **Recomputation Planning：** 基于 CFO 与节省阈值决定每 chunk 重算 token 比例与提前终止层；对高 CCI + 低 beta_prime 的块投入更高重算，低相关块早停 [PAPER FACT]
- **Layer-wise Preloading：** 计算层 l 时并行从存储加载层 l+1 的 chunk-cache，掩盖 PCIe/NVMe 加载时间 [PAPER FACT]
- **Handling Partial Prefill：** 修改 LLM 前向，使选中 HKVD tokens 重新走 QKVO 与 attention，其余 tokens 的 KV 直接来自复用缓存并参与 attention 拼接 [PAPER FACT]
- **Hierarchical Management：** GPU HBM 存热 chunk，Host/SSD 存量大冷数据；逐层按需调入 GPU 队列 [PAPER FACT]
- **Cache Variants Retrieval and Eviction：** 维护每 chunk 的多版本目录，按潜在节省（无需大重算即可复用）x 期望频率排序，优先保留高复用/高频版本 [PAPER FACT]
- **vLLM 集成：** 非平凡集成，保留 FlashAttention/PagedAttention 的算术强度优化；论文称 non-trivial 并 plan to open-source [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **Redundant computation reduction** 相对于 prefix-caching 与 full recomputation 的比例 [PAPER FACT]
  - **TTFT（prefill latency，time to first token）** 平均值 [PAPER FACT]
  - **Throughput** 在 continuous batching 下请求/分或 tokens/s [PAPER FACT]
  - **End-to-end response latency** [PAPER FACT]
  - **Generation quality：** ROUGE-L F1（long answer/summarization）与 Jaccard Similarity（short answer/True/False）[PAPER FACT]；ROUGE>=0.6 视为 good，>=0.8 几乎不可区分 [PAPER FACT]；另含 250 人 user study Yes/No 正确率 [PAPER FACT]
- **Secondary：**
  - Recompute Savings / 重算比例敏感（20%/30%/45%/60% 等）[PAPER FACT]
  - 不同存储介质与预取下的 TTFT [PAPER FACT]
  - inter/intra 分布、beta/gamma/CCI/CFO 与输出偏差相关性 [PAPER FACT]
  - Prefill length / batch size / model size 敏感度 [PAPER FACT]
  - 成本节省 $ 估算 [PAPER FACT]

## 7. Baselines [PAPER FACT]

- **Full Recomp（Full-Recompute）：** 全量 prefill 无复用，质量上界 [PAPER FACT]
- **Prefix-Cache：** 基于 exact prefix 匹配复用 KV，精度完美但复用率低 [PAPER FACT]；论文用 Kwon et al. 2023 [PAPER FACT]
- **Set-Cache：** 修改 RPE 对 chunk-cache 重排序以找最长 exact prefix 匹配，复用更高但精度较低 [PAPER FACT]
- **Naive KV Reuse（Full-Cache）：** 与 prefix 无关地复用每块 KV，仅校正新位置的 RPE，无重算 [PAPER FACT]
- **Recomputation 策略对比：** Random-Recomp（随机选 token 重算）与 Prefill-H2O（Zhang et al. 2024，选高 attention token 重算），保持与 Cache-Craft 相同平均重算比例以公平对比 [PAPER FACT]
- **Compression Techniques：** Lingua2（Jiang et al. 2023a，基于 GPT-2 训练模型丢弃不显著 token）与 MapReduce（Dean & Ghemawat 2008，摘要压缩上下文），压缩率与 Cache-Craft 重算率对齐（80% 压缩约 20% 重算）[PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Models：** LLaMA-3-8B（TP=1）与 LLaMA-3-70B（TP=4）为主 [PAPER FACT]
- **Datasets / Workloads：**
  - (1) Real-world Sys-X：企业 SaaS 手册工作流，top-k=5，每 chunk 长度可变，总输入 1k-20k tokens 中位数 3.3k [PAPER FACT]
  - (2) Single-Hop QnA：SQuAD（提取式）与 DROP（数值推理），各抽 200 问，切 512-token 块，k=5 [PAPER FACT]
  - (3) Multi-Hop QnA：2WikiMQA 与 MuSiQue，需跨多块事实，抽 200 问 [PAPER FACT]
  - (4) Summarization：CNN（Nallapati 2016）与 XSUM（Narayan 2018），将长块切小后随机选 top-k=5，每大块生成 200 任务，共 40 大块 [PAPER FACT]
- **Tasks：** Single/Multi-Hop 同时做 long vs short answering（通过 mother prompt 控制），另生成 200 True/False 问；Summarization 仅 long summary [PAPER FACT]
- **Cache Warm-Up：** 每数据集前 20 queries 预热以建立 steady-state 缓存 [PAPER FACT]
- **缓存规模：** N=100 chunks x M=5 variants 作为示例评估 [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **平台：** AWS EC2 p4de.24xlarge [PAPER FACT]：8x NVIDIA A100 80GB HBM [PAPER FACT]，Intel Xeon Platinum 8275L 48 核 (96 vCPUs) [PAPER FACT]，1152 GB 主存 [PAPER FACT]，8 TB NVMe SSD 读吞吐 16 GB/s [PAPER FACT]，CPU-GPU 经 PCIe 4.0 x16 带宽 64 GB/s [PAPER FACT]
- **部署：** LLaMA-3-8B 用 TP=1，70B 用 TP=4；评估也含 vLLM + FlashAttention + PagedAttention [PAPER FACT]
- **Model Cache per token：** 8B 约 0.1 MB/token，70B 约 0.3 MB/token [PAPER FACT]
- **软件：** vLLM [Kwon 2023] 基础，定制 partial prefill kernel [PAPER FACT]
- **精度：** [NOT REPORTED] 未明确 FP16/BF16；LLaMA-3 默认可推断但论文未披露 [NOT REPORTED]

## 10. Main Results [PAPER FACT]

- **冗余计算大幅下降：** 在真实生产与合成数据集上，vs SOTA prefix-caching 减少 51% 冗余计算，vs full recomputation 减少 75% [PAPER FACT]
- **Throughput / 延迟（Continuous Batching + ORCA，Sys-X 真实负载）：**
  - LLaMA-3-8B：吞吐提 1.6x，端到端延迟降 2.1x vs prefix-caching [PAPER FACT]
  - LLaMA-3-70B：吞吐提 1.6x，延迟降 2x vs prefix-caching；30% tokens 重算时保持平均 90% base ROUGE F1 [PAPER FACT]
- **质量-重算权衡（图20-21，LLaMA-3-8B/70B，multi/single-hop + summarization）：**
  - Full-Cache 无重算时 ROUGE 仅 0.65（2WikiMQA/MuSiQue 等 multi-hop）[PAPER FACT]
  - 8B 重算 20% ROUGE 提 30%，重算 30% 再提 42% [PAPER FACT]；single-hop 与摘要也有约 20-35% 提升，8B/70B 一致 [PAPER FACT]
  - 8B 45%/60% 与 70B 30%/40% 重算时 ROUGE 距 Full-Recomp 仅 1% 以内 [PAPER FACT]
  - Short-QA/True/False 上比 Full-Cache 高至 50%，达 ROUGE 0.87 vs 0.59 [PAPER FACT]
  - Prefix-Cache 虽精确但 80-95% tokens 需常规计算，compute saving 极低；Set-Cache 多 15-35% 节省但 ROUGE 较低 [PAPER FACT]
  - Random-Recomp 甚至劣于 Full-Cache；Prefill-H2O 仅比 Full-Cache 高 2-10%，多跳任务弱 [PAPER FACT]
- **与 Prompt Compression 对比（5.2.2）：** 在相等压缩率下 Cache-Craft 质量显著高于 Lingua2/MapReduce [PAPER FACT]
- **User Study（250 参与者，2WikiMQA/SQuAD）：** ROUGE>=0.6 时 81% 给 Yes，>=0.8 时 93% 给 Yes，验证指标与感知相关性 [PAPER FACT]
- **受控 TTFT（5.4）：** 不同 prefill 长度/batch size/model size 下均显著降低 TTFT，增益随长度与批量增大而放大 [PAPER FACT]
- **Hierarchical Preloading（5.3.2）：** 层间预取可近完全掩盖加载开销 [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- 输入可切为可复用文本段（RAG stuff 模式），检索基于向量相似度 [PAPER FACT]
- Transformer 架构，RoPE 位置编码可通过旋转校正适配新位置 [PAPER FACT]
- Attention sparsity 成立：仅少数 token 受跨块影响大，故少量重算足够 [PAPER FACT]
- Token embedding 跨层缓慢变化，inter/intra 度量跨层相关 [PAPER FACT]
- 离线 profiling 可估算 recompute vs load 时延以做 planning [PAPER FACT]
- 知识库有限且访问偏态稳定，小热点可命中大流量 [PAPER FACT]
- 文档顺序敏感不可随意重排（lost-in-the-middle）[PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 论文未设独立 Limitations 章；讨论中隐含：依赖离线 attention 统计与阈值调优，偏态分布漂移时需重校准 [PAPER FACT]
- 仅评估 LLaMA-3 系列与 6 个数据集 + Sys-X/Y，未验证更广模型族与长上下文极端 [PAPER FACT]
- 层次化存储评估仅示例 N=100x5，未探索更大规模知识库下的元数据与一致性 [PAPER FACT]
- 未与 KV 压缩（量化/稀疏/逐出）联合 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- 阈值（30% 重算保 90% ROUGE 等）可能随领域/问型漂移，缺在线自适应 r 调优与 SLA 感知 [AGENT INFERENCE]
- CFO 的 f 拟合依赖经验拟合，缺乏形式化误差界与跨模型泛化验证 [AGENT INFERENCE]
- 合成语料与采样（200 问/数据集）规模有限，用户研究仅 250 人x2 数据集，代表性受限 [AGENT INFERENCE]
- 评估以短 answer + ROUGE/Jaccard 为主，长链推理与开放生成质量覆盖不足 [AGENT INFERENCE]
- 多租户/隐私：跨用户复用 chunk KV 潜在信息泄露与访问控制未讨论 [AGENT INFERENCE]
- 未开源代码，可复现性受限；vLLM 集成细节未完全披露 [AGENT INFERENCE]
- 成本 $50k 估算基于 8xA100 9600h，未含存储与预计算能耗 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 如何在线学习 beta_prime/CFO 权重与重算比例，适配分布漂移与不同模型/长度？ [AGENT INFERENCE]
2. 能否将选择性重算与 KV 压缩（KIVI/GEAR/SnapKV）与驱逐联合，实现更细粒度全局 budget 分配？ [AGENT INFERENCE]
3. 变长/语义感知切块与重叠切分如何影响 inter/intra 稀疏度与 CCI？ [AGENT INFERENCE]
4. 对于 Mamba/RetNet 等非 Transformer，chunk 状态复用形式为何？ [AGENT INFERENCE]
5. 在 PD 解耦与跨节点分布式推理下，层次化预取如何扩展至网络传输？ [AGENT INFERENCE]
6. 如何给出形式化质量保证：重算比例与 attention deviation / 下游错误率的误差界？ [AGENT INFERENCE]
7. 多版本同一 chunk 的存储冗余与一致性作废策略如何最优？ [AGENT INFERENCE]
8. 如何在保持 lost-in-the-middle 的前提下做顺序不变或重排感知的校正？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **RAGCache (Jin et al. 2404.12457)：** 面向 RAG 的 knowledge-tree + 分层缓存，仅前缀复用 [PAPER FACT]
- **CacheBlend (Yao et al. 2405.16444 → EuroSys25 Best)：** 任意位置 chunk-KV 融合 + 选择性重算，通过偏差掩盖实现 2–5x 加速 [PAPER FACT]
- **vLLM (Kwon et al. SOSP23) PagedAttention：** 基座 serving 与 block hashing [PAPER FACT]
- **SGLang (Zheng et al. 2312.07104) RadixAttention：** 前缀树复用标准 [PAPER FACT]
- **PromptCache (Gim et al. 2023) / CacheGen (Liu et al. 2310.07240)：** 位置无关复用与压缩加载互补 [PAPER FACT]
- **DistServe / Splitwise / TetriInfer：** PD 解耦，可能与 chunk-cache 复用正成交互 [AGENT INFERENCE]
- **LMCache (Cheng et al. 2510.09665)：** 企业级 KV 缓存层标准化实现 [PAPER FACT]
- **H2O / Scissorhands / SnapKV / PyramidKV：** 注意力稀疏与 KV 驱逐/压缩，与 HKVD/CCI 同源 [AGENT INFERENCE]
- **Lost in the Middle (Liu et al. 2024c)：** 揭示文档乱序对质量影响，约束重排假设 [PAPER FACT]
- **GEAR / KIVI：** 量化压缩可与复用叠加 [AGENT INFERENCE]

---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
