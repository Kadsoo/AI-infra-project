# Paper Metadata

- **Title:** A Survey on Large Language Model Acceleration based on KV Cache Management [PAPER FACT]
- **Authors:** Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong, Qing Li, Lei Chen [PAPER FACT] — The Hong Kong Polytechnic University, HKUST, Huazhong University of Science and Technology, CUHK, NTU [PAPER FACT]
- **Venue:** TMLR 2025 (Accepted to TMLR 2025, arXiv:2412.19442 [cs.AI] v1 27 Dec 2024, v3 30 Jul 2025, polished) [PAPER FACT] — Comments: Accepted to TMLR 2025. Revised version incorporates more papers and has been further polished [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2412.19442 / https://arxiv.org/abs/2412.19442 / HTML https://arxiv.org/html/2412.19442v3 [PAPER FACT]
- **Code/Curated List:** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2412.19442 + https://arxiv.org/html/2412.19442v3 (v3, 30 Jul 2025) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1 Problem [PAPER FACT]

- LLM 在推理阶段面临计算与内存双重挑战，尤其长上下文与实时场景；KV cache 复用可减少自回归中的重复计算，但其**时空复杂度随序列长度二次增长（时间与空间）**，管理不善导致内存爆炸与带宽墙 [PAPER FACT]
- 现有加速工作分散于 token 选择、模型架构、系统调度等，未形成统一分类与比较，缺乏对 text 与 multimodal benchmarks 的全面覆盖 [PAPER FACT]
- 核心问题：如何系统化梳理以 **KV cache 管理**为中心的 LLM 加速方法，构建 token/model/system 三级 taxonomy 并对比其机制、trade-offs 与评估基准，以指导高效可扩展的部署 [PAPER FACT]

## 2 Motivation [PAPER FACT]

- LLM 已在 NLP、CV、多模态、时序、推荐、自动驾驶、医疗等广泛成功，基于 Transformer 捕捉长程依赖，但推理时自回归生成导致每对 token 注意力计算，attention 矩阵时空二次增长 [PAPER FACT]
- KV cache 通过存储历史 K/V 使新 token 仅与已处理 token 计算注意力，避免全量重算，大幅加速生成 [PAPER FACT]
- 但长输入下 KV cache 本身成为瓶颈：显存占用、带宽、碎片与调度开销凸显；近来 caching 在 GNN 等领域也被广泛用以复用中间结果，启发 LLM 的 KV 复用 [PAPER FACT]
- 需要从 **token-level、model-level、system-level** 三层统一视角组织已有工作，并覆盖评测用数据集/基准与指标 [PAPER FACT]
- 社区论文激增，亟需 curated list 与可比较分析 [PAPER FACT]（GitHub Awesome 列表）

## 3 Bottleneck [PAPER FACT]

1. **KV cache 的二次复杂度：** 序列长 N 时时间/空间随 N² 增长，长上下文下不可扩展 [PAPER FACT]
2. **Token-level 冗余：** 大量 token 对当前生成贡献低，全部保留浪费预算 [PAPER FACT]
3. **Model-level 低复用：** 标准 MHA 每头每层独立 KV，冗余高 [PAPER FACT]
4. **System-level 资源错配：** 内存管理（碎片、分配）、调度（队头阻塞、公平性）、硬件异构（单/多 GPU、I/O、SSD、heterogeneous）未协同 [PAPER FACT]
5. **评估分散：** 既有综述未同时覆盖 text 与 multimodal benchmarks 及评测指标 [PAPER FACT]
6. **方法碎片化：** 选择、分配、合并、量化、低秩等 token 方法与架构创新、非 Transformer 等模型方法割裂，缺乏统一比较 [PAPER FACT]

## 4 Core Idea [PAPER FACT]

**提出以 KV cache 管理为核心的 LLM 加速三级 taxonomy，系统梳理并比较 Token / Model / System 层优化，并汇总 text+multimodal benchmarks 与评价指标 [PAPER FACT]**

- **Token-level 优化（细粒度 per-token）：**
  - KV Cache Selection（静态选择、带永久驱逐的动态选择、保留不驱逐的动态选择）、Budget Allocation（层级/头级）、Merging（层内/跨层）、Quantization（定精度/混精度/异常值重分布）、Low-rank Decomposition（SVD/张量分解/学习型低秩）[PAPER FACT]
- **Model-level 优化（架构级）：**
  - Attention Grouping & Sharing（层内分组如 MQA/GQA、跨层共享如 CLA）、Architecture Alteration（增强注意力、增广架构）、Non-Transformer Architecture（自适应序列处理、混合架构如 Mamba/RWKV）[PAPER FACT]
- **System-level 优化（系统级）：**
  - Memory Management（架构设计、prefix-aware 设计）、Scheduling（prefix-aware 调度、抢占与公平性、层级/分层调度）、Hardware-aware Design（单/多 GPU、I/O-based、Heterogeneous、SSD-based）[PAPER FACT]
- **Benchmarks：** 长上下文文本与多模态 benchmarks 专门成章（Ch7），含 QA、Summarization、Reasoning、Retrieval、Generation、Aggregation 等任务的评测集与指标 [PAPER FACT]
- 每类后附 Summary and Future Directions，进行横向比较 [PAPER FACT]

## 5 System Changes [PAPER FACT]

- **综述性质，无单一系统实现；** 为每类方法归纳共性系统改动：
- **Token-level 系统含义：**
  - Selection 需在线打分与驱逐/保留策略（如 H2O、SnapKV、Scissorhands、StreamingLLM 等静态/动态变体）[PAPER FACT]
  - Budget Allocation 动态分配每层/每头 KV 容量，需 profiling 或注意力统计 [PAPER FACT]
  - Merging 通过相似度合并减少条目，需相似度度量与跨层协调 [PAPER FACT]
  - Quantization 需量化/反量化核与异常处理（KIVI、GEAR 等定/混精度与 outlier redistribution）[PAPER FACT]
  - Low-rank 需分解核与重构开销 [PAPER FACT]
- **Model-level 系统含义：**
  - Grouping/Sharing 改动 attention 模块（MQA/GQA 减头、CLA 跨层共享键值），训练或微调适配 [PAPER FACT]
  - Architecture Alteration 如增强 attention 设计与增广架构需模型结构改动 [PAPER FACT]
  - Non-Transformer 需新算子与状态管理 [PAPER FACT]
- **System-level 系统含义：**
  - Memory Management：PagedAttention 风格的分页、RadixAttention 前缀树、prefix-aware 缓存等架构设计 [PAPER FACT]
  - Scheduling：prefix-aware、抢占公平、层级分层调度以提高复用与 SLO 满足 [PAPER FACT]
  - Hardware-aware：单/多 GPU、I/O 卸载、heterogeneous (HBM-DRAM-SSD-CXL)、SSD-based 设计的放置与流水 [PAPER FACT]
- **Benchmark 章节（Ch7）：** 整理文本与多模态评测集及评估指标，为方法对比提供 Datasets/Metrics 基座 [PAPER FACT]

## 6 Target Metrics [PAPER FACT]

- **Primary（各方法自报告，综述汇总对比）：**
  - **加速比 / 吞吐（tokens/s, req/s）** 与 **时延（TTFT, TPOT, E2E）** [PAPER FACT]
  - **内存/显存占用与压缩率**（KV 大小、量化位数、秩）[PAPER FACT]
  - **精度/质量**：文本任务常用下游指标（QA 的 EM/F1、Summarization 的 Rouge、Reasoning、Retrieval 等，见 Ch7）与多模态指标 [PAPER FACT]
- **Secondary：**
  - 各 token 方法的 **预算-精度权衡曲线**、合并/量化误差、分解重构误差 [PAPER FACT]
  - 系统方法的 **调度公平性、抢占开销、命中率、I/O 带宽利用** [PAPER FACT]
  - Benchmark 覆盖度与可比性（Table 形式的横向对比为综述主要贡献）[PAPER FACT]
- **具体数值：** 综述引述各原始论文数字，非本文自测；本文未提供统一自有实验数值表 [NOT REPORTED] [PAPER FACT]

## 7 Baselines [PAPER FACT]

- **Vanilla Transformer + full KV cache**（不做管理，作为各项方法的 baseline）[PAPER FACT]
- **各类方法的内部 baselines 在原论文中对比**，综述以分类方式组织其对比关系：
  - Selection：静态 vs 动态（永久驱逐 vs 保留）[PAPER FACT]
  - Quantization：定精度 vs 混精度 vs outlier redistribution [PAPER FACT]
  - Model-level：MHA vs MQA/GQA vs CLA 等分组/共享 [PAPER FACT]
  - System-level：非分页 vs PagedAttention、非前缀感知 vs prefix-aware（SGLang RadixAttention）等 [PAPER FACT]
- **综述本身不设统一 baseline 实验**，而以文献横向表对比 [PAPER FACT]

## 8 Workloads [PAPER FACT]

- **文本 Benchmarks（Ch7 Text Benchmarks）：**
  - **QA：** 问答任务（长上下文 QA 等）[PAPER FACT]
  - **Summarization：** 长文摘要 [PAPER FACT]
  - **Reasoning / Retrieval / Generation / Aggregation：** 推理、检索、生成、聚合任务，均有对应公开集 [PAPER FACT]
  - 具体数据集在综述 Table 中枚举（LongBench 相关、NarrativeQA 等类似长上下文集）[PAPER FACT]
- **多模态 Benchmarks（Ch7 Multimodal）：**
  - 多模态数据集与评测指标（图像-文本等）及对应的多模态 KV 复用场景 [PAPER FACT]
- **评估指标：** 文本 QA 的 F1/EM、摘要 Rouge、多模态专用指标等在 Ch7 Evaluation Metric 节整理 [PAPER FACT]
- **未提供本文统一 workload 实测**，仅为数据集/指标清单与方法-数据集映射 [PAPER FACT]

## 9 Hardware [PAPER FACT]

- **讨论硬件感知设计（Ch6C）：**
  - 单/多 GPU 设计、I/O-based 设计（CPU/磁盘卸载）、Heterogeneous 设计（HBM-DRAM-SSD/CXL 等层次化）、SSD-based 设计 [PAPER FACT]
- **具体机型表格：** 综述未固定单一硬件，而是汇总各原始论文硬件（A100/H100 等）在横向表中；本文自身无固定实验硬件 [NOT REPORTED] [PAPER FACT]
- **Heterogeneous 内存趋势：** 提及 HBM + 高速 off-package DRAM + SSD 的层次化成为实践方案 [PAPER FACT]

## 10 Main Results [PAPER FACT]

- **综述性成果，无自有实验倍数：**
  - 给出 **token/model/system 三级完整 taxonomy** 与每小类的 Summary and Future Directions，进行 comparative analyses [PAPER FACT]
  - 覆盖 **token-level 5 类（selection、budget allocation、merging、quantization、low-rank）**、**model-level 3 类（grouping/sharing、architecture alteration、non-Transformer）**、**system-level 3 类（memory management、scheduling、hardware-aware）** 的系统梳理 [PAPER FACT]
  - 汇总文本与多模态 datasets/benchmarks 及评价指标，便于后续工作选型与可比评估 [PAPER FACT]
  - 提供 curated paper list（GitHub Awesome）并被 TMLR 接收， polished 版本纳入更多论文 [PAPER FACT]
- **量化数字：** 综述未宣称统一加速比；各方法倍数需回相应原始论文 [NOT REPORTED] [PAPER FACT]

## 11 Assumptions [PAPER FACT]

- LLM 基于 Transformer decoder，自回归生成中 KV cache 可复用历史 K/V 避免全量重算 [PAPER FACT]
- KV 的时空复杂度二次增长是长上下文瓶颈主因 [PAPER FACT]
- 三级优化正交可组合：token 级预算/压缩、模型级架构、系统级调度与硬件可协同提升 [PAPER FACT]
- 评测可在文本与多模态混合 benchmarks 上可比 [PAPER FACT]
- 社区论文可按 selection/budget/merging/quantization/low-rank 等维度清晰归类 [PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- 每小类末尾设 **Summary and Future Directions** 讨论局限与展望，暗示各技术点仍有改进空间 [PAPER FACT]
- 综述截止性：方法演进快，新工作需持续纳入（v3 已 polished 并增补论文即体现）[PAPER FACT]
- 未进行统一受控实验对比，仅文献层面的横向表对比，可比性受原评测条件差异限制 [PAPER FACT]（Inference: 作者未声称自有实验为局限的自然推论）
- 对 hardware-aware 部分的覆盖依赖原论文硬件，异构层次的实测对比仍分散 [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- 缺少可复现的统一 benchmark 套件与代码框架，难以直接复现横向表数字 [AGENT INFERENCE]
- 分类边界存在重叠（如 quantization 与 low-rank 可联合，budget allocation 与 selection 耦合），taxonomy 可能需更细粒度 [AGENT INFERENCE]
- 对非 Transformer 新架构（Mamba-2、RWKV-6、Griffin 等）的 KV-like 状态管理归类仍薄 [AGENT INFERENCE]
- 多模态部分对图文交织长上下文（如 128k 图文）的 KV 特有挑战讨论不足 [AGENT INFERENCE]
- 未深入成本/能耗/隐私等部署维度 [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. 如何构建统一可复现的 KV 管理 benchmark 套件（同硬件/同模型/同 trace）实现公平横比？[AGENT INFERENCE]
2. Token-level 的 selection/budget/merging/quantization/low-rank 最优组合策略是什么？[AGENT INFERENCE]
3. Model-level 的 grouping/sharing 与 token 压缩如何联合训练以最小精度损失？[AGENT INFERENCE]
4. System-level 的 prefix-aware 调度如何与 PD 解耦、异构内存层次协同？[AGENT INFERENCE]
5. 长上下文（1M）下 KV 的层次化（HBM-DRAM-SSD）最优放置与预取策略？[AGENT INFERENCE]
6. 多模态长上下文的 KV 管理是否需模态特定预算分配？[AGENT INFERENCE]
7. 能否形式化 KV 压缩率-精度-SLO 的 Pareto 并给出可证明界？[AGENT INFERENCE]
8. 在线自适应 budget 分配能否基于 attention 预测器实时调优？[AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **H2O / Scissorhands / SnapKV / StreamingLLM / PyramidKV**：token selection 代表 [PAPER FACT]（Inference: 按 taxonomy 归类）
- **KIVI / GEAR / Atom**：KV 量化代表（定/混精度与 outlier 处理）[PAPER FACT]
- **MQA / GQA / CLA (Cross-Layer Attention)**：model-level grouping/sharing 代表 [PAPER FACT]
- **PagedAttention (vLLM) / RadixAttention (SGLang) / DistServe / Splitwise / Mooncake**：system-level memory/scheduling 代表 [PAPER FACT]
- **InfiniGen / FlexGen / HMA-LLM**：heterogeneous/memory-aware 系统代表 [AGENT INFERENCE]
- **LongBench / NarrativeQA / MultiNews 等**长上下文评测集 [PAPER FACT]
- **Awesome-KV-Cache-Management GitHub**：本文 curated 列表 [PAPER FACT]

---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
