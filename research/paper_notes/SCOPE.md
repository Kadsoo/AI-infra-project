# Paper Metadata

- **Title:** SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation [PAPER FACT]
- **Authors:** Jialong Wu* , Zhenglin Wang* , Linhai Zhang , Yilong Lai , Yulan He , Deyu Zhou? (* equal contribution, ? corresponding) [PAPER FACT] — School of Computer Science and Engineering, Key Laboratory of Computer Network and Information Integration, Ministry of Education, Southeast University, China; Department of Informatics, King’s College London, UK; The Alan Turing Institute, UK [PAPER FACT]
- **Venue:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025, Volume 1: Long Papers), Vienna, Austria, July 2025, pages 10775–10790 [PAPER FACT]; arXiv:2412.13649 [cs.CL] v1 18 Dec 2024, v3 3 Jun 2025 [PAPER FACT]; ACL 2025 Oral [PAPER FACT] — verified via https://aclanthology.org/2025.acl-long.529/ (phase-aware prefill vs decoding separation)
- **DOI/URL:** https://aclanthology.org/2025.acl-long.529/  doi:10.18653/v1/2025.acl-long.529 [PAPER FACT]; arXiv https://arxiv.org/abs/2412.13649 [PAPER FACT]
- **Code:** https://github.com/Linking-ai/SCOPE [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2412.13649v3 + https://aclanthology.org/2025.acl-long.529 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numeric values traced to paper, else [NOT REPORTED].

## 1. Problem [PAPER FACT]

LLM inference 中 KV cache 随上下文长度线性增长成为显存/计算瓶颈，64K上下文在 RTX 3090 上 LLaMA-3.1-8B 已难以承载 [PAPER FACT]。现有压缩分两类：(1) Prefill-Only (SnapKV, PyramidKV) 仅压缩 prefill 阶段、decoding 全保留，输出线性增长导致 OOM；(2) Unified (StreamingLLM, H2O, PyramidInfer) 将两阶段视为统一池，用 greedy Top-K Ψ_K 按累积注意力选 heavy hitters，导致长输出任务中注意力偏向末端，pre fill 关键信息被逐出 [PAPER FACT]。对需细粒度理解的长输入+长输出推理任务（如多题同时推理）尚无 phase-aware 压缩专用探索 [PAPER FACT]。

## 2. Motivation [PAPER FACT]

- 推理任务分 prefill（处理全 prompt 生成首 token）与 decoding（自回归逐 token）[PAPER FACT]。长输入短输出任务只需优化 prefill；长输入长输出（如长文总结、多问答 LongGenBench）两阶段同等重要 [PAPER FACT]。
- Pilot Observation (i): 对 reasoning 任务 (GSM8K+)，prefill 阶段 20% 压缩即导致准确率近 95% 下降，而 PassageRetrieval-en/HotpotQA 在 LongBench 上 20% 压缩仍近乎无损 [PAPER FACT]。说明需特定全上下文的推理任务 attention 并不稀疏 [PAPER FACT]。
- Pilot Observation (ii): 随着 decoding 增长，Top-15% heavy hitters 分布显示保留的 heavy hitters 主要来自 decoding 阶段生成，且随 step 1/300/500 漂移（layers 0/13/31 均如此）[PAPER FACT]；因果注意力使末端 token 获得更高权重，导致偏差 [PAPER FACT]。
- Insight：必须将 prefill 与 decoding 的 KV cache 预算分开分配 [PAPER FACT]。decoding 阶段 KV 仍具稀疏性，可动态选 essential heavy hitters + 保留 recent 窗口 [PAPER FACT]。此外输出可达 8K，decoding 压缩的边际损失远小于 prefill 极致压缩 [PAPER FACT]。

## 3. Bottleneck [PAPER FACT]

1. **Prefill 过度压缩损失理解：** 推理任务需保留问题上下文，统一压缩为给 decoding 腾预算而丢弃 Φ^p [PAPER FACT]。
2. **Heavy hitter 漂移：** greedy Ψ_K 在 unified 池中随着 t 增大，Top-α1 全落在 decoding，Φ^p 缩小、Φ^d 膨胀，造成未来步所需信息丢失 [PAPER FACT] (Fig.2b,3)。
3. **Decoding 窗口必要性：** 自回归需保留 recent tokens（与当前 token 强相关），同时需保留能定位当前预测位置的 heavy hitters [PAPER FACT] (Fig.2c)。
4. **Memory/Transfer 开销：** 每次 decoding step 执行 Top-K 导致频繁 GPU I/O 与池更新；naive sliding 每次更新 Φ^d，Transfer 压力大 [PAPER FACT]。
5. **层间稀疏不显著：** 对长输出任务，PyramidInfer 的层间 budget 调整效果不明显，与 H2O 无显著差异 [PAPER FACT]。
6. **评估缺口：** 主流 LongBench/∞Bench 输出短，无法暴露 decoding 压力；需 LongGenBench 长输出基准 [PAPER FACT]。

## 4. Core Idea [PAPER FACT]

**SCOPE = Separately performs KV Cache Optimization during Prefill and dEcoding [PAPER FACT]。自称首个 phase-level 分离框架 [PAPER FACT]。**

- **不变 Φ^p：** prefill 结束后用 α1 (essential history) + α2 (local window) 压缩得 K0V0 存 Φ^p，后续所有 t 保持不变，视为 attention sinks 保留理解能力 [PAPER FACT]。Eq.2: K0V0 = Ψ_α1(Att_P[:-α2])·KV_P[-α2:] [PAPER FACT]。
- **仅优化 Φ^d 的三策略：**
  - **Slide：** 当 t > M+β1+β2 时，对 Att_t[α1+α2 :-β2] 执行 Ψ_β1 选 Top-β1，与末 β2 拼接作为新 Φ^d；操作完全隔离 Φ^p [PAPER FACT]。
  - **Adaptive：** 从内存视角将 β1 线性自适应：\hat β1 = (t-β2)·β1/(T-β2) (t>β2) [PAPER FACT] Eq.4；池大小 = β2+ (t-β2)β1/(T-β2) < β1+β2 当 t<T，随自回归逐步增长，启动早于 Slide，无额外超参 [PAPER FACT]。
  - **Discontinuous：** 在 Adaptive 上降低 Ψ_K 执行频率，从 T-β2 次降至 β1 次（间隔 (T-β2)/β1）[PAPER FACT]；利用相邻 query 选相似 keys 的特性，仅每间隔执行一次，缓解 I/O [PAPER FACT]。
- **兼容性：** 可作为插件与 SnapKV/PyramidKV 等 prefill-only 方法结合（decoding 用 SCOPE）[PAPER FACT]；正交于 InfLLM/ClusterKV 等复用方法 [PAPER FACT]。

## 5. System Changes [PAPER FACT]

- **Cache Pool 抽象：** Φ = Φ^p ∪ Φ^d；Φ^p_t 常数，Φ^d_t 动态 [PAPER FACT]；CachePool 类含 prefill_cache, decoding_cache, total_cache() [PAPER FACT] (伪代码 Fig.8)。
- **Prefill 算子：** compute_qkv → Att_P → select_top_k_cache(k=α1) on Att_P[:-α2] → 拼接 local 窗口 [PAPER FACT]。
- **Decoding 循环：** 每步 append 当前 K_t,V_t → compute Att_t on total_cache → 按策略条件判断是否 Ψ [PAPER FACT]；Slide 条件 `step > max_prompt_len + β1+β2`，Adaptive `> max_prompt_len+β2`，Discontinuous 额外 `step % jump_interval==0` [PAPER FACT]。
- **超参：** α1+α2 =2048 (LongGenBench-4K) /4096 (8K) 约 60% 输入长，α2=8 [PAPER FACT]；β2=256 以容纳 CoT 长，β1+β2=512/1024 对应 25%/12.5% (4K) 与 12.5%/6.25% (8K) decoding 压缩比 [PAPER FACT]；En.Sum 输入 >170K 平均输出 1.1K 时 prefill 2048 + decoding 512 (50%) [PAPER FACT]。
- **实现：** 基于 HuggingFace，集成 FlashAttention-2；效率测试在 RTX 3090 eager attention batch 8 上 [PAPER FACT]；greedy decoding [PAPER FACT]。

## 6. Target Metrics [PAPER FACT]

- **Primary:**
  - LongGenBench 各子任务 Accuracy (GSM8K+/++, MMLU+/++, CSQA+/++) 及 Avg [PAPER FACT]
  - ∞Bench En.Sum ROUGE-L-Sum [PAPER FACT]
  - 整体压缩率 35% 时是否近 Full Cache (Φ^p 2K + Φ^d 0.5K vs 平均 7.4K 长) [PAPER FACT]
- **Secondary:**
  - 位置敏感准确率 (question position vs accuracy) 验证 H2 丢失缓解 [PAPER FACT] (Fig.4a)
  - 不同 Ψ_K (Observation Window Top-K vs Cumulative Attention Top-K) 与 β1+β2 scaling 敏感度 [PAPER FACT] (Fig.4b)
  - Peak KV Mem (GiB/%) 与 Tokens/s 效率 [PAPER FACT] (Table3)
  - 泛化：∞Bench En.Sum 同 budget 下 ROUGE [PAPER FACT]
  - 插件组合增益 (prefill SnapKV + decoding SCOPE) [PAPER FACT]
- **Not primary:** 首次 token 延迟细分、能耗 [NOT REPORTED]

## 7. Baselines [PAPER FACT]

- **Full Cache:** 无压缩上限 [PAPER FACT]
- **Unified Compression:**
  - **StreamingLLM [Xiao 2024b]:** 保留开端 + 末端 attention sinks [PAPER FACT]
  - **H2O [Zhang 2023]:** recent + Heavy Hitters 基于累积注意力 [PAPER FACT]
  - **PyramidInfer [Yang 2024b]:** 层间金字塔，深层少预算；开源未集成 FlashAttention-2，作者基于 H2O/PyramidKV 复现 [PAPER FACT]
- **Prefill-Only Compression:**
  - **SnapKV [Li 2024]:** observation window + pooling 仅 prefill [PAPER FACT]
  - **PyramidKV [Cai 2024]:** SnapKV 变体跨层预算调整 [PAPER FACT]；两者 decoding 全保留 [PAPER FACT]
- **公平性：** 所有方法 prefill+decoding 总预算一致；初步对比中 H2O/StreamingLLM 甚至在 prefill 独占全 budget 下仍劣于 SCOPE (Table5) [PAPER FACT]

## 8. Workloads [PAPER FACT]

- **LongGenBench-4K (subtask+):** 每个 query 含多个并发推理题，输出 4K；子任务 GSM8K+ (K=30,T=43), MMLU+ (30/53), CSQA+ (40/30) [PAPER FACT] (Table4, 附录B 脚本 https://github.com/Dominic789654/LongGenBench) [PAPER FACT]
- **LongGenBench-8K (subtask++):** 输出 8K 题数翻倍：GSM8K++ 60/21, MMLU++ 60/53, CSQA++ 80/15 [PAPER FACT]
- **∞Bench En.Sum:** 103 例 英文总结，平均输入 >170K（截断公平），平均输出 1.1K [PAPER FACT]
- **Prompt 模板：** System Prompt + 8 CoT 示例 + 后续约 28-52 题 (9-36) 需按 Answer_i / The answer is 格式连答 [PAPER FACT] (Fig.9 probe case)
- **模型：** LLaMA-3.1-8B-Instruct 与 Mistral-7B-Instruct-v0.3 [PAPER FACT]
- **解码：** Greedy [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **实验环境：** NVIDIA A100 (80GB) 与 RTX 3090 (24GB) GPUs [PAPER FACT]
- **加速：** 集成 Flash Attention 2 [PAPER FACT]
- **效率测量：** RTX-3090 (24GB) batch size 8, eager attention 测 Peak KV Mem 与 Tokens/s [PAPER FACT]
- **CPU/RAM/互连：** [NOT REPORTED] 具体 CPU 型号、主机内存、PCIe 版本于正文中未披露 [NOT REPORTED]
- **精度：** [NOT REPORTED] 未明确 fp16/bf16，但基于 FlashAttention-2 推断为常用半精度 [UNVERIFIED]

## 10. Main Results [PAPER FACT]

- **总体主张：** SCOPE 在整体压缩率 35% 时可达 Full Cache 相当性能；分离优于统一极致压缩 prefill [PAPER FACT]。
- **Table1 LongGenBench (LLaMA-3.1-8B-Instruct, prefill 60% ≈2048):**
  - Full Cache Avg: 59.78 (4K) /53.46 (8K) [PAPER FACT]
  - Decoding 25% (4K) /12.5% (8K): SCOPE(Slide) 56.21/49.21 最优；H2O 50.88/43.44, PyramidInfer 53.09/42.88, StreamingLLM 29.48/44.53 [PAPER FACT]
    - GSM8K+ 细节：Slide 46.51 vs H2O 35.04 vs PyramidInfer 38.76 vs StreamingLLM 10.78 [PAPER FACT]
    - GSM8K++: Slide 30.24 vs H2O 22.54 [PAPER FACT]
  - Decoding 12.5% (4K)/6.25% (8K) 更高压：Slide 55.67/47.09 仍领先，H2O 45.94/40.77 [PAPER FACT]
  - Mistral-7B-Instruct-v0.3 Full 34.55/27.15，SCOPE Adaptive/Discontinuous 在 25% 时达 35.01/35.04 超 Full，12.5% 时 Adaptive 34.83 接近 Full，显著优于 H2O 31.60/26.01 [PAPER FACT]
- **Table2 插件 (GSM8K+ LLaMA-3.1-8B):** Prefill Full 53.26；单独 SnapKV/PyramidKV 27.75 崩；+SCOPE Slide 52.17 在 25% decoding 下接近 Full，Compress 35% 甚至有策略超 Full [PAPER FACT]；验证 Φ^d 稀疏性与模块化 [PAPER FACT]
- **Table3 效率 (prefill 60%+decoding 12.5% LLaMA-3.1-8B):**
  - Full 15.6 (100%) 36.57 tok/s；SnapKV/PyramidKV 12.5 (80.1%) 38.28/36.90；StreamingLLM/SCOPE(Slide) 5.8 (37.1%) 但 Slide 18.28 tok/s 因频繁 I/O 低于 StreamingLLM 22.02，Adaptive 同 18.28，Discontinuous 25.92 tok/s 最优 [PAPER FACT]；说明 Adaptive 增预算频繁更新增延迟，Discontinuous 有效缓解 [PAPER FACT]
- **Fig.4a：** H2O 在后段预测位置准确率骤降，SCOPE 三策略缓解 [PAPER FACT]
- **Fig.4b：** decoding 压至25%仅降15%，而 prefill 同比降95%，证明分阶段必要；Cumulative Attention Top-K 优于 Observation Window [PAPER FACT]
- **Fig.4c ∞Bench En.Sum (β1+β2=512)：** Adaptive 最贴近 Full，验证非多问任务亦适用 [PAPER FACT]
- **Table5 公平总 budget 2560 (2048+512)：** GSM8K+ Slide 42.56 vs H2O 32.83 vs StreamingLLM 11.83；En.Sum Slide 17.52 vs H2O 17.64 接近 [PAPER FACT]
- **敏感度：** 固定 β2=256 时 SCOPE 对 β1 稳定而 H2O 敏感；固定总 512 时 Slide 稳定，Adaptive/Discontinuous 在 β2<128 时崩溃（过早逐出 sinks 破坏 CoT）[PAPER FACT]，故选 256 [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- Attention sparse 假设仅对非推理任务成立，推理任务需保留细粒度上下文 [PAPER FACT]
- Heavy hitter 可由 Top-K Ψ_K 近似，且需保留 recent 窗口 β2 保证自回归连续性 [PAPER FACT]
- Max length T 已知或可估计，用于 Adaptive 公式 [PAPER FACT] (T?β1+β2)
- 相邻 decoding 步的 query 选相似 keys，可 discontinuous [PAPER FACT]
- Greedy decoding 足以评估压缩影响 [PAPER FACT]
- 截断长输入（>170K→2K）对对比公平 [PAPER FACT]
- Φ^p 可视为 attention sinks，其质量决定理解 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

1. **Prefill 仍 Top-K：** 仅用朴素 Top-K，未来可用 chunking 等提升估计；增强整体 attention sinks 质量是方向 [PAPER FACT]
2. **正交可集成：** 与 InfLLM 等 KV 复用方法正交，可结合块级选择实现更细粒度 [PAPER FACT]
3. **Decoding I/O 未极致：** 目前更新整个 Φ，理应仅优化 Φ^d（Φ^p 常数）以减 I/O 大小 [PAPER FACT]
4. **模态单一：** 仅验证文本长输出，未试多图生成等 vision 长输出 [PAPER FACT]
5. **数据集局限：** 仅 LongGenBench 与 ∞Bench 两个基准，期望更多样挑战集 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- **T 需预知：** Adaptive 依赖预设 T (4K/8K)；真实 serving 中输出长度不确定，T 估计偏差会错配预算 [AGENT INFERENCE]
- **静态 α 分配：** α1+α2 固定 60%，未按题型/长度自适应；Layer-wise 仍均等，未结合 Pyramid 思想 [AGENT INFERENCE]
- **Ψ_K 仍昂贵：** 即便 Discontinuous 减至 β1 次，每次仍需全 Att 排序，FlashAttention 下 Top-K 与注意力融合未描述 [AGENT INFERENCE]
- **小 batch/长输入评估：** 效率仅 batch 8 eager，≥32 batch 或量化/分页内存下表现未测 [AGENT INFERENCE]
- **截断公平性争议：** En.Sum 平均 170K 截至 2K 可能掩盖真实长上下文压缩收益 [AGENT INFERENCE]
- **β2=256 启发式：** 依赖 CoT 长度先验，对非 CoT 短答任务可能浪费 [AGENT INFERENCE]
- **未评估 Rocks 内存层次：** 仅 GPU DRAM，未涉及 CPU offload/磁盘分层（如 InfiniGen/ShadowKV）[AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 如何在不预知 T 时在线估计输出长度以自适应 β1 增长？能否与 speculative decoding 结合动态伸缩？ [AGENT INFERENCE]
2. 能否学习式预分配 α/β：按问题难度或注意力熵自动决定 prefill 保留度？ [AGENT INFERENCE]
3. Discontinuous 间隔均匀是否最优？能否基于 Att 分布变化检测触发更新？ [AGENT INFERENCE]
4. 如何将 SCOPE 与量化 (KIVI/GEAR) 或 eviction (SnapKV) 联合量化-逐出协同到同一 phase-aware 框架？ [AGENT INFERENCE]
5. 在 PD 分离架构中，仅传输 Φ^d 的细粒度 I/O 能否进一步将 Discontinuous 的 25.92 tok/s 提升至接近 Full 的 36.57？ [AGENT INFERENCE]
6. 多模态长输出（多图/视频）的 KV 结构是否仍呈现相同 heavy hitter 漂移？ [AGENT INFERENCE]
7. 能否将 Φ^p 的 attention sinks 质量通过检索头或语义聚类提升，而非单纯 Top-K？ [AGENT INFERENCE]
8. 更长输出（32K-128K）下 SCOPE 的 35% 压缩是否仍近无损？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **StreamingLLM [Xiao 2024b ICLR]:** 首提 attention sinks 保留首+尾，是 SCOPE 对照的 unified 极简 [PAPER FACT]
- **H2O [Zhang 2023 NeurIPS]:** 累积注意力 heavy hitters + recent 平衡，SCOPE prefill 复用其 Ψ [PAPER FACT]
- **PyramidInfer [Yang 2024b ACL Findings]:** 层间预算递减，SCOPE 证实其在长输出任务不显著 [PAPER FACT]
- **SnapKV [Li 2024 NeurIPS Oral] & PyramidKV [Cai 2024]:** Prefill-only 池化压缩，SCOPE 以插件形式复用 [PAPER FACT]
- **LongGenBench [Liu 2024c EMNLP Findings]:** 本工作主力基准，长输入长输出多问推理 [PAPER FACT]
- **∞Bench [Zhang 2024 ACL]:** 超长上下文 (>100K) 含 En.Sum，验证泛化 [PAPER FACT]
- **InfLLM / ClusterKV / InfiniGen [Xiao 2024a/ Liu 2024a/ Lee 2024 OSDI]:** KV 复用/动态管理，与 SCOPE 正交可叠加 [PAPER FACT]；[AGENT INFERENCE] 后续 ChunkKV/RocketKV 等 two-stage 压缩可视为 SCOPE 思想延伸



## Review Log — Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://aclanthology.org/2025.acl-long.529/ and https://arxiv.org/html/2412.13649v3 — verified phase-aware SCOPE: Φ^p invariant α1+α2=2048/4096 (60% input, α2=8) + Φ^d Slide/Adaptive/Discontinuous (β2=256, β1+β2=512/1024 = 25%/12.5% or 12.5%/6.25%); LongGenBench-4K/8K triple GSM8K+/MMLU+/CSQA+ workloads; Table1 LLaMA-3.1-8B 4K Slide 56.21 vs H2O 50.88, 8K 49.21 vs 43.44; Table3 RTX3090 eager batch8 Full 15.6GiB 36.57 tok/s vs Discontinuous 5.8GiB 37.1% 25.92 tok/s verified.
- **Correction 1 — Venue enrichment:** Added ACL anthology link verification for oral.
- **Correction 2 — Hyperparameter attribution:** Verified α1+α2 2048/4096 ≈60% and β formulas Adaptive \hat β1 = (t-β2)β1/(T-β2) and Discontinuous jump (T-β2)/β1 interval via §4; retained.
- **Correction 3 — Efficiency nuance:** Confirmed Slide 18.28 tok/s slower due to frequent I/O vs StreamingLLM 22.02 and Discontinuous 25.92 best — retained note that Adaptive frequent updates increase latency.
- **Correction 4 — Pilot observations:** Verified 20% prefill compression →95% drop on GSM8K+ vs PassageRetrieval near-lossless, and heavy-hitter drift to decoding (Fig.2b,3) — retained.
- **Status:** All numbers traceable [PAPER FACT]; minor enrichment only.
