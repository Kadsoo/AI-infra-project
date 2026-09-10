# Research Question

> **RQ-2 — How fragile are fixed retention, precision, and chunk budgets when task type or generation phase changes?**
>
> ## Observation
>
> Retention, compression, and quantization methods often expose static budgets or group/residual parameters. The corpus also shows that attention patterns and quality sensitivity differ by task, layer, model, and decode phase.
>
> ## Existing Approach
>
> H2O/StreamingLLM/SnapKV/PyramidKV select or retain state; SCOPE separates prefill/decode behavior; KIVI and GEAR reduce numerical precision. Most published comparisons hold the chosen setting fixed while changing only part of the workload.
>
> ## Fragile Assumption / Limitation
>
> A setting selected from an offline average is assumed to remain near-optimal and quality-safe when the workload changes. This does not follow from a good result on a single benchmark.
>
> ## Failure Scenario
>
> Switch from retrieval QA to summarization/reasoning, change output length, use a model with MQA/GQA behavior, or move from prefill-selected to long decode state.
>
> ## Observable Consequence
>
> - Accuracy/F1/LongBench/PPL crosses a predeclared tolerance at a fixed memory target.
> - The setting that maximizes memory saving increases TPOT or kernel overhead.
> - A per-condition oracle exposes a substantial loss from the static setting.
>
> ## Relevant Papers
>
> H2O (2023), StreamingLLM (2023), SnapKV (2024), PyramidKV (2024), KIVI (2024), GEAR (2024), SCOPE (2025); `assumption_map.md` A8–A12 and `research_tensions.md` T1/T8.
>
> ## Why Worth Investigating
>
> The experiment can rule out an overgeneralized claim cheaply and would inform whether adaptation is needed before anyone designs a controller.
>
> ## Cheap Validation
>
> Hold implementation and memory budget constant. Test 3–4 existing methods across a small matrix of retrieval, summarization, reasoning, and long-generation tasks. Compare fixed published settings against a small offline per-condition oracle; report quality, HBM, TTFT/TPOT, and overhead.
>
> ## Research Potential
>
> High.
>
> ## Confidence
>
> High for sensitivity; Medium for a general online-adaptation research gap.

# Revision History (post-adversarial-review)

> Changes applied 2026-08-27 after `design_review.md` (critic verdict: require rework). Each change is traceable to a review issue:
>
> - **R1 (critic #1/#3/#A, native definitions):** the native/non-native rule is now uniform: *native = exactly the task/model/layout set the paper actually reported*, including summarization for SnapKV and Falcon-7B for KIVI. H1's non-native cells are genuinely out-of-suite: SnapKV → ∞Bench En.Sum + GSM8K 8-shot CoT; KIVI → ∞Bench En.Sum + the long-decode generation-phase cell. In-suite crossings (SnapKV GovReport, KIVI Falcon) are demoted to "published-point validity" observation cells and can never witness H1.
> - **R2 (critic #1, phase gating):** the generation-phase cell runs unconditionally in wave 1.
> - **R3 (critic #4, evidence starvation):** the open-value cells (StreamingLLM retrieval-at-evicted-distance, H2O long-form/CoT, GEAR per-task, PyramidKV per-task native) run unconditionally in E5.
> - **R4 (critic #4/#6, anchor mismatch):** triage uses same-model rows only; KIVI cells match the papers' max_seq 4096 truncation protocol + one untruncated sensitivity cell.
> - **R5 (critic #9/#E, statistics):** per-cell bootstrap CIs + pre-registered non-discriminating band (±2 metric points); determinism check on the custom-kernel cell.
> - **R6 (critic #D, contradiction):** universal fragility (native also ≥10% everywhere) is now uniformly classified as **H1 falsified** (attribution to task/phase/layout change fails).
> - **R7 (critic #7, H2 leg):** the "≥2× memory saving with <5% throughput gain" leg applies to the KV-representation HBM delta at the tested batch, not to peak HBM including weights.

# Motivation

本地语料中的所有 KV 保留/压缩方法都以**固定预算、位宽或分组参数**作为公开发表的操作点，且每一篇只在单一任务套件上验证。H2O 固定 20% KV budget（5× memory reduction），在 HELM/lm-eval-harness 任务（OpenBookQA、COPA、PIQA、MathQA、RTE、XSUM、CNN/Daily Mail）上验证（H2O.md §8/§10）；StreamingLLM 固定 4 sinks + rolling window（Llama-2 总计 2048 容量），以 PG19 长文 PPL 和 StreamEval（答案仅位于 20 行之前）为质量指标（StreamingLLM.md §6/§8）；SnapKV 固定 observation window（16/32/64）、max_capacity_prompt（1024/2048/4096）与 pooling kernel（5/7/13），在 LongBench 16 任务、380K Needle-in-a-Haystack 与 Command-R RAG 上验证，且压缩只在 prefill 结束时执行一次、generation 阶段不再更新（SnapKV.md §5/§8/§12）；PyramidKV 固定 α=8 个 instruction tokens 与 β=20 的算术递减逐层分配，操作点为 KV=64（约 0.7–0.8%）与 KV=2048（约 12%）（PyramidKV.md §4/§5）；KIVI 固定 2-bit、G=32、R=128 全精度 residual（KIVI.md §4）；GEAR 固定 2-bit、s=2% 稀疏、r=4（prefill）/r=2（decode buffer）、n_b=20（GEAR.md §4）；SCOPE 固定 prefill 预算 λ1+λ2=2048/4096（约 60% 输入）与 decode 预算 512/1024、λ2=256（SCOPE.md §5）。stage2_final_review.md 的三条 fragility 行——"Offline profiles and fixed thresholds transfer"、"One lossy KV configuration preserves quality"、"Past/local attention predicts future token importance"——与 T1/T6/T8 的 falsifier 全部以"固定设置在条件变化时保持质量安全"为前提，但本地没有任何一项证据在同一 harness 内跨任务、跨阶段验证过这一前提。

语料内部已记录多条**任务敏感性**信号（跨论文、未匹配 harness）：SnapKV 自证 context-dependent importance——同一文档上不同 instruction 命中率显著下降（SnapKV.md §4.2 Fig.4），且其 1024 capacity 下 summarization 任务 GovReport 27.97→19.79（相对 −29%）远大于 retrieval 任务 NrtvQA 18.18→18.02（相对 −0.9%），升至 4096 才恢复至 25.34（SnapKV.md §10）；PyramidKV 在 few-shot 任务 TREC 上获益 +20.5 绝对点，而 summarization 增益小（PyramidKV.md §10），且 KV=64 时 Mistral-7B 平均 LongBench 42.71→32.19（相对 −24.6%），KV=2048 时仅 −2.5%（PyramidKV.md §10）；H2O 自己的数据即显示 Local（recent-only）在 60% budget 就 collapse（LLaMA-13B XSUM、LLaMA-7B CNN/Daily Mail），OpenBookQA 5-shot Local 25.20 vs Full 43.20/44.40（H2O.md §10）；GEAR 证明 2-bit 下 reasoning 任务（GSM8k 8-shot CoT 平均）baseline 全线崩坏——per-token 量化 7.67、KIVI 25.25，而 GEAR 40.20 ≈ FP16 40.52；同一篇的 easy-task 结果显示 2-bit per-token 在 LongBench 已 near-lossless（27.69 vs FP16 26.82）（GEAR.md §10）；SCOPE pilot 观测到 20% prefill 压缩在 GSM8K+ 上 ≈95% 精度下降、在 PassageRetrieval-en/HotpotQA 上近乎无损（SCOPE.md §2）。这些信号一致指向"同一固定预算的质量代价随任务类别变化"，但全部来自不同模型、不同 evaluation harness，不能直接相加。

模型 KV layout 与 generation phase 是同样被记录的敏感维度：KIVI 在 Falcon-7B（MQA，单 KV head）上 2-bit 失效（CoQA 57.48 vs 59.83，GSM8K 3.41 vs 4.55，作者明确"Falcon needs 4-bit"），而同设置对 Llama-2-7B 仅 CoQA 63.88→63.05（KIVI.md §10）；KIVI 的 group size 从 32 升到 128 使 GSM8K 从 20.77 掉到 17.29，去掉 R=128 全精度 residual 的 fake 2-bit 只有 12.21（KIVI.md §10）；StreamingLLM 自证 Falcon/MPT 1 sink 已稳定而固定 4 sinks 浪费 3× 容量（StreamingLLM.md Table2），且其 rolling 设计无法检索 recent window 之外的中段信息——StreamEval 的答案只放在 20 行之前，未测被驱逐距离的 retrieval（StreamingLLM.md §13 推断限制）；SCOPE 直接观测 heavy-hitter 漂移：decode 阶段注意力质量转移到生成 token（step 1/300/500、layer 0/13/31），并指出 prefill-only 方法（SnapKV/PyramidKV）在长生成下 decode KV 不压缩、必然 OOM 或突破固定内存承诺（SCOPE.md §1/§2）。assumption_map.md 对 A9（attention sparsity）、A10（attention sinks）、A12（local observation window）的置信度评级均为 "strong for chat/QA, weak for summarization/retrieval-dense"——即语料自己就把这些假设标为条件性的，但没有任何方法在其论文中把该条件性显式量化。

TPOT/kernel 开销信号同样已在语料中记录但从未成为独立测量对象：SCOPE 的 decode-phase 每步 Top-K 重选（Slide）在 batch 8 下仅 18.28 tok/s vs Full Cache 36.57 tok/s（−50%），降低选择频率的 Discontinuous 提升到 25.92 tok/s（SCOPE.md Table3）；GEAR 每 n_b=20 步对 streaming buffer 重压缩（r=2）产生周期性开销（GEAR.md §4）；KIVI 依赖 fused dequant+matmul kernel 才能避免额外每步成本（KIVI.md §3.3）。这意味着"内存节省最大的设置"在长 decode 或不同 KV head layout 下可能把省下的内存变成每步 kernel 时间——RQ-2 的第二个 observable consequence 可以被直接量化。

因此 RQ-2 值得作为 Tier 1 验证（stage2_final_review.md §8："Fast to falsify and foundational for several compression/retention claims; it can separate genuine operating-boundary evidence from a generic 'adaptive controller' story"）。本阶段只回答"问题是否存在"：以每个方法自己发表的固定设置为处理，在同一 harness 内横跨 task class × generation phase × KV layout 矩阵，与 full-KV 基线以及同预算 per-condition oracle 对比。**不提出任何自适应控制器、动态预算策略或新压缩方法**；不断言 novelty，动机完全来自本地语料；Null 结果同样有价值（缩小研究空间，避免不必要的控制器设计）。

# Primary Hypothesis

Under a fixed memory target equal to each method's published operating point (H2O 20% KV, StreamingLLM 4 sinks, SnapKV capacity 1024, PyramidKV α=8 with KV=64, KIVI 2-bit G=32 R=128, GEAR 2-bit s=2% r=4), the static configuration's embedded task- and phase-specific importance assumption (local observation-window voting, heavy-hitter persistence, fixed sink count, uniform bit-width/group size) causes the primary quality metric to cross a predeclared tolerance (≥10% relative degradation vs the full-KV baseline on the same task) on at least one non-native condition (long-form summarization, chain-of-thought reasoning, decode phase beyond the prefill-selected state, or MQA/GQA KV layout), while a same-method same-budget per-condition oracle exceeds the fixed setting by >10% in quality or >15% in TPOT, compared with baseline B = full-KV cache on the identical task.

# Sub-Hypotheses

三个子假设与 RQ-2 的三个 observable consequence 一一对应；H1 单独不能区分"质量失效"与"性能失效"，H1+H2 不能把失效归因于"静态设置"而非"方法本身"，因此三者都需要。

## H1 — 固定预算在原任务上安全、在非原任务/阶段上越界

同一方法 X 以其公开发表预算运行时，在至少一个非原生条件（任务类别、generation phase 或 KV layout）上，主质量指标相对 full-KV 基线下降 ≥ 预声明容差（主指标相对下降 ≥10%；PPL 类指标相对上升 ≥15%），而其原生任务上的下降 < 容差。

- **统一 native 定义（revision R1）**："原生任务" = 该论文实际验证的任务套件，**逐字包含论文报告过的所有任务/模型/layout 组合**：SnapKV 的原生套件 = LongBench 16（**含 GovReport/QMSum/MultiNews 等 summarization**）+ Needle + Command-R RAG；KIVI 的原生套件 = Llama-2-7B/13B + **Falcon-7B（MQA）** + Mistral-7B 上的 CoQA/TruthfulQA/GSM8K + LongBench 8（含 MultiNews/TREC/SAMSum）。**套件内条件上的质量下降（SnapKV GovReport、KIVI Falcon 等）属于"发表点有效性"观察，永远不能作为 H1 证据**。
- **非原生条件（H1 唯一合法证据来源）**：SnapKV → ∞Bench En.Sum（套件外长文 summarization）+ GSM8K 8-shot CoT（套件外 reasoning）；KIVI → ∞Bench En.Sum + long-decode generation-phase 单元（output ≥1024，prefill-selected 状态之外的 decode 阶段）。因 H2O 原生套件已含短 summarization，其非原生 summarization 条件必须使用长文 summarization（GovReport 类、∞Bench En.Sum）。
- 可证伪阈值：**H1 成立 iff 至少一个非原生条件出现 ≥10% 相对质量下降，且同方法原生任务 <10%**（原生含其套件内 summarization/layout 单元）。

## H2 — 内存节省最大化的设置引入 TPOT/kernel 开销

在固定内存预算、匹配 batch 下，使内存节省最大化的静态设置（低 bit 量化 + fused dequant、decode-phase 选择、周期性重压缩）在至少一个条件下（长 decode、MQA/GQA KV layout、decode 阶段选择）使 TPOT P50 相对 full-KV 基线上升 ≥15%，或单步 kernel 开销（dequant/selection/recompression）> 解码步时间的 10%，或"内存节省无法转化为吞吐"：measured HBM 节省 ≥2× 而吞吐优势 <5%。

- 覆盖 RQ-2 第二个 observable consequence：SCOPE 的 Slide 变体（18.28 vs 36.57 tok/s at batch 8，SCOPE.md Table3）、GEAR 的 n_b=20 周期重压缩（GEAR.md §4）、KIVI 的 fused dequant 每步成本（KIVI.md §3.3）都是候选机制；在 MQA layout 上 KV 本已很小，dequant 固定开销占比上升。
- 可证伪阈值：**H2 成立 iff 至少一个条件下上述三项之一被实测超过阈值**。
- 记账说明（revision R7）："measured HBM 节省 ≥2×" 指 KV 表示本身的 measured 字节差（压缩 vs full-KV，同一 batch/context 几何下），不含模型权重（batch 1/短上下文下权重主导 peak HBM，会掩盖 KV 节省）；该腿只在 batch ≥8、上下文 ≥4K 的单元上评估。

## H3 — 同预算 per-condition oracle 显著优于固定设置

同一方法、同一实现、同一 measured HBM 预算，仅按条件重调其自有旋钮（sink 数量 4→1、group size、逐层预算分布 β、prefill/decode 预算切分、选择频率），在至少一个条件下比固定设置获得 ≥10% 相对质量提升或 ≥15% TPOT 改善。

- 排除"换方法"的混淆：oracle 不得更换方法族（GEAR 在 2-bit GSM8k 上 40.20 vs KIVI 25.25 是方法差异，不是重调）；也不得通过增大内存获益（KIVI 2→4-bit、SnapKV 1024→4096 只证明 H1 的退化存在，不证明"同预算下静态设置是问题"）。
- 可证伪阈值：**H3 成立 iff 至少一个条件下 oracle 与固定设置在质量或延迟上的差距 ≥10%/≥15%**；若所有条件上两者差距 <5%（质量和延迟均如此），H3 被杀。

# Independent Variables

最小集合（每项直接映射 RQ-2 的 failure scenario）：

1. **Task class**：retrieval QA（LongBench single/multi-doc：NrtvQA、Qasper、HotpotQA）、long-form summarization（GovReport、QMSum、∞Bench En.Sum）、chain-of-thought reasoning（GSM8k/BBH 8-shot CoT、LongGenBench-4K）。每类 ≥2 个数据集以避免单数据集伪影。
2. **Generation phase**：prefill-selected（压缩在 prefill 结束时一次性决定，decode 不更新——SnapKV/PyramidKV 的模式）vs long decode（output ≥1024 tokens、decode steps 显著多于 prefill——SCOPE 声称的解码阶段状态）。在同一个锚定任务上按 output length 512/1024/2048 分层。
3. **Model KV layout**：MHA（Llama-2-7B/13B）、GQA（Llama-3-8B-Instruct、Mistral-7B）、MQA（Falcon-7B，KIVI 已知失效点）。
4. **Memory budget**：每个方法的公开发表操作点 ±1 个扫描档（H2O 20%/40%；SnapKV 1024/4096；PyramidKV 64/2048；KIVI 2/4-bit；StreamingLLM 4 sinks；GEAR 2/4-bit），以 **measured HBM bytes** 记账而非名义百分比（防止 A8 式"uniform cost"错误）。
5. **Output length**（折叠于 phase，但对长生成类任务独立保留）：影响错误累积（GEAR.md §2: 量化误差逐 token 复合）与固定内存承诺是否可维持。

# Dependent Variables

只保留与假设直接绑定的指标：

1. **Quality per task suite**（LongBench F1/ROUGE-L/Acc、GSM8k/BBH exact match、PPL for streaming、LongGenBench Acc）——H1/H3 的主要证据，全部以同任务 full-KV 基线做配对差分。
2. **TPOT P50/P95（ms/token）**——H2 的主要证据；decode 阶段每步成本，含 dequant/selection/recompression。
3. **Kernel overhead（解码步时间中 quantize/dequant/Top-K/residual-repair 的占比）**——H2 的机制证据，把 TPOT 差异归因于 kernel 而非 batching 噪声。
4. **Peak HBM usage（measured，GB）**——验证"固定内存目标"真实相等；所有方法的名义百分比口径不同（H2O 是全部 KV 的 20%，SnapKV 只压缩 prompt KV，KIVI 还有 R=128 fp16 residual）。
5. **TTFT（s）**——次要指标，覆盖 prefill 端投票/选择开销（SnapKV/PyramidKV 的 observation-window 计算、KIVI 的量化）。

# Control Variables

- **Model 权重与基座精度**：固定模型 checkpoint 与 FP16/BF16 基座，量化只作用于 KV（GEAR/KIVI 的论文约定）；排除权重量化混淆（GEAR.md §4.2 只用 8-bit 权重做效率测试，质量实验保持 FP16）。
- **GPU**：单张 A100-80GB（所有语料方法的主要硬件，SnapKV.md §9、KIVI.md §9）；避免跨 GPU 架构比较（T11）。
- **Serving framework**：统一 HF Transformers + FlashAttention-2 eager 实现（SnapKV/StreamingLLM/KIVI/SCOPE 的原生实现环境，SCOPE.md §5/§9）；不引入 vLLM 式 paged 层（那会引入 block 粒度混淆）。
- **Decoding 策略**：全部 greedy（SnapKV/PyramidKV/SCOPE 使用 greedy，PyramidKV.md §8、SCOPE.md §8），固定 seed，消除采样随机性。
- **Request count**：每条件使用完整官方 benchmark split（LongBench 官方 4,750 例）；合成/压力条件 ≥100 请求。
- **Batch size**：固定（如 batch 8，与 SCOPE 效率测量一致），跨条件不变；TPOT 测量 ≥1000 个 decode step。
- **Tolerance 定义**：预声明——质量相对下降 ≥10%（PPL 相对上升 ≥15%）即"越过容差"；TPOT 上升 ≥15%；oracle gap 阈值 >10%（质量）/>15%（延迟）。任何事后调整都判为作弊。
- **Evaluation harness**：官方脚本（LongBench、lm-eval-harness、LongGenBench、∞Bench），每任务的 metric 与其论文一致（F1/ROUGE-L/Acc/Edit Sim）。

# Confounders

1. **任务难度基线差异**：跨任务绝对分数不同，质量下降可能只是任务难。检测/中和：所有质量证据采用**同任务内配对差分**（compressed vs full-KV on the same task），永不比较跨任务绝对分；H1 的容差只定义在配对差分上。
2. **Output length 差异**：长输出 → 更多 decode 步 → 更多自回归误差复合（GEAR.md §2）与更多 KV 增长。中和：跨任务类匹配生成长度（LongBench 官方 gen 256；SnapKV 速度基准 gen 512）；另在锚定任务上把 output length 作为 IV 分层扫描，使 phase 效应与任务效应可分离。
3. **Greedy vs sampling 随机性**：全部 greedy + 固定 seed；在 KIVI Falcon 单元（自定义 CUDA/Triton kernel，实际存在非确定性风险的单元）上跑 ≥3 seeds 报告方差，确认确定性指标无 seed 敏感性（revision R5：原方案把检查放在纯 torch 的 SnapKV 单元上，度量的是不存在的方差）。
4. **Metric floor/ceiling 效应**：MultiNews ROUGE 基线仅 3.51（KIVI.md Table4），TREC 接近饱和（90+），小基数指标的相对下降会被放大。检测/中和：预注册排除规则——基线 <5 或 >95 的指标不进入相对下降统计，同时报告绝对值与相对值；**每个单元的报告值须附 bootstrap CI（≥1000 resamples），落在 ±2 metric-point 非判别带内的 crossing 判为不可判别（revision R5）**。
5. **TPOT batching 噪声**：decode TPOT 对 batch 组成敏感。中和：固定 batch、跨条件使用相同请求集、P50/P95 基于 ≥1000 步、条件交错测量；若差异只在 batch 1 出现则视为噪声（见 Ambiguous Outcome）。
6. **内存记账口径不一致**："20%"在方法间含义不同（H2O = 全部 KV 的 20%；SnapKV = prompt KV 的 8%；KIVI = 2-bit + R=128 fp16 residual）。中和：用 nvidia-smi/pytorch profiler 直接测 peak HBM，按 bytes 匹配预算，不按名义百分比。
7. **Instruction 位置与模板**：observation-window 有效性依赖 instruction 位于末尾（A12；SnapKV 的 position-invariance 只在 QMSum/Openreview/SPACE 上验证过，SnapKV.md §4.2）。中和：跨条件固定 prompt 模板；对 summarization 条件刻意使用"summarize the whole document"的 instruction-at-front 模板（T8 的 breakdown condition），并在模板维度上报告差异。

# Expected Observation If True

- **H1 true**：matched harness 中至少一个非原生（套件外）条件出现 ≥10% 相对下降而原生任务 <10%。语料一致的锚点预测：SnapKV 1024 在 ∞Bench En.Sum 类长文 summarization 上越界（语料 GovReport 观察值 −29% 提示方向，但套件内观察不作 H1 证据）；KIVI 2-bit 在长 decode 阶段（output ≥1024）上越界，或 ∞Bench En.Sum 上越界（语料 Falcon −25%/−38.9% 提示方向，但套件内观察不作 H1 证据）；对 StreamingLLM 预测：在需中段信息检索的任务（needle 位于 recent window 之外）上质量越界，PPL 类指标相对上升 ≥15%（E5-b 无条件测量）。GEAR 的语料位置（revision R9）：其原生 reasoning 近无损（avg 40.20 vs FP16 40.52 = −0.8%），LongBench 平均 −5.0% 为边界值 → H1 相关证据只能在 per-task LongBench 解析（E5-c）中寻找，不预设其越界。
- **H2 true**：长 decode 条件下 TPOT P50 ≥15% 高于 full-KV 基线（语料锚点：SCOPE Slide 18.28 vs Full 36.57 tok/s，吞吐 −50% 即 TPOT +100%）；GEAR 在 n_b=20 重压缩步出现周期性 TPOT 尖峰；KIVI 在 Falcon-7B（小 KV）上 dequant 开销占比升高、内存节省 2× 但吞吐优势 <5%。
- **H3 true**：同预算重调至少在一个条件上 ≥10% 质量或 ≥15% 延迟改善。语料锚点：SCOPE Table2 的 phase-split 重调（同 35% 总预算下 prefill-only 27.75 → prefill+decode 52.17 on GSM8K+，+88% 相对）；选择频率重调 Discontinuous 25.92 vs Slide 18.28 tok/s（+42%）；StreamingLLM 在 Falcon 上 sinks 4→1（更少内存，同样 PPL 12.12，StreamingLLM.md Table2）；PyramidKV β 重调（同总预算的逐层再分配，语料已示 α/β 敏感性，PyramidKV.md Appendix I）。

# Falsification Condition

- **H1 被杀，当且仅当**：在 matched harness 中，没有任何非原生条件在方法自己的公开发表预算下出现超过预声明容差的相对下降；或者所有条件（含原生任务）同样下降 ≥10%。前者证明静态设置跨条件稳健；后者（universal fragility，revision R6 统一口径）同样判为 **H1 falsified**——问题属于"发表点全局不安全"，机制归因（任务/阶段/layout 变化导致失效）不成立，但其结果仍记录为"静态设置全局不安全"的有价值发现。**明确表述：H1 falsified if no out-of-suite condition shows a drop beyond the predeclared tolerance at the method's own published budget while its native-task delta stays below the tolerance, OR if native and non-native conditions drop ≥10% alike.**
- **H2 被杀，当且仅当**：在匹配 batch ≥8 与匹配 measured HBM 下，所有条件上固定设置的 TPOT P50 与 full-KV 基线差距 <15%，且 kernel overhead 占比 <10% 且从未出现"≥2× 内存节省但 <5% 吞吐优势"的配置（KV-表示字节口径，revision R7）。若 TPOT 差异只在 batch 1 出现、batch ≥8 消失，H2 同样被杀（batching 噪声而非 kernel 开销）。**明确表述：H2 falsified if no condition shows ≥15% TPOT degradation, ≥10% kernel-share overhead, or a ≥2× memory saving that fails to convert into ≥5% throughput gain at matched batch.**
- **H3 被杀，当且仅当**：在所有条件上，per-condition oracle 的最佳设置与固定设置的质量差距 <5% 且延迟差距 <5%（质量和延迟同时小于 5%）；或 oracle 的收益只能通过更换方法族或增大内存预算获得。**明确表述：H3 falsified if the oracle's best setting differs from the fixed setting by <5% on both quality and latency on every condition.**

# Ambiguous Outcome

- **所有任务类别均匀下降 ≥10%**：**按 revision R6 直接判为 H1 falsified**（universal fragility：发表点全局不安全；任务/阶段/layout 变化不是失效机制）。该结果仍有价值（静态设置不安全），但机制归因不成立，不得表述为 H1 支持。
- **恰好一个任务失败**：单数据集越界不足以支撑 H1——可能是 metric floor/ceiling（如 MultiNews 3.51 基数）、数据集伪影或单任务敏感性。要求 ≥2 个任务类别或每类 ≥2 个数据集同时越界；任何 crossing 判定须带 bootstrap CI，落在 pre-registered ±2-point 非判别带内的视为不可判别（revision R5）。
- **TPOT 差异只在 batch 1 出现**：批量噪声而非 kernel 开销，H2 不成立。
- **质量只在低于发表预算处下降**：发表操作点自身安全，静态设置脆弱性未在"固定内存目标"上被证明（RQ-2 要求"at a fixed memory target"）。
- **H2O 在其原生短 summarization（XSUM/CNN-Daily Mail）上下降**：该套件属于其发表范围，此结果攻击的是"发表点有效性"而非"跨任务迁移"；非原生 summarization 必须以长文形式（GovReport/∞Bench En.Sum）单独测试，二者不可混用。
- **oracle 收益只出现在预算增大时**（KIVI 2→4-bit、SnapKV 1024→4096）：只确认 H1 的退化存在，不能支持 H3 的"同预算静态设置为问题"声明；此时 H3 判为未证实而非证伪，需在同预算重调旋钮上重测。