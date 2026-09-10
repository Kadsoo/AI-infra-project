# Paper Metadata

- **Title:** Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention [PAPER FACT]
- **Authors:** Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo [PAPER FACT] — Affiliations: National University of Singapore, Shanghai Jiao Tong University, Huawei Cloud [PAPER FACT]
- **Venue:** arXiv:2403.19708 [cs.CL] v1 23 Mar 2024, v3 30 Jun 2024 (807 KB) [PAPER FACT]; Accepted to USENIX Annual Technical Conference (ATC) 2024 [PAPER FACT]; CC BY 4.0 [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2403.19708 doi:10.48550/arXiv.2403.19708 [PAPER FACT]; HTML https://arxiv.org/html/2403.19708v3 , PDF https://arxiv.org/pdf/2403.19708 [PAPER FACT]
- **Code:** N/A（论文实现于 PyTorch + Python，集成 LLaMA/Falcon 基于 Transformers，未披露公开仓库）[PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2403.19708v3 + https://arxiv.org/abs/2403.19708（完整 HTML 52K+ 字符，含 Fig.1-25、Table1-2、§2-6 全文）[PAPER FACT]
- **别名：** AttentionStore 为其层次化 KV 缓存系统名称 [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1. Problem [PAPER FACT]

- 多轮对话是 LLM 核心能力，ShareGPT 73% 会话为多轮，随轮数增长历史 tokens 累积 [PAPER FACT]
- 现有 serving 引擎每轮结束后丢弃 KV cache 以释放 HBM，下一轮需对全历史 tokens 重算 KV，导致高达 99% 的 prefilling 计算为重复计算 [PAPER FACT]；轮次越多重复占比越高，新轮次 >99% tokens 为历史 [PAPER FACT]
- 单纯将历史 tokens 保留在 HBM 亦不可行：LLaMA-65B 每 token 2.5 MB KV，生成速度 13.9 GB/s，4x A100 80GB 中 130GB 存模型剩余 190GB 空闲 HBM 仅 14 秒即被填满，512GB host memory <1 分钟填满 [PAPER FACT]
- 需在不丢弃历史 KV 的前提下，利用廉价大容量存储层次实现跨轮复用，同时不让慢速存储的访问阻塞推理关键路径 [PAPER FACT]

## 2. Motivation [PAPER FACT]

- 数据集统计：ShareGPT 90K 对话，73% 多轮，30% 会话 >4K tokens，47% >2K tokens（Fig.2）[PAPER FACT]；平均轮次 5.75 的 9K 会话含 52K turns [PAPER FACT]
- 重复代价量化：Mistral-7B 在 1x A100 上历史 vs 新 tokens 的 prefilling GPU 时间对比，重复部分占主导（Fig.4b）[PAPER FACT]
- KV 体积：GPT-3 每 token 4.5 MB，数千 tokens 会话即数 GB [PAPER FACT]；LLaMA-65B 2K tokens 预填充 360ms 产生 5GB KV，而 host→GPU 加载该 5GB 需 192ms（PCIe Gen4 16 lanes 有效 26 GB/s），加载开销与计算同量级不可忽略 [PAPER FACT]
- 上下文窗口溢出为常态：LLaMA-2 4K 窗口下 30% 会话溢出，OPT 2K 窗口下 47% 会话溢出，溢出后传统 token truncation 使已存 KV 因位置编码错位而失效 [PAPER FACT]
- 机会：若能跨轮复用历史 KV，理论上可削减 99% prefilling 成本 [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **KV 访问阻塞：** HBM ↔ AttentionStore 经低速链路，加载必须在计算前完成、保存必须在下一任务前完成，否则阻塞 GPU；192ms/5GB host→HBM 与 360ms/2K tokens 计算相比不可隐藏 [PAPER FACT]
2. **容量-速度矛盾：** HBM 容量仅 80GB/卡，host DRAM 数百 GB，SSD 数十 TB；多数 KV 最终位于 SSD（<5 GB/s），随机到达时命中 DRAM 概率低，访问性能差 [PAPER FACT]
3. **放置决策难：** 传统 LRU/FIFO 仅利用历史访问信息，无法预知未来；需利用调度器对等待队列的先知来预取/驱逐以提升 DRAM 命中率 [PAPER FACT]
4. **位置编码耦合导致失效：** 采用绝对位置编码或在 KV 中嵌入相对位置后，截断后 token 位置改变，原 KV 位置编码失配而整体失效；每溢出一次传统方案需重算截断后剩余 tokens 的全量 KV [PAPER FACT]
5. **并发批处理干扰：** 连续批处理下新任务 prefill 阻塞正在进行的 decode，增加端到端延迟；KV 加载/保存未重叠会进一步放大该阻塞 [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**CachedAttention = 复用历史 KV 的新注意力机制 + AttentionStore 层次化缓存 + 重叠访问 + 调度器感知放置 + 位置编码解耦的截断 [PAPER FACT]**

- **AttentionStore：** 以 block 为管理粒度，跨 HBM（执行缓冲）→ host DRAM → SSD 三级扩展；利用连续批处理与按需分配提升利用率 [PAPER FACT]
- **Layer-wise Pre-loading：** 按层流水重叠加载与计算；读流在执行流计算第 i 层时并发加载第 i+1 层 KV；增设 HBM 读缓冲以消除上任务结束→首层加载间隙；当 T_load·L_hist > T_pref·L_new 时，用更大缓冲填补差距，缓冲大小 S_buf = B·(T_load·L_hist − T_pref·L_new) [PAPER FACT]；实验无缓冲即降 35%，15 层缓冲达完美重叠降 61% [PAPER FACT]
- **Asynchronous Saving：** 区分 prefill（并发产生大量 KV，逐层保留与 decode 阶段重叠）与 decode（逐 token 产生，逐层写回与解码重叠）；设 HBM 写缓冲避免解码结束时未写完而阻塞下一任务；降低保存开销 13–15% [PAPER FACT]
- **Scheduler-aware Fetching：** 利用作业调度器等待队列的先知，设前瞻预取窗口 L_pw = C_mem / S_kv；若等待任务的 KV 在 SSD 则提前预取至 DRAM；host 设缓冲保证预取不因满而阻塞 [PAPER FACT]
- **Scheduler-aware Eviction：** 驱逐窗口长度 (C_mem + C_disk)/S_kv；需驱逐时在窗口内从尾向头扫描，优先驱逐靠近尾部（未来最远）的会话；会话为最小驱逐粒度（要么全用要么全不用）[PAPER FACT]；128G/10T 下总体命中 86% vs LRU 58% vs FIFO 48%，且 99.6% 命中在 DRAM（LRU/FIFO 仅 0.5–0.6% DRAM）[PAPER FACT]
- **Decoupled Positional Encoding & Truncation：** 要求使用相对位置编码 RPE（RoPE 等，LLaMA/Mistral/Falcon/T5/Transformer-XL 均用）[PAPER FACT]；保存 KV 前不嵌入位置编码，加载时再嵌入新位置，从而截断可直接在 KV 上丢弃前半段而不使剩余 KV 失效 [PAPER FACT]；溢出时直接截断 KV，避免重算；PPL 与 TT 方案差异 <0.02，朴素截断 PPL >1e3 [PAPER FACT]

## 5. System Changes [PAPER FACT]

- **架构总览（Fig.5）：** 在 GPU 执行器前增加 AttentionStore，调度器提供作业队列提示；读/写流使用独立 CUDA streams，磁盘迁移使用独立 IO 线程与执行重叠 [PAPER FACT]
- **存储管理：** 类 PagedAttention 的块式分配器按需分配/释放 host/disk 块；连续批处理使新任务 prefill 完成后立即加入 decode 批 [PAPER FACT]
- **加载路径：** 读流逐层预加载至 HBM 执行缓冲，计算流逐层消费；读缓冲大小可配置（以层数为单位 PL-B0 … PL-B15）[PAPER FACT]
- **保存路径：** 写流逐层异步写回：prefill 阶段每层自注意力产出后即保留，decode 阶段每迭代产出一 token KV 即层式写回；未写完部分暂移写缓冲 [PAPER FACT]
- **层次迁移：** 命中 DRAM 则直接预加载；否则触发从 SSD→DRAM 的调度器感知预取；DRAM 满时触发感知驱逐 DRAM→SSD，SSD 满时驱逐出系统 [PAPER FACT]
- **截断适配：** 保存时 KV 无位置编码；溢出比 0.5（丢弃最早一半 tokens）[PAPER FACT]；截断操作直接在 KV cache 上执行（Fig.12）[PAPER FACT]
- **模型集成：** 实现于 PyTorch/Python，基于 Transformers 集成 LLaMA 与 Falcon；支持 Mistral-7B 32K 上下文 [PAPER FACT]

## 6. Target Metrics [PAPER FACT]

- **Primary：**
  - **Cache hit rate** 总体/DRAM/SSD 分别统计 [PAPER FACT]
  - **TTFT (time to first token)** 端到端首 token 延迟 [PAPER FACT]
  - **Prefilling throughput** prompt 处理吞吐（tokens/s）[PAPER FACT]
  - **End-to-end GPU time** 完成全部推理任务的 GPU 时间 [PAPER FACT]
  - **Inference cost** 基于 AWS EC2 按需定价：$5/hour/A100，$0.0088/hour/GB DRAM，$0.000082/hour/GB SSD [PAPER FACT]
  - **Perplexity (PPL)** 与下游准确率（MMLU/LongEval/PIQA）用于验证解耦位置编码的正确性 [PAPER FACT]
- **Secondary：**
  - Layer-wise 预加载与异步保存的开销削减比例 [PAPER FACT]
  - 不同存储配置 (128G/2T vs 128G/10T, HBM-only vs HBM+DRAM vs HBM+DRAM+SSD) 下的命中与时间 [PAPER FACT]
  - 缓存容量比 RCC/CCpUT 与命中/吞吐曲线 [PAPER FACT]
  - 会话到达率 0.5–2.0/s 的敏感度 [PAPER FACT]
  - 上下文溢出对命中与 GPU 时间的影响（CA vs OF baseline）[PAPER FACT]

## 7. Baselines [PAPER FACT]

- **RE (Re-computation)：** 仅保留历史 tokens 文本，每轮结束丢弃 KV，下一轮用历史文本重算全量 KV；溢出时采用 token truncation（丢最早一半）并重算剩余 [PAPER FACT]
- **OF (Overflow with coupled PE)：** 位置编码耦合在 KV 中的朴素方案，溢出即失效需重算，用于对比解耦收益 [PAPER FACT]
- **NKVT (Naive KV Truncation)：** 直接截断耦合位置编码的 KV，用于展示 PPL/准确率崩溃 [PAPER FACT]
- **TT (Token Truncation)：** 移除历史 tokens 后重算剩余 KV，质量上界但高代价 [PAPER FACT]
- **NO-PL / PL-B0 … PL-B15：** 有无 layer-wise 预加载及不同读缓冲大小的消融 [PAPER FACT]
- **LRU / FIFO：** 传统驱逐策略对比调度器感知策略 [PAPER FACT]
- **HBM-only / HBM+DRAM / HBM+DRAM+SSD：** 不同层次缓存容量对比 [PAPER FACT]
- **提及但非数值对比的并发工作：** LMDeploy (HBM 缓存多轮)、RadixAttention、ChunkAttention、Pensieve（GPU+CPU）[PAPER FACT]

## 8. Workloads [PAPER FACT]

- **Datasets：** ShareGPT 原始 90K 对话 [PAPER FACT]；实验取 9K 会话，按 Poisson 到达生成时间戳，λ=1.0 会话/秒 [PAPER FACT]；平均轮次 5.75，总 turns ~52K，预热 10K turns，评估后 42K turns [PAPER FACT]；PPL 用 WikiText-2、C4、PTB；准确率用 MMLU、LongEval、PIQA [PAPER FACT]
- **Models：** 主评测 LLaMA-1 65B、LLaMA-2 13B/70B、Falcon 40B [PAPER FACT]；中间激活 FP16 [PAPER FACT]；渗漏实验用 LLaMA-7B/13B (PPL) 与 Mistral-7B 32K（长上下文）[PAPER FACT]
- **执行配置：** LLaMA-13B 默认 2 GPUs batch 24，LLaMA-65B/70B 与 Falcon-40B 默认 4 GPUs batch 24 [PAPER FACT]；消融中 LLaMA-13B 单 GPU batch 16 测 1K tokens 不同历史/新 token 配比（900/100 等）[PAPER FACT]；prompt 长度 1K–1.6K、解码 20 步等 [PAPER FACT]
- **到达与上下文：** 截断比 0.5 [PAPER FACT]；TTL 1 小时用于容量需求实验 [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **Testbed：** 4x NVIDIA A100 80GB HBM [PAPER FACT]；128GB DRAM，10TB SSD [PAPER FACT]；PCIe Gen4 16 lanes 有效 26 GB/s [PAPER FACT]
- **GPU 配置：** 见 §8 的 2/4 GPUs 分配；持续批处理开启 [PAPER FACT]
- **存储：** Host memory 与 SSD 以块管理，独立分配器 [PAPER FACT]；对比实验中 HBM cache 配 10GB，DRAM 128GB，SSD 10TB [PAPER FACT]；容量对比含 128G/2T 与 128G/10T 两种 [PAPER FACT]
- **软件：** PyTorch + Python，Transformers 集成，专用 CUDA streams 搬运，独立 IO 线程磁盘迁移 [PAPER FACT]
- **成本模型：** AWS EC2 p4d 按需价（见 §6）[PAPER FACT]
- **精度：** FP16 [PAPER FACT]

## 10. Main Results [PAPER FACT]

- **端到端（9K 会话 42K turns 评估）：**
  - **Cache hit rate：** CA 86% (LLaMA-13B)、71% (65B)、89% (70B)、90% (Falcon-40B) [PAPER FACT]；65B 较低因每 token 2.5MB（13B 0.78MB，70B 0.31MB GQA 8，Falcon 0.12MB GQA 16）同容量容纳会话数少 [PAPER FACT]
  - **TTFT：** 相对 RE 降低 85% (13B)、61% (65B)、87% (70B)、86% (Falcon) [PAPER FACT]；命中时 TTFT 仅取决于新输入 tokens 数 [PAPER FACT]
  - **Prefilling throughput：** 加速 6.8x (13B)、2.6x (65B)、7.8x (70B)、7.2x (Falcon) [PAPER FACT]
  - **End-to-end GPU time：** 加速 4.0x / 1.9x / 3.3x / 3.4x（同序）[PAPER FACT]；收益来自历史重算消除与溢出重算避免，且 prefill 缩短亦缩短 decode 阻塞时间 [PAPER FACT]
  - **Inference cost：** 节省 70% (13B)、43% (65B)、66% (70B)、68% (Falcon) [PAPER FACT]；存储成本占比 16.4% / 9.0% / 9.0% / 9.0% [PAPER FACT]
- **重叠访问消融（LLaMA-13B 1x A100 batch16）：**
  - 无缓冲 layer-wise 预加载即降 prefilling 35%，PL-B15 完美重叠降 61%（历史 1K/新 100）[PAPER FACT]
  - 异步保存使整体执行降低 13–15%（prompt 1K–1.6K + 解码 20 步）[PAPER FACT]
  - 不同历史/新配比（600/400→900/100）下 CA 均优于 RE，且缓冲可隐藏加载>计算的场景 [PAPER FACT]
- **调度器感知放置（128G/2T）：** CA 总命中超 LRU 27%、超 FIFO 31%；128G/10T 时 CA 86% vs LRU 58% vs FIFO 48% [PAPER FACT]；LRU/FIFO 的 DRAM 命中仅 0.5–0.6%，其余为 SSD，CA 99.6% 命中在 DRAM [PAPER FACT]；GPU 时间加速至多 2.7x [PAPER FACT]
- **解耦截断（128G/10T）：** CA vs OF，命中下降 OF 降低 17.6% (13B)、41.5% (65B)、18.1% (70B)、18.4% (Falcon) [PAPER FACT]；65B 因 2K 窗口更小受溢出影响最大 [PAPER FACT]
- **正确性：** PPL CA vs TT 差异 <0.02（如 LLaMA-7B WikiText-2 5.47 vs 5.48），NKVT >1e3 崩溃 [PAPER FACT]；MMLU/LongEval/PIQA 准确率 CA 与 TT 相当（13B MMLU 52.3% vs 53.2%），NKVT 仅 29.6% 等 [PAPER FACT]（Table1/2）
- **容量需求：** TTL 1h，RCC/CCpUT=0.1 时命中 51%，0.25 时 98% 达峰，吞吐亦达峰 [PAPER FACT]
- **存储介质对比（HBM 10GB / DRAM 128GB / SSD 10TB）：** HBM-only 命中 ~0%，HBM+DRAM 仅 3.4% (13B)/1.7% (65B)/7.7% (70B)/19.1% (Falcon)，HBM+DRAM+SSD 达 86%/71%/89%/90% [PAPER FACT]
- **到达率敏感：** 0.5→2.0 会话/秒，命中 82%→77%，TTFT 0.122s→0.154s，prefill 858K/s→681K/s，GPU 6.25H→7.01H，变化平缓 [PAPER FACT]

## 11. Assumptions [PAPER FACT]

- 采用 RPE（RoPE 等）而非 APE，位置可解耦；广泛用于现代 LLM [PAPER FACT]
- 多轮对话中历史 KV 要么全用要么全不用，故会话为最小驱逐/预取粒度 [PAPER FACT]
- 块式存储管理与连续批处理已启用，调度器可提供等待队列先知 [PAPER FACT]
- 溢出策略固定截断最早一半 tokens（ratio 0.5）[PAPER FACT]
- PCIe 有效 26 GB/s、SSD <5 GB/s、DRAM 数十 GB/s 的带宽假设 [PAPER FACT]
- ShareGPT 的 Poisson 到达与真实流量近似 [PAPER FACT]
- TTL 1h 内的 DSpUT 统计可代表容量规划需求 [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 结论未设独立 Limitations 小节，局限散见于评估与未来工作；本文归纳为作者陈述 [PAPER FACT]
- 65B 因更大每 token KV (2.5MB) 与更小上下文窗口 (2K) 导致命中与收益显著低于其他模型，提示对大 KV/小窗口模型需更大容量 [PAPER FACT]
- 仅在 4x A100 + 128GB/10TB 单机层次评估，未验证跨机分布式 AttentionStore 与更大集群 [PAPER FACT]
- 仅评估至 70B 规模，未覆盖 MoE（如 Mixtral）与更长上下文（>32K） [PAPER FACT]
- 成本模型基于 AWS 按需价，未含网络与运维成本 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- 未评估加密/多租户隔离与隐私：跨会话 KV 明文存 DRAM/SSD 存在泄露风险，未讨论访问控制 [AGENT INFERENCE]
- 调度器先知依赖理想队列可见性，实际在线 serving 中到达突发与优先级抢占可能使预取窗口失配 [AGENT INFERENCE]
- 解耦 RPE 对 RoPE 外的其他 RPE 变体（ALiBi、相对偏置）兼容性未系统验证 [AGENT INFERENCE]
- 块大小与缓冲层数需手工调优，未提供自适应算法；不同模型 GQA 因子差异未自动适配 [AGENT INFERENCE]
- 未与 KV 压缩（量化/稀疏）联合，HBM 节省与 AttentionStore 容量增益可正交叠加但未量化 [AGENT INFERENCE]
- 单点故障与持久化：SSD 上的 KV 未讨论崩溃一致性与恢复 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 能否将调度器感知策略与学习式预取结合，利用历史访问模式预测而非仅窗口扫描？ [AGENT INFERENCE]
2. 在 PD 解耦（Splitwise/DistServe/Mooncake）与多机部署下，AttentionStore 如何与 RDMA/NIXL 零拷贝传输集成以避免 PCIe 瓶颈？ [AGENT INFERENCE]
3. 如何自适应选择 TTL 与 RCC/CCpUT 比，以在成本与命中间动态权衡？ [AGENT INFERENCE]
4. 对于超长上下文（>32K）与窗口溢出频繁场景，0.5 截断是否为最优，是否可学习截断策略？ [AGENT INFERENCE]
5. 能否与 H2O/SnapKV 等重要性稀疏化结合，减少每 token KV 体积从而容纳更多会话？ [AGENT INFERENCE]
6. 多租户公平性：如何避免热点会话长期占据 DRAM 而饿死长尾？ [AGENT INFERENCE]
7. 如何形式化证明解耦位置编码在所有 RPE 变体下的等价性与数值稳定性？ [AGENT INFERENCE]
8. 能否将 AttentionStore 扩展为跨地域共享的分布式缓存，并保证一致性与低延迟？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **vLLM (Kwon et al. SOSP 23) PagedAttention**：块式显存管理，本文块管理灵感来源 [PAPER FACT]
- **Orca (Yu et al. OSDI 22) / Sarathi-Serve**：连续批处理与 chunked prefill，本文启用 [PAPER FACT]
- **LMDeploy / RadixAttention (Zheng et al. 2312.07104) / ChunkAttention (Ye et al. 2402.15220) / Pensieve (Yu & Li 2312.05516)**：并发的多轮/前缀共享 KV 复用，与本文同期的层次化/树结构方案 [PAPER FACT]
- **FlexGen / DeepSpeed-Inference / Lina / PowerInfer / FastServe**：参数与 KV 的 DRAM/SSD offloading，本文 KV offloading 的正交工作 [PAPER FACT]
- **CacheGen (Liu et al. SIGCOMM 24) / H2O / Scissorhands / Gear / KIVI / KVQuant**：KV 压缩与传输时压缩，可与 AttentionStore 叠加 [AGENT INFERENCE]
- **Mooncake (Qin et al. 2407.00079) / Splitwise / DistServe / TetriInfer**：PD 解耦与分布式 KV 池，AttentionStore 的分布式扩展参考 [AGENT INFERENCE]
- **LMCache (Liu et al. 2510.09665) / RAGCache / CacheBlend**：企业级/ RAG 场景的层次化 KV 缓存与融合 [AGENT INFERENCE]

---
## Review Log

Reviewer: Paper Reader — Supplement (Coverage Gaps) — 2026-08-27
Scope: webfetch arXiv:2403.19708v3 全文（含 Fig.18-25, Table1-2, 4.3.1-4.3.8 消融）逐节抽查
Webfetch verification: hit rate 86/71/89/90、TTFT -85/-61/-87/-86%、throughput 6.8/2.6/7.8/7.2x、GPU 4.0/1.9/3.3/3.4x、cost -70/-43/-66/-68% 已核；PL-B15 -61%、async -13-15%、128G/10T 86 vs 58 vs 48、99.6% DRAM、RCC 0.1→51% 0.25→98%、HBM-only ~0% 已核；PPL 5.47 vs 2198.7、MMLU 43.7 vs 21.8 已核
Problems Found: 无新增幻觉；部分硬件细节以 [NOT REPORTED] 标注已补
Confidence: High
