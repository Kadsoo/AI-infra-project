# Paper Metadata

- **Title:** Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs [PAPER FACT]
- **Authors:** Hyungwoo Lee, Kihyun Kim, Jinwoo Kim, Jungmin So, Myung-Hoon Cha, Hong-Yeon Kim, James J. Kim, Youngjae Kim (corresponding) [PAPER FACT] — (1) Sogang University, (2) ETRI, (3) Soteria Inc [PAPER FACT]
- **Venue:** arXiv:2504.11765 [cs.AI] v1 16 Apr 2025 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2504.11765 / https://arxiv.org/abs/2504.11765 / HTML https://arxiv.org/html/2504.11765v1 [PAPER FACT]
- **Code:** [NOT REPORTED] [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2504.11765 + https://arxiv.org/html/2504.11765v1 [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1 Problem [PAPER FACT]

- RAG 通过检索外部知识增强 LLM，但显著增加输入 token 数，prefill 阶段计算开销 O(L·N²·D) 膨胀（L 层数，N 输入长度含检索上下文，D 隐维）导致 TTFT 显著拉长与吞吐下降 [PAPER FACT]
- 每 query 的 prompt 结构为 Document: (retrieved texts) + Query + Answer:，检索文档越多/越长则自注意力二次复杂度越突出，且大模型进一步放大成本 [PAPER FACT]
- 多实例 RAG 服务环境下各实例独立推理、无法天然共享他实例已算 KV，冗余计算严重；同时受 GPU/CPU 内存容量限制，现有内存级缓存难以承载大量高频文档 [PAPER FACT]
- 现有工作 RAGCache（GPU/CPU 多级缓存）受容量限制，TurboRAG（盘基离线预计算）未显式支持多实例/多主机共享，难以规模化 [PAPER FACT]

## 2 Motivation [PAPER FACT]

- **RAG 的 TTFT 瓶颈在 prefill：** 需为所有输入 token 计算 Q/K/V 及其 KV，检索扩展后 N² 项主导时延 [PAPER FACT]
- **Query locality 存在：** 实测 HotpotQA/SQuAD/TriviaQA 上，用 FAISS + all-MiniLM-L6-v2 embeddings 内积检索 top-1，50% queries 仅需 22.9% / 3.1% / 31.4% 的最热点文档即可覆盖（CDF Fig.1）[PAPER FACT]；Introduction 概括为 3.1-31.39% 文档覆盖 50% queries [PAPER FACT]——小子集缓存可服务大比例请求
- **文档静态性：** 外部知识库建成后更新不频繁，适合离线预计算并持久化 KV，减少重复 prefill [PAPER FACT]
- **多实例排队机会：** 服务负载升高时超出单实例容量的请求会排队，QPS 超过处理能力后 queue wait 指数增长（Fig.2 单 Llama-3.2-1B 单 GPU 测量：1-2 qps 时 LLM 处理 >70%，超阈值后排队主导），可利用等待期的空闲 GPU/CPU 预取并生成下一请求的文档 KV，令 GPU 聚焦纯推理 [PAPER FACT]
- **盘基持久化可突破容量墙：** 磁盘可 semi-permanent 容纳随参数与上下文增长而膨胀的 KV，克服 GPU/CPU 内存约束，并跨实例/主机共享 [PAPER FACT]

## 3 Bottleneck [PAPER FACT]

1. **Prefill 计算复杂度二次方：** O(L·N²·D) 随 N 快速上升，成为 TTFT 主因 [PAPER FACT]
2. **容量墙：** RAGCache 等内存缓存受 GPU/CPU 容量限制，难以缓存大规模高频文档 [PAPER FACT]
3. **实例隔离：** 多实例环境各实例独立，无法直接复用他实例 KV，需共享存储/内存机制 [PAPER FACT]
4. **盘 I/O 延迟：** 盘基若无层次化与预取则读延迟高；需 in-memory (CPU RAM) 热点缓存缓解 [PAPER FACT]
5. **排队延迟随负载指数增长：** 高 QPS 下等待时间主导端到端时延，若不利用则资源浪费 [PAPER FACT]
6. **RAG 检索分布偏态但 workload 相关：** 需高效向量库 + KV 联合管理，避免检索与缓存分离导致的二次开销 [PAPER FACT]

## 4 Core Idea [PAPER FACT]

**盘基 KV 缓存 + 向量库一体化 + 多实例共享与等待期主动预生成（Shared RAG-DCache）[PAPER FACT]**

- **RAG-DCache（单实例）：** 为向量库中每个 document chunk 预计算 chunked-document KV cache，并以 (embedding, doc ID, text, KV cache) 四元组持久化于 disk-resident 向量库；推理时直接复用检索文档的预计算 KV，跳过对文档文本的完整 prefill，仅对 query 部分及其与文档的交互做必要计算 [PAPER FACT]；可离线用空闲硬件生成，也可命中未缓存时 on-the-fly 生成并回写 [PAPER FACT]
- **Shared RAG-DCache（多实例扩展）：** 多 LLM 实例共享同一 disk KV，并引入 KV Cache Generator 在请求排队等待超过阈值时主动预取/预生成下一批查询相关文档的 KV，完成后置于共享盘供任意实例消费；从而将原本空等的 queue time 转化为有用预计算 [PAPER FACT]
- **三组件：** KV Cache Manager（离线/在线生成、管理、落盘与检索，带 CPU RAM 热点缓存以减盘 I/O）、KV Cache Generator（等待期预取，特别在多实例下利用空闲 GPU/CPU）、Prompt Generator / RAG Processor（将检索到的 KV 与用户 query 拼接为最终 prompt，使 LLM 直连复用缓存）[PAPER FACT]
- **与现有区分：** 强调持久化盘存储 + 多实例/多主机共享，利用文档低频更新特性实现跨实例持久复用 [PAPER FACT]

## 5 System Changes [PAPER FACT]

- **Integrated Vector Database：** 传统向量库存 (embedding, ID, text)，扩展为含预计算 KV cache 的四元组；文档静态则离线生成，新文档按需生成追加 [PAPER FACT]
- **KV Cache Manager：** 负责 KV 的生成、落盘、检索与热点管理；引入 CPU RAM in-memory 缓存热点/近期 KV，减少重复盘读；作为慢速持久存储与系统其余部分的接口，优化生成与查找 [PAPER FACT]（Sec III-A）
- **KV Cache Generator（Shared 版新增）：** 监控推理服务队列，超阈值等待的请求触发预取；可调度至空闲 GPU 或 CPU 执行增量 prefill/KV 生成，避免与主推理批竞争 [PAPER FACT]
- **Prompt Generator / RAG Processor：** 接收 query，先对向量库做相似度检索得 top-k 文档 ID，再向 Manager 请求对应 KV，拼接 query KV 构造最终 LLM prompt [PAPER FACT]
- **端到端流程：** 1) query 嵌入 -> 2) FAISS 相似度检索 top-k -> 3) Manager 盘/内存取 KV（miss 则生成） -> 4) 拼接 query -> 5) LLM 推理（可跳过文档部分的完整 prefill） -> 6) 新生成 KV 可持续回写 [PAPER FACT]
- **实现细节：** 原型在单主机 2 GPU +1 CPU 上演示；FAISS 向量库 + all-MiniLM-L6-v2 embeddings；disk + CPU RAM 两级 [PAPER FACT]

## 6 Target Metrics [PAPER FACT]

- **Primary：**
  - **TTFT (Time-To-First-Token)** 平均值，反映 prefill 加速 [PAPER FACT]
  - **Throughput（requests/s 或 tokens/s）** [PAPER FACT]
  - **End-to-end latency** 分解为 queue wait + LLM processing + network（客户-服务器通信）[PAPER FACT] Fig.2
  - **Latency reduction & Throughput increase 百分比**（ headline 如 throughput +15~71%，latency 最高 -65%）[PAPER FACT]
- **Secondary：**
  - RAG-DCache 单实例下 TTFT 降幅 vs 模型大小/批量大小 [PAPER FACT]
  - 不同资源配置下的共享收益 [PAPER FACT]
  - Queue wait 占比随 QPS 变化 [PAPER FACT]
- **Not elaborate：** 精度/召回对缓存新鲜度的影响、盘空间占用、成本 [NOT REPORTED]

## 7 Baselines [PAPER FACT]

- **Baseline (no cache / recompute)：** 每次对检索文档全量重算 KV（隐含对比）[PAPER FACT]
- **RAGCache (Jin et al.)：** GPU+CPU 多级动态缓存中间 KV，显著降时延但受容量限制 [PAPER FACT]
- **TurboRAG：** 盘基离线预计算缓存，大幅降 prefill 但未显式支持多实例/多主机共享 [PAPER FACT]
- **RAG-DCache (本文单实例盘基)：** 作为 Shared 版的消融对比，验证单机盘基收益 [PAPER FACT]
- **Shared RAG-DCache（本文完整多实例共享+预取）：** 在不同资源配置下的配置搜索最优 [PAPER FACT]

## 8 Workloads [PAPER FACT]

- **验证 locality：** HotpotQA、SQuAD、TriviaQA 问答数据集，FAISS + all-MiniLM-L6-v2 嵌入，内积相似度检索 top-1 统计文档覆盖率 [PAPER FACT]
- **端到端推理：** 主要以 **SQuAD** 为代表 workload 在服务器上评测 RAG-DCache / Shared RAG-DCache [PAPER FACT]
- **Latency 分解测量：** 单 Llama-3.2-1B 在单 GPU 上测不同 QPS 下的 queue/LLM/network 分解（Fig.2）[PAPER FACT]
- **模型提及：** 评测中涉及 Llama-3.2-1B（Fig.2），另在 throughput-vs-model-size 实验中覆盖更大模型（描述为随模型增大收益更显著，具体型号枚举 [NOT REPORTED] 文中仅称 larger models）[PAPER FACT]
- **检索：**  top-1 用于 locality 统计；RAG 推理时 top-k（k 未在摘要中固定）检索文档拼接到 prompt [PAPER FACT]
- **Prompt 结构：** Document: (retrieved texts) + Query: (user question) + Answer: [PAPER FACT]

## 9 Hardware [PAPER FACT]

- **主评测平台：** 单主机配备 **2 GPUs + 1 CPU** [PAPER FACT]（Abstract/Evaluation 配置一致）
- **Latency 分解实验：** 单 GPU 跑单 Llama-3.2-1B 实例 [PAPER FACT]
- **存储：** Disk-based KV（盘）+ CPU RAM in-memory 热点缓存 [PAPER FACT]
- **细节补充：** Table I（Sec 4 Evaluation）列出详细实验设置（文中引用但具体 CPU 型号/单 GPU 型号在 HTML 截断中未明示） [PAPER FACT]；网络时延 included in breakdown [PAPER FACT]
- **GPU 型号/显存/盘类型与带宽** 在摘要/截断正文中 [NOT REPORTED] 完整 Table I 数值（预计含 GPU 型号、显存、盘带宽等） [NOT REPORTED]

## 10 Main Results [PAPER FACT]

- **Locality 量化（Fig.1 CDF）：** 覆盖 50% queries 仅需最热点文档的 **22.9% (HotpotQA)、3.1% (SQuAD)、31.4% (TriviaQA)** [PAPER FACT]
- **Queue 行为（Fig.2）：** 1-2 qps 时 LLM 处理 >70% E2E；超过处理能力后 queue wait 指数上升，引入预取动机 [PAPER FACT]
- **RAG-DCache（单实例盘基）：** **TTFT 降低约 10%-20%**，且随模型与批量增大吞吐提升更显著 [PAPER FACT]（Sec I 末段 summarry）
- **Shared RAG-DCache（多实例）：**
  - 摘要 headline：**吞吐提升 15~71%，延迟最高降低 12~65%**，取决于资源配置 [PAPER FACT]
  - 正文补充：**吞吐最高 +71%，延迟最高 -65%** 的峰值 [PAPER FACT]（与 headline 一致，区间下界 15% 与 12% 也来自摘要）
  - **随资源配置变化**：不同 GPU/CPU 分配与实例数下收益不同，需 optimal system configuration 搜索 [PAPER FACT]
- **相对 Text：** RAG-DCache vs Baseline 已有 10-20% TTFT 收益，Shared 在其上进一步放大，尤其高负载下利用等待期 [PAPER FACT]

## 11 Assumptions [PAPER FACT]

- Transformer 自回归、KV 复用有效；每步生成复用历史 KV [PAPER FACT]
- RAG 流水线：文档切块 -> 嵌入（all-MiniLM-L6-v2） -> FAISS 向量库存 embedding+ID+文本；query 同样嵌入后 top-k 检索拼接 [PAPER FACT]
- 文档数据建库后极少变更，可离线预计算並持久化 [PAPER FACT]
- LLM 推理服务多为多实例并行以应对实时并发 [PAPER FACT]
- 排队延迟在高负载下显著，可被用于预取而不干扰主推理批 [PAPER FACT]
- 文档切块的 KV 可独立复用，拼接后仍近似正确（未深入讨论 cross-attention 偏差）[PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- 评估主要在单机 2 GPU + 1 CPU + SQuAD 代表负载，未展示更大集群/多机共享与跨主机一致性实现的完整评测 [PAPER FACT]（由 TurboRAG 对比段落可推断作者强调多主机支持但实验仅单机为局限）
- 盘 I/O 虽有 CPU RAM 热点缓解，但未深入评估盘空间膨胀、写放大与新鲜度/一致性（文档更新时 KV 作废）开销 [PAPER FACT]（系统中提及可按需生成但未量化）
- 未对比 KV 压缩（量化/稀疏）与盘基的正交叠加 [PAPER FACT]（Inference: 文中未提及压缩）
- 依赖向量库嵌入质量与 top-k 检索稳定性，对 locality 分析仅基于 top-1 [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- 盘基 KV 大小与模型参数、上下文长度、层数成正比，随模型增大（如 70B）盘空间与 I/O 放大可能显著 [AGENT INFERENCE]
- 跨实例/跨主机共享需分布式文件系统或网络存储，本文未评估一致性、并发写冲突与网络传输开销 [AGENT INFERENCE]
- 预取策略对 QPS 抖动与长尾 query 敏感，阈值调优未给出自适应算法 [AGENT INFERENCE]
- 拼接复用的 correctness：独立计算的 chunk KV 忽略跨块 attention，类似 CacheBlend 指出的质量风险未被评估 [AGENT INFERENCE]
- 未与 prefix caching / RadixAttention / PagedAttention 等细粒度复用对比，收益边界不清 [AGENT INFERENCE]
- SQuAD 单数据集、短上下文问答难以代表长上下文 summarization 或多跳推理负载 [AGENT INFERENCE]
- 单机 2 GPU 原型难以推断 8+ GPU 高并发下的加速扩展性 [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. 盘基 KV 的最佳分块粒度与编码（token 级 vs chunk 级 vs page 级）如何权衡命中率、I/O 与 cross-attention 误差？[AGENT INFERENCE]
2. 文档更新时的 KV 失效与增量重算如何低开销完成？是否可与向量库版本一致性联动？[AGENT INFERENCE]
3. 多实例/多主机下最优资源配比与预取阈值的在线学习算法为何？[AGENT INFERENCE]
4. 盘基与压缩（KIVI/GEAR/量化）/稀疏化（H2O/SnapKV）联用能否进一步降 I/O 与存储？[AGENT INFERENCE]
5. 跨块 attention 偏差在 RAG 多文档拼接时是否导致质量下降？是否需要类似 CacheBlend 的 selective recompute 校正？[AGENT INFERENCE]
6. 在 PD 解耦架构中，检索 KV 的放置与传输如何与 prefill/decode 解耦协同？[AGENT INFERENCE]
7. 如何在分布式共享盘上保证多租户隔离、隐私与缓存安全？[AGENT INFERENCE]
8. 层次化存储（HBM-DRAM-SSD-对象存储）的自动分层与驱逐策略如何设计？[AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **RAGCache (Jin et al.)**：KG-tree + 层次缓存，仅前缀复用，GPU/CPU 容量受限对比基线 [PAPER FACT]
- **TurboRAG**：盘基离线预计算，大幅降 prefill 但无多实例共享 [PAPER FACT]
- **CacheBlend (Yao et al. 2405.16444)**：多块 KV 融合需校正 cross-attention，本文拼接复用可借鉴其 selective recompute 思路 [AGENT INFERENCE]
- **LMCache / Mooncake**：企业级/分布式 KV 池与 RDMA，盘基可视为其慢层扩展 [AGENT INFERENCE]
- **DeepSpeed-Inference**：KV caching 加速推理库，被引为基础 [PAPER FACT]
- **FAISS + SentenceTransformers (all-MiniLM-L6-v2)**：向量检索基座 [PAPER FACT]
- **vLLM / SGLang / DistServe**：并行 serving 与 PD 解耦系统，可与 Shared RAG-DCache 正交结合 [AGENT INFERENCE]
