# Paper Metadata

- **Title:** RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation [PAPER FACT]
- **Authors:** Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin [PAPER FACT] — 1 Peking University, 2 ByteDance Inc. [PAPER FACT]
- **Venue:** arXiv:2404.12457 [cs.DC, cs.CL, cs.LG] v1 18 Apr 2024, v2 25 Apr 2024 [PAPER FACT]; 未投期刊会议（preprint）[PAPER FACT] — verified via https://arxiv.org/html/2404.12457v2 (PGDSF 2%–75% hit gain, TTFT 1.2–4× reduction)
- **DOI/URL:** https://arxiv.org/abs/2404.12457 doi:10.48550/arXiv.2404.12457 [PAPER FACT]; HTML https://arxiv.org/html/2404.12457v2
- **Code:** N/A (原型基于 vLLM v0.3.0 + Faiss) 约 5000 行 C++/Python [PAPER FACT]; 无独立开源仓库链接 [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2404.12457v2 + https://arxiv.org/abs/2404.12457 [PAPER FACT]

## 1. Problem [PAPER FACT]

RAG 先检索 top-k 文档注入 prompt 再交 LLM 生成，提升质量但引入长序列：原始请求 100 tokens 时检索文档可达 1000 tokens，增量计算/显存 >10× [PAPER FACT]。LLM prefill 阶段需对全序列算 KV，4000 tokens 时在 A10G 上推理已达 1 秒，远超检索 ms 级 [PAPER FACT]，成为端到端瓶颈 [PAPER FACT]。现有 LLM 缓存（vLLM PagedAttention 仅单请求内复用，SGLang RadixTree 仅 GPU 内存且不感知 RAG 检索分布与文档顺序敏感性）无法跨请求复用外部知识中间态且容量受限 [PAPER FACT]。同时检索(CPU) 与推理(GPU) 串行，GPU 空闲 [PAPER FACT]。

## 2. Motivation [PAPER FACT]

- RAG 工作流：embedding → 向量库索引 → 相似度检索 top-k → 增强 prompt → LLM 生成 [PAPER FACT]
- 实测瓶颈：prefill 时延随长度急增，序列长由文档数决定；文档平均 3718 tokens (Wikipedia 0.3M 篇) 显著长于 MMLU 请求长度 [PAPER FACT]
- 优化机会：
  1) **重复性：** 多请求检索到相同文档，可共享 KV [PAPER FACT]
  2) **偏态：** 少数文档占多数请求：MMLU top 3% 文档被 60% 请求命中 (20× 均匀)；在 MMLU/Natural Questions/HotpotQA/TriviaQA 及不同 embedding/ANN (FlatL2, IVF, HNSW) 下均偏态 [PAPER FACT] (Fig.5,6)
- 缓存收益量化：缓存前缀时 prefill 仅算请求 tokens，Full prefill 比 Cached prefix 慢达 11.5×；即便计入 GPU?Host 传输，hit 仍比 miss 快 3.9× [PAPER FACT] (Fig.4)
- avg prefill 模型：Prefill Latency = MissRate*Full + (1-MissRate)*Hit [PAPER FACT]

## 3. Bottleneck [PAPER FACT]

1. **顺序敏感：** LLM 注意力使 KV 与顺序强耦合，[D1,D2] 与 [D2,D1] 的 KV 不同，且不能随意重排（影响生成质量 Chen 2023, Liu 2024）[PAPER FACT]
2. **容量与层次：** GPU HBM 小而 KV 大，需借助 Host 内存分层，但 PCIe 带宽远低于 HBM，传输开销需控 [PAPER FACT]
3. **放置策略：** 传统 LRU 未考虑大小/频率/重算成本，且不知前缀重算代价：在 [S,D1,D2,Q] 中为 D2 算 cost 时若用缓存 [S,D1] 或仅 [S] 的 cost 均不精确 [PAPER FACT] (Fig.9)
4. **请求乱序抖动：** 到达随机使同文档请求分散，导致 thrashing：容量 1 时交替 Q奇/D1 与 Q偶/D2 使命中 0%，重排后可 66% [PAPER FACT]
5. **串行流水空泡：** 检索与推理串行，GPU 在检索时空闲，端到端 = sum；但检索中 top-k 候选可能早期已稳定 [PAPER FACT]
6. **负载敏感推测：** 盲目推测生成会增加错误计算，高负载下恶化 [PAPER FACT]

## 4. Core Idea [PAPER FACT]

**RAGCache = 多级动态缓存系统 + 知识树组织 + 前缀感知替换 + 缓存感知重排 + 动态推测流水 [PAPER FACT]**

- **Knowledge Tree：** 以文档 ID 为节点的前缀树，根 S 为 system prompt，路径代表文档序列顺序；非连续存储 KV blocks (复用 vLLM PagedAttention)，多请求共享节点；检索为沿树前缀匹配 O(h) [PAPER FACT] (Fig.8)
- **PGDSF (Prefix-aware Greedy-Dual-Size-Frequency)：** 在经典 GDSF 上扩展，Priority = Clock + Frequency×Cost/Size [PAPER FACT] Eq.1；Clock 每驱逐更新为 max Priority；Cost/Size 改为对 m 次未命中请求的平均 NewSize 归一化：1/m Σ Cost_i/NewSize_i，用离线 profiling 的 T(α,β) (α cached, β 新) 双线性插值估 Cost_i [PAPER FACT] Eq.3；节点按优先级分 GPU(快)/Host(慢)/Free 三段，父在 GPU、子在 Host 保持层次，仅逐出叶节点，驱逐后父变叶则纳入候选 [PAPER FACT] (Alg.1)；Swap-out-only-once：首次逐出才 PCIe 拷 Host，后续直接释放零拷，因 Host 容量大 1-2 数量级可留一份 [PAPER FACT]
- **Cache-aware Reordering：** OrderPriority = CachedLength / ComputationLength [PAPER FACT]；优先大缓存/小计算请求，结合窗口防饥饿（最迟 window 内调度）[PAPER FACT]
- **Dynamic Speculative Pipelining：** 将检索分多段，每段结束得候选 top-k，若与上段不同则通知 LLM：若候选变则终止上推测开新推测，否则继续；最终结果一致则直接返回推测，否则重算 [PAPER FACT] (Fig.11)；定理 5.1：当 T≥d 时，最优为池空且候选变时才发推测 [PAPER FACT]；仅当 retrieved docs 变 且 pool.size < max_prefill_bs 时插入池 [PAPER FACT] (Alg.2)

## 5. System Changes [PAPER FACT]

- **RAG Controller：** 统一调度器，协调检索→cache retriever→LLM 引擎→回写缓存刷新状态 [PAPER FACT] (Fig.7)
- **Cache Retriever：** 维护知识树，支持前缀匹配查找与更新 Frequency/Cost/Clock [PAPER FACT]
- **LLM 引擎扩展：** 基于 vLLM v0.3.0 扩展 prefill kernel (PyTorch+Triton) 支持 prefix caching，兼容 MHA 与 GQA (Mistral) [PAPER FACT]
- **Pipelined Vector Search：** 改造 Faiss IVF 与 HNSW：IVF 分簇多段每段搜部分簇返回当前 top-k；HNSW 按平均搜索时间切片每片返回当前 top-k [PAPER FACT]
- **多级内存：** GPU 为一级，Host 为二级；两套逻辑 Clock 分别维护 [PAPER FACT]
- **容错：** GPU 失效时 Host 备份上层高频节点（如 S）快恢；请求超时重试：首 iteration 前失败重算，否则复用已存 KV [PAPER FACT]
- **支持模型：** Mistral-7B (0.125 MiB/token), LLaMA2-7B (0.5 MiB), Mixtral-8×7B (0.125), LLaMA2-70B (0.3125) [PAPER FACT] (Table1)

## 6. Target Metrics [PAPER FACT]

- **Primary:**
  - **Average TTFT (time to first token)** 在不同 request rate 下 [PAPER FACT]
  - **Throughput：** 满足 TTFT < 5× 最低 rate TTFT 时的最大 request rate [PAPER FACT]
- **Secondary:**
  - **Cache hit rate：** hit 文档数 / 检索文档数 [PAPER FACT]
  - 不同 host 内存大小 (8-128 GiB) 下 hit/TTFT [PAPER FACT]
  - Top-k=1/3/5 可扩展性 [PAPER FACT]
  - Non-overlapping vector search time 与推测占比 [PAPER FACT]
  - 调度时间 (tree 查找/更新+重排+推测决策) [PAPER FACT]

## 7. Baselines [PAPER FACT]

- **vLLM [Kwon 2023] + Faiss：** SOTA LLM serving，迭代调度+PagedAttention，无跨请求文档级复用 [PAPER FACT]
- **SGLang [Zheng 2023] + Faiss：** 跨请求 KV 复用前缀树，LRU，仅 GPU 内存 [PAPER FACT]
- **消融变体：** RAGCache with GDSF / LRU / LFU [PAPER FACT]；No DSP (无推测) [PAPER FACT]；无重排 [PAPER FACT]

## 8. Workloads [PAPER FACT]

- **知识库：** Wikipedia 0.3M 最热门页，平均 3718 tokens/文档 [PAPER FACT]；embedding 用 OpenAI text-embedding-3-small [PAPER FACT]；ANN 索引默认 IVF 1024 clusters [PAPER FACT]
- **数据集：**
  - **MMLU [Hendrycks 2020]：** 单 token 选择题，用文档检索分布采样 1 小时工作负载 [PAPER FACT]
  - **Natural Questions [Kwiatkowski 2019]：** 采样输出长分布均 6 tokens，99% ≤32 [PAPER FACT]
- **到来过程：** Poisson，参数为 arrival rate [PAPER FACT]
- **Top-k：** 默认 2，case study 1/3/5（top-5 时截断以 fit GPU）[PAPER FACT]
- **模型：** Mistral-7B, LLaMA2-7B 主力；可扩展 Mixtral-8×7B, LLaMA2-70B [PAPER FACT]

## 9. Hardware / Environment [PAPER FACT]

- **主测：** AWS EC2 g5.16xlarge: 64 vCPUs AMD EPYC 7R32, 256 GiB host, 25 Gbps NIC, 1× NVIDIA A10G 24 GiB via PCIe 4.0×16 [PAPER FACT]；7B 单实例 + 192 GiB host cache [PAPER FACT]
- **大模型：** 2× NVIDIA H800 80 GiB NVLink, PCIe 5.0×16, 384 GiB host cache [PAPER FACT]
- **向量库部署：** 同实例 RESTful API [PAPER FACT]
- **精度/CPU 细项：** [NOT REPORTED] 未披露具体精度 [NOT REPORTED]

## 10. Main Results [PAPER FACT]

- **Overall MMLU/NQ 7B max batch 4 (Fig.13-14)：**
  - TTFT: vs vLLM 降 1.2–4×；vs SGLang 降 1.1–3.5× [PAPER FACT]
  - Throughput: vs vLLM +30%–110% (up to 2.1×)，vs SGLang +20%–80% (up to 1.8×) [PAPER FACT]
  - Mistral 增益 > LLaMA2-7B，因后者 KV 4× 大 [PAPER FACT]
- **Top-k 可扩展 (Fig.15 MMLU Mistral-7B)：** top1/3/5 均 vs vLLM 1.7–2.1×, vs SGLang 1.2–2.5× TTFT 改善 [PAPER FACT]
- **大模型 (Fig.16)：** 低 rate 时 vs vLLM TTFT 降 1.4–2.1×；vLLM 在 >2 rps 超 SLO，RAGCache 保持 <1.4s；vs SGLang 1.2–1.6× [PAPER FACT]
- **PGDSF 消融 (Fig.17 Table2, rate 0.8 rps)：**
  - Host 8–128 GiB hit：PGDSF 比 GDSF +2%–32%，比 LRU +6%–62%，比 LFU +6%–75% [PAPER FACT]
  - TTFT: MMLU 8 GiB PGDSF 1.38s vs GDSF 1.68 LRU 1.78 LFU 1.81；128 GiB 0.83 vs 0.98/1.01/1.03 [PAPER FACT]
- **Reordering (Fig.18, 高负载)：** TTFT 降 1.2–2.1× [PAPER FACT]
- **DSP (Fig.19)：** TTFT 降 up to 1.6×；non-overlapping 搜时 降 1.5–2.3× [PAPER FACT]
- **调度开销：** MMLU Mistral-7B rate 0.5-2 rps 时调度 <1ms [PAPER FACT] (Table4)

## 11. Assumptions [PAPER FACT]

- 文档重算成本可通过离线 profiling T(α,β) 双线性插值近似 [PAPER FACT]
- 检索分布偏态且稳定 [PAPER FACT]
- 文档顺序敏感不可重排 [PAPER FACT]
- 检索候选早期稳定可推测 [PAPER FACT]
- T ≥ d 使定理成立 [PAPER FACT]
- Max prefill bs 由显存或 SM 较小者决定 [PAPER FACT]
- Host 容量 1-2 数量级大于 GPU [PAPER FACT]

## 12. Author-Stated Limitations [PAPER FACT]

- 预取开销权衡：推测在高负载下可能增加错误计算，需动态控制 [PAPER FACT]
- 仅文档级缓存，未探索更细粒度 token/block 级共享 [PAPER FACT]
- Host 容量假设：依赖大 Host 内存 [PAPER FACT]
- 容错开销：仅备份上层高频节点 [PAPER FACT]
- 索引限定：仅实现 IVF/HNSW 流水改 [PAPER FACT]

## 13. Inferred Limitations [AGENT INFERENCE]

- 嵌入与切分未评估 [AGENT INFERENCE]
- 顺序敏感未量化质量影响 [AGENT INFERENCE]
- 多租户隔离缺失 [AGENT INFERENCE]
- 动态负载阈值静态 [AGENT INFERENCE]
- PCIe 瓶颈未极致 [AGENT INFERENCE]
- 未与量化/压缩联动 [AGENT INFERENCE]
- 评估偏 QA 短输出 [AGENT INFERENCE]

## 14. Open Questions [AGENT INFERENCE]

1. 能否将 PGDSF Cost 从离线 profiling 改为在线学习？ [AGENT INFERENCE]
2. 文档级树 vs token 级 RadixTree 何者更优？ [AGENT INFERENCE]
3. 如何自适应决定推测分段数与 max_prefill_bs？ [AGENT INFERENCE]
4. 能否结合语义模糊缓存？ [AGENT INFERENCE]
5. Host 到 NVMe/远端时策略？ [AGENT INFERENCE]
6. 多租户隔离？ [AGENT INFERENCE]
7. 与 SCOPE 等压缩结合？ [AGENT INFERENCE]
8. 长输出 RAG decoding 管理？ [AGENT INFERENCE]

## 15. Related Papers To Read [PAPER FACT + AGENT INFERENCE]

- **vLLM [Kwon 2023 SOSP] PagedAttention：** 基座 [PAPER FACT]
- **SGLang [Zheng 2023] RadixAttention：** 前缀树复用仅 GPU LRU [PAPER FACT]
- **Faiss [Johnson 2019] & IVF/HNSW：** 向量检索实现 [PAPER FACT]
- **Lewis RAG [2020]：** 范式 [PAPER FACT]
- **CacheBlend [Yao 2024 EuroSys] / LMCache [2025]：** 后续 RAG KV 复用 [AGENT INFERENCE]
- **SCOPE [Wu 2025 ACL]：** 长输出压缩互补 [AGENT INFERENCE]


## Review Log — Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2404.12457v2 — verified RAGCache multi-level cache: knowledge tree prefix matching O(h), PGDSF Priority = Clock + Frequency×Cost/Size with Cost/Size via T(α,β) bilinear interpolation, Knowledge Tree non-contiguous blocks via PagedAttention; top 3% docs hit 60% requests (20× uniform), cached prefix 11.5× faster than full prefill (3.9× with transfer), reordering OrderPriority = CachedLength/ComputationLength verified.
- **Correction 1 — Venue enrichment:** Added PGDSF/TTFT summary to venue.
- **Correction 2 — Hardware/capacity:** Confirmed AWS g5.16xlarge A10G 24GB PCIe4.0×16 256 GiB host + 192 GiB cache, H800 80GB NVLink for 70B large model 384 GiB host; retained.
- **Correction 3 — Baselines:** Verified vs vLLM+Faiss (no cross-request reuse) and SGLang RadixTree LRU GPU-only; PGDSF +2%–32% over GDSF, +6%–62% over LRU, +6%–75% over LFU at host 8–128GiB, TTFT 1.38s vs 1.68/1.78/1.81 at 8GiB (MMLU 0.8rps) verified Fig.17/Table2.
- **Correction 4 — DSP theorem:** Verified Theorem 5.1 T≥d optimal: pool empty and candidate change triggers speculation, dynamic speculative pipelining 1.5–2.3× non-overlapping search reduction, 1.2–2.1× reordering gain, scheduling <1ms verified.
- **Status:** All primary numbers traceable [PAPER FACT]; minor enrichment only.
