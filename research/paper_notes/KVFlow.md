# Paper Metadata

- **Title:** KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows [PAPER FACT]
- **Authors:** Zaifeng Pan, Ajjkumar Patel, Zhengding Hu (corresponding), Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, Yufei Ding [PAPER FACT] ? Affiliations: University of California, San Diego (UCSD) (Pan, Patel, Hu, Shen, Guan, Li, Qin, Ding), Amazon Web Services (AWS) (Wang) [PAPER FACT]
- **Venue:** arXiv:2507.07400 [cs.DC] v1 10 Jul 2025, 567 KB [PAPER FACT]; CC BY 4.0 license [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2507.07400 / https://arxiv.org/abs/2507.07400 [PAPER FACT]; HTML https://arxiv.org/html/2507.07400v1 [PAPER FACT]
- **Code:** [NOT REPORTED] ? paper does not list public repository; implementation described as prototype based on SGLang v0.4.4 [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2507.07400 + https://arxiv.org/html/2507.07400v1 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM-based agentic workflows coordinate multiple specialized agents, each with a fixed prompt (role, behavioral traits, task description, few-shot examples) invoked repeatedly to solve complex tasks (e.g., MetaGPT software roles) [PAPER FACT]. Prefix caching reuses KV tensors for static prompt tokens to avoid redundant prefill [PAPER FACT]. However existing systems evict KV caches using Least Recently Used (LRU) when GPU memory insufficient [PAPER FACT]. In iterative sequential workflows (e.g., 4-agent cycle Planner?Executor?Expresser?Reviewer from PEER [Wang et al. 2024]), LRU evicts the soon-to-be-reused agent (Expresser) while retaining recently-generated dynamic suffixes unlikely to be reused, causing frequent cache misses and substantial recomputation or swapping overhead [PAPER FACT] (Fig. 1: timestamp 13 Executor active ? Expresser evicted, timestamp 14 Expresser miss) [PAPER FACT]. Problem is amplified by large fixed prompts (TestBench Agent >3000 tokens, RTL Generator >1000 tokens in MAGE [Zhao et al. 2412.07822] [PAPER FACT]) and by high concurrency of many distinct workflows competing for KV cache.

## 2 Motivation [PAPER FACT]

- Agentic workflows leverage human domain expertise via structured agent graphs to achieve consistent, robust performance across software engineering, retrieval, and reasoning tasks, but incur high latency from repeated LLM invocations per agent [PAPER FACT].
- Fixed prompt portions are constant across iterations and large (hundreds to thousands tokens); caching them significantly reduces prefill latency [PAPER FACT]. Dynamic parts (user questions, task progress) vary rapidly and are less valuable to cache [PAPER FACT].
- Prefix caching alone insufficient under limited GPU memory: two exhaustion sources ? high concurrent workflows (many active KV entries) and large prompts exceeding capacity (Llama-3.1-8B KV size grows linearly with tokens, Fig. 2a) [PAPER FACT]. CPU memory as secondary cache via PCIe swapping is faster than recomputation (Fig. 2b shows PCIe transfer < prefill compute), confirming offload viability [PAPER FACT].
- Workflows increasingly use multi-agent frameworks (MetaGPT, CAMEL, AutoGen, GPTSwarm, AFLOW, AgentScope) providing message passing, tool usage, multi-threaded concurrency, and computation-graph abstractions with edge pruning / topology optimization, but rely on conventional LLM serving for generation [PAPER FACT].
- Prior serving optimizations (PagedAttention vLLM [Kwon et al. SOSP23], RadixAttention SGLang [Zheng et al. NeurIPS24], continuous batching Orca, CachedAttention, etc.) target general request scheduling or KV fragmentation, while Autellix/ParrotServe explore agentic scheduling but do not consider prefix cache management [PAPER FACT]; InferCept predicts tool-call durations to retain/swap/discard KV for intercepted requests, orthogonal but not workflow-aware [PAPER FACT].
- Opportunity: workflow structure predicts future agent execution order; using it to guide eviction and prefetch can preserve imminent KV and hide transfer latency [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **LRU ignores future reuse distance.** [PAPER FACT] An agent about to execute may have been idle longest, while just-completed agent may not be needed soon. LRU evicts the former (e.g., Expresser) prematurely, causing recomputation stalls [PAPER FACT].
2. **Dynamic suffix pollution.** [PAPER FACT] Recently generated varying suffixes occupy cache despite low reuse likelihood, yet are retained under LRU while fixed prefixes are evicted [PAPER FACT].
3. **Shared prefix fragmentation.** [PAPER FACT] Multiple agents may share common prefix segments in radix-tree layout (Fig. 3b); agent-level eviction would discard shared nodes incorrectly; fine-grained node-level decision needed but LRU operates on nodes without workflow semantics [PAPER FACT].
4. **Reactive CPU-GPU loading stall.** [PAPER FACT] Existing hierarchical radix cache (HiCache) reactively loads offloaded KV only when agent scheduled (top timeline Fig. 4), incurring PCIe latency (2 GB/s Gen1 on A10G, 64 GB/s Gen5 on H100 still noticeable) that blocks generation [PAPER FACT].
5. **Insufficient overlap even with pipelining.** [PAPER FACT] HiCache overlaps layer *l* compute with layer *l+1* load (two-stage pipeline), but still suffers cold-start and limited overlap when compute < transfer, especially under high concurrency where bandwidth queuing delays arise [PAPER FACT].
6. **SGLang fragmentation hurts bandwidth.** [PAPER FACT] Fragmented KV storage layout prevents full PCIe bandwidth utilization; concurrent workflows exacerbate contention [PAPER FACT].
7. **Diverse workflow dependencies.** [PAPER FACT] Agent interactions are heterogeneous: synchronization barriers (need both Executor1 AND Executor2 ? max) vs conditional branches (either suffices ? min); traditional CFGs/DAGs insufficient to unify [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**KVFlow = Agent Step Graph + steps-to-execution workflow-aware eviction at KV-node granularity + fully overlapped proactive KV prefetching with status-aware scheduling [PAPER FACT].**

- **Agent Step Graph abstraction (?3.1):** [PAPER FACT] Nodes = agent invocations, edges = dependencies, each node has step aggregation function deriving steps-to-execution from predecessors (earliest possible step). Example upper workflow: Expresser = max(E1,E2)+1 (needs both); lower conditional: Expresser = min(E1,E2)+1 (either) [PAPER FACT]. Recursive propagation yields steps-to-execution for arbitrary structures including conditional branching and barriers [PAPER FACT].
- **Workflow-aware eviction priority assignment:** [PAPER FACT] Only fixed prompt portion prioritized; varying suffixes always highest eviction priority [PAPER FACT]. For each agent, steps-to-execution assigned to last node of its fixed prompt, propagated upward through radix tree; shared node gets **minimum** (least evictable) among children, ensuring retention as long as any near-future agent needs it [PAPER FACT]. Under memory pressure, evict varying suffixes first, then prefix nodes in descending priority order (larger steps-to-execution first) [PAPER FACT]. Resolves conflicts across concurrent workflows by taking lowest priority [PAPER FACT].
- **Overlapped KV prefetching (?3.2):** [PAPER FACT]
  - *Proactive prefetching:* While current agent (Planner) executes, anticipate next agent(s) via Step Graph and asynchronously load required KV from CPU?GPU in background threads. GPU execution (model forward + sampling, GPU?CPU output) and CPU?GPU KV loading use different hardware resources and PCIe full-duplex allows parallel without contention [PAPER FACT]. For branching, conservatively prefetch all possible next agents within concurrent prefetch limit [PAPER FACT].
  - *Status-aware scheduling:* Four cache-node states: in GPU, backup in CPU, loading, offloading [PAPER FACT]. Scheduler inspects required nodes, skips requests with loading status to avoid redundant loads, prioritizes other ready requests (e.g., Executor2 or other workflows) (Fig. 4 bottom timeline). Offloading nodes excluded from eviction to avoid race [PAPER FACT]. Together eliminates stalls and fully overlaps compute with prefetch [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Base:** Prototype on **SGLang v0.4.4** backend (radix tree prefix KV) + frontend [PAPER FACT]; extends radix cache to support workflow-aware eviction and overlapped prefetch; modifies frontend & backend to transmit workflow info via HTTP [PAPER FACT]. Method portable to other frameworks via HTTP request modification (not SGLang-locked) [AGENT INFERENCE/PAPER FACT ? paper states not limited to SGLang].
- **Step Information Capture (?3.3):** [PAPER FACT] Assumes each `sgl.function` corresponds to independent agent. Just-in-time substitution of LLM call embeds metadata into HTTP request: current agent identity + steps-to-execution of all agents in graph indicating upcoming invocations. Backend updates eviction priorities and triggers prefetch if evictable GPU memory sufficient [PAPER FACT].
- **Fixed vs dynamic boundary:** [PAPER FACT] Need to track last KV nodes of fixed prompt (Fig. 3b). Two alternatives: (1) primitive interface for user to explicitly mark end position of fixed part; (2) heuristic tracking cache hit history, treating consistently hit prefix as fixed part [PAPER FACT].
- **Client Tracking:** [PAPER FACT] Assigns unique **client ID** per application, attached to every request, disambiguates same agent names (e.g., two Planners from different workflows) to avoid interference across concurrent clients [PAPER FACT].
- **Cache-node status FSM:** Implements in-GPU / backup-in-CPU / loading / offloading states [PAPER FACT]; background load thread updates status upon completion, informing scheduler readiness; scheduler skips loading nodes and excludes offloading nodes from eviction [PAPER FACT].
- **Prefetch trigger:** Workflow-aware; proactive background thread loads upcoming agents? KV; concurrent prefetch limit enforced [PAPER FACT].
- **Hierarchical cache:** CPU memory as secondary cache for evicted fixed-prefix KV; leverages PCIe transfer (2 GB/s Gen1 A10G, 64 GB/s Gen5 H100) which paper shows < recompute time [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Speedup over baselines** (end-to-end latency ratio) for single-workflow and high-concurrency scenarios [PAPER FACT]; paper highlights up to 1.83? / 2.19? over SGLang w/ HiCache [PAPER FACT].
  - **End-to-end workflow latency** (10-agent sequential workflow total time) under different Fixed/Dynamic/Output token configs [PAPER FACT].
  - **High-concurrency throughput / speedup** with many simultaneous workflows (queries per second implicitly via latency under fixed concurrency) [PAPER FACT].
- **Secondary:**
  - **Cache miss overhead breakdown** (prefill vs PCIe load vs compute) [PAPER FACT]
  - **Sensitivity to fixed prompt length (4096 vs 8192) and output length** showing diminishing gain when decoding dominates [PAPER FACT]
  - **CPU-GPU overlap effectiveness** (qualitative pipeline timelines) [PAPER FACT]
  - **Realistic PEER workload distribution** (token length histograms) [PAPER FACT]
- **Not primary but preserved:** Semantic correctness (model weights/prompts/decoding unchanged, system-level only) [PAPER FACT]; therefore not measuring accuracy.

## 7 Baselines [PAPER FACT]

- **SGLang (GPU-only radix cache):** Radix-structured KV in GPU memory only, LRU eviction, recomputation on miss [PAPER FACT]; version v0.4.4 [PAPER FACT].
- **SGLang w/ HiCache (hierarchical radix cache):** SGLang?s default CPU-based extension; asynchronously backs up frequently used nodes to host memory, reloads from CPU on reuse (reactive), overlaps layer *l* compute with layer *l+1* load (two-stage pipeline) [PAPER FACT].
- **Not compared:** InferCept, vLLM APC, CachedAttention, Autellix etc. mentioned as orthogonal but not benchmarked [PAPER FACT].
- **Fairness:** Same workflow definitions, same token lengths, warm-up for cache construction; same hardware per experiment [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Synthetic sequential workflow:** 10 agents in sequence, each prompt = fixed prefix (shared) + dynamic suffix (varying) + output tokens; token sequences randomly sampled with controlled lengths [PAPER FACT].
  - Configs tested: Fixed / Dynamic / Output e.g., 8192/32/32, 4096/32/32 and varying output lengths (Fig. 5 x-axis) [PAPER FACT].
  - Warm-up: execute each agent?s fixed prompt multiple times to construct/backup prefix cache, then run workflow 10 times with varying dynamic suffix, average latency [PAPER FACT]; simulates repeated invocations / loop behavior [PAPER FACT].
- **High-concurrency synthetic:** Multiple independent workflows non-interacting non-sharing, on single H100; 4 configurations labeled by fixed prompt length per agent (512, 1024) and concurrent workflows (counts per setting ? e.g., 64 concurrent); dynamic/output fixed at 256 [PAPER FACT].
- **Realistic PEER simulation (?4.2 Realistic Workflow Simulation):** Based on PEER framework [Wang et al. 2407.06985] with 4 agents per workflow, using PEER templates; role & instruction sampled, LLM-generated prompts ? variability + partially overlapping prefixes (same app context) [PAPER FACT]. Input: Financial QA dataset from PEER [PAPER FACT]. Prompt lengths moderate: few dozen to several hundred tokens per agent (Fig. 7 distribution) [PAPER FACT].
- **Datasets/models for Fig.1/2:** Llama-3.1-8B KV size characterization (Fig. 2a/2b) batch=1 [PAPER FACT]; TestBench/RTL Generator token counts cited from MAGE [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Single-workflow setups (?4.1):**
  - (1) **Llama-3.1-8B** on **NVIDIA A10G 24GB, PCIe Gen1 bandwidth 2 GB/s** [PAPER FACT]; 32 attention heads, 8 KV heads [PAPER FACT].
  - (2) **Qwen2.5-32B** on **NVIDIA H100 80GB, PCIe Gen5 bandwidth 64 GB/s** [PAPER FACT]; 40 attention heads, 8 KV heads [PAPER FACT].
  - Deterministic decoding temperature 0 greedy sampling [PAPER FACT]; selected to represent tight GPU memory constraints [PAPER FACT].
- **High-concurrency:** Single **H100 80GB** [PAPER FACT].
- **Software:** SGLang v0.4.4 backend/frontend [PAPER FACT]; radix tree implementation [PAPER FACT].
- **Other hardware details:** CPU model, DRAM size, OS, CUDA version **[NOT REPORTED]**.
- **Interconnect:** PCIe Gen1 vs Gen5 as above; NVLink [NOT REPORTED] not discussed.

## 10 Main Results [PAPER FACT]

All numbers from ?4, Figs. 5-8.

- **Synthetic single-workflow 10-agent latency (Fig. 5 speedup over GPU-only SGLang):**
  - Under **8192/32/32 on A10G**, KVFlow **1.83? over SGLang w/ HiCache** and **2.91? over GPU-only SGLang** [PAPER FACT].
  - Increasing **output tokens diminishes relative gain** (decoding dominates) [PAPER FACT].
  - Increasing **fixed length magnifies gain**: avg speedup 1.48? at fixed 8192 vs 1.28? at fixed 4096 [PAPER FACT] (cache miss overhead larger with longer prefix).
  - Across all tested Fixed/Dynamic/Output configs, KVFlow consistently highest speedup [PAPER FACT].
  - HiCache generally > GPU-only (CPU load faster than recompute) but on **H100 8192/32/32 shows marginal/degraded vs GPU-only** due to suboptimal pipelining under memory contention/high volume [PAPER FACT] (author suspicion).
- **High-concurrency synthetic on H100 (Fig. 6, dynamic/output 256):**
  - Across 4 configs (512 vs 1024 fixed, varying concurrency), KVFlow consistently outperforms both, up to **1.25? speedup** over both baselines [PAPER FACT].
  - Gain larger with 1024 fixed vs 512 (higher miss overhead) [PAPER FACT].
  - HiCache particularly poor under concurrency: e.g., **1024 fixed, 64 concurrent ? 0.57? of GPU-only SGLang** (slower) [PAPER FACT]; suspected frequent reactive loads disrupt schedule-compute pipeline + fragmented layout underutilizes PCIe [PAPER FACT].
  - Overall **up to 2.19? performance gain over naive LRU HiCache with reactive loading** (abstract & ?4.2) [PAPER FACT].
- **Realistic PEER 4-agent workflows (Fig. 8, Financial QA, few hundred tokens/agent):**
  - **Up to 1.12? over SGLang and 1.08? over HiCache** [PAPER FACT]; demonstrates practical potential despite smaller prefixes.
- **Aggregate claims (Abstract):** Up to **1.83? single-workflow with large prompts** and **2.19? many concurrent workflows** vs SGLang with hierarchical radix cache [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Each `sgl.function` maps to independent agent; workflow metadata can be embedded via HTTP [PAPER FACT].
- Fixed prompt portions of agents remain constant across invocations and are the valuable caching granularity [PAPER FACT].
- Agent Step Graph?s earliest-step estimate with min/max aggregation sufficiently captures future reuse distance; specific dependency type can be abstracted away for cache management [PAPER FACT].
- CPU memory as secondary cache with PCIe transfer faster than recomputation (validated Fig. 2b) holds across evaluated models [PAPER FACT].
- PCIe full-duplex allows GPU compute (forward+samping) to overlap with CPU?GPU KV loading without contention [PAPER FACT].
- Varying suffixes always low reuse, given highest eviction priority safely [PAPER FACT].
- Steps-to-execution can be computed recursively and propagated upward via min among children for shared prefixes [PAPER FACT].
- Workload repetitiveness (multiple runs of same workflow) justifies warm-up and reuse expectation [PAPER FACT].
- System-level cache management does not affect semantic correctness [PAPER FACT].
- Providing fixed-part boundary explicitly or heuristic hit-history detection is sufficient to distinguish fixed vs dynamic [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Paper has **no explicit Limitations section** [PAPER FACT]; ?5 Related Work and ?6 Conclusion implicitly bound scope:
  - Focuses on agentic workflows with repeating fixed prompts; not validated on general unrelated request mixes where workflow predictability absent [PAPER FACT].
  - Does not resolve fragmentation issue itself (still SGLang fragmented layout) [PAPER FACT] ? gains via scheduling/overlap but underlying layout unchanged.
  - H100 HiCache degradation under large contexts suggests pipelining logic not fully optimal; similar contention could affect KVFlow at scale (branch conservative prefetch within limit) [PAPER FACT].
  - Decoding-dominated gain dilution: when output tokens large, KVFlow benefit proportion drops; orthogonal to speculative decoding / KV sparsity / early exit which authors suggest co-apply [PAPER FACT].
  - Prototype limited to SGLang v0.4.4 frontend API; adaptation to other frameworks requires HTTP modification but not demonstrated [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Workflow information completeness:** Requires accurate Agent Step Graph and steps-to-execution at runtime; if workflow is dynamic, user-defined conditionals not fully captured, or agents spawn sub-agents ad-hoc, prediction may be inaccurate and evict wrong nodes [AGENT INFERENCE].
- **Fixed-part detection heuristic fragility:** Hit-history heuristic may misclassify partially overlapping but non-identical prefixes (PEER variability) and incorrectly assign priorities [AGENT INFERENCE].
- **Prefetch variance under branching:** Conservative prefetch of all possible next agents within limit may waste PCIe bandwidth and pollute cache if branch prediction wrong, especially high concurrency [AGENT INFERENCE].
- **Scalability to hundreds of concurrent workflows:** Evaluated up to 64 concurrent on H100; 2 GB/s Gen1 limited; CPU memory capacity/bandwidth for many large 8192-token prefixes (each ~ tens of MB) could become bottleneck not measured [AGENT INFERENCE].
- **Mixed workload evaluation missing:** Synthetic and PEER workloads are agentic; no evaluation with hybrid chat + agentic + RAG serving on same instance [AGENT INFERENCE].
- **Fairness across clients:** Client ID disambiguation does not address fairness/scheduling priority among clients; one client's prefetch may starve another [AGENT INFERENCE].
- **Persistence beyond CPU:** No disk tiering or RDMA remote pool; evicted beyond CPU still requires recompute [AGENT INFERENCE].
- **Quantitative ablation lacking:** No isolated contribution of eviction policy vs prefetch vs status-aware scheduling quantified; cannot tell which dominates [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. How to automatically learn/verify Agent Step Graph from traces without manual sgl.function mapping, handling loops and dynamic tool results? Could LLM-generated workflow specs be parsed to auto-derive min/max aggregation? [AGENT INFERENCE]
2. Can steps-to-execution be replaced by learned reuse probability (e.g., history-based predictor) that adapts to non-stationary workflows and variable tool latency? [AGENT INFERENCE]
3. What is optimal fixed/dynamic boundary detection that guarantees correct prefix hit without user annotation, especially for semi-structured prompts with templated variables? [AGENT INFERENCE]
4. How to co-optimize prefetch degree and PCIe bandwidth allocation under high concurrency to avoid thrashing ? should prefetch be admission-controlled via utility model like Continuum?s TTL? [AGENT INFERENCE]
5. Could KVFlow?s node-level priorities integrate with compression/quantization (KIVI, GEAR) or eviction (H2O, Scissorhands) to further reduce per-token footprint while preserving workflow-aware retention? [AGENT INFERENCE]
6. Does the scheme extend to PD-disaggregated or CXL-based memory pools (Beluga, Mooncake) where hierarchy is flatter and transfer semantics differ? [AGENT INFERENCE]
7. How to provide SLO guarantees per workflow (p95 latency) when status-aware skipping may reorder requests ? what is starvation/fairness trade-off? [AGENT INFERENCE]
8. Would combining KVFlow with cache-aware scheduling (e.g., RadixAttention LRU vs LFU) yield additive gains, or does workflow-aware already subsume frequency? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **SGLang (Zheng et al., NeurIPS 2024) RadixAttention & HiCache:** Direct base system; radix tree prefix reuse and hierarchical CPU backup with layer-wise pipeline [PAPER FACT].
- **vLLM (Kwon et al., SOSP 2023) PagedAttention & APC:** Paged KV storage, alternative prefix caching via block hashing; baseline KVFlow implicitly compares against but uses SGLang [PAPER FACT].
- **PagedAttention/KV-cache transfer works:** CacheBlend, RAGCache, Cache-Craft, LMCache, Mooncake, Dynamo, MemServe ? RAG/knowledge KV reuse and remote pooling [PAPER FACT].
- **InferCept (Abhyankar et al., ICML 2024):** Predicts tool-call durations to retain/swap/discard intercepted requests; orthogonal, not workflow graph-aware [PAPER FACT].
- **CachedAttention (Gao et al., USENIX ATC 2024) & AttentionStore:** Multi-turn conversation prefix caching specialized strategies [PAPER FACT].
- **Agentic workflow frameworks:** MetaGPT (Hong et al. 2308.00352), CAMEL (Li et al. NeurIPS23), AutoGen (Wu et al. 2308.08155), PEER (Wang et al. 2407.06985), GPTSwarm (Zhuge et al. ICML24), AFLOW (Zhang et al. 2410.10762), AgentScope (Pan et al. 2407.17789) ? provide structured roles/graphs that KVFlow exploits [PAPER FACT].
- **Workflow scheduling without KV awareness:** Autellix (Luo et al. 2502.13965) program-level scheduling, ParrotServe (Lin et al. OSDI24) semantic variables ? address scheduling but not prefix cache management, complementary [PAPER FACT].
- **Decoding optimizations:** Speculative decoding (Chen 2302.01318, Leviathan ICML23), StreamingLLM attention sinks, CertaiNEx early exit ? noted as orthogonal co-applicable [PAPER FACT].
- **[AGENT INFERENCE] Continuum (Li et al. 2511.02230) TTL for multi-turn tools:** Similar tool-interleaved multi-turn KV retention problem but models per-turn queueing delay + TTL vs KVFlow?s step-graph prefetch; potential synergy for variable tool durations.
- **[AGENT INFERENCE] Beluga (Yang et al. 2511.20172) CXL memory pool:** Could serve as high-capacity, low-latency secondary cache for KVFlow?s offloaded prefixes, reducing PCIe contention vs CPU DRAM.
- **[AGENT INFERENCE] SCOPE/ChunkKV/SnapKV:** Phase-aware and compression/compression that could be layered atop KVFlow priorities.


---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
