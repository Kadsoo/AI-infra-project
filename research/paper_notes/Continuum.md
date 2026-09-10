# Paper Metadata

- **Title:** Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live [PAPER FACT]
- **Authors:** Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica [PAPER FACT] ? Affiliations: UC Berkeley (Li, He, Mang, Mao, Cheung, Gonzalez, Stoica), Stanford University (Zhang), Tensormesh (Chen), Tsinghua University (Zhou) [PAPER FACT]
- **Venue:** arXiv:2511.02230 [cs.OS] v1 4 Nov 2025 (437 KB), last revised v6 25 May 2026 (348 KB) [PAPER FACT]; CC BY 4.0 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2511.02230 / https://arxiv.org/abs/2511.02230 [PAPER FACT]; HTML https://arxiv.org/html/2511.02230v6 [PAPER FACT]
- **Code:** [NOT REPORTED] ? paper states will open-source traces, code, and agent serving testbed upon publication; implementation described as ~1k lines Python on vLLM, not yet released [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2511.02230 + https://arxiv.org/html/2511.02230v6 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numeric values traced to paper, else [NOT REPORTED].

## 1 Problem [PAPER FACT]

Modern agentic workloads (software engineering, computer use, scientific research) follow ReAct loop alternating LLM reasoning and external tool calls [PAPER FACT]. They are long-horizon multi-turn, interleaving thought ? tool ? context update over dozens to hundreds of turns, evaluated on ?-bench, MINT, AgentBench [PAPER FACT]. Existing inference engines (vLLM, SGLang) use end-of-turn eviction: discard KV cache once decoding finishes to maximize utilization, assuming request complete [PAPER FACT]. This breaks for agents because tool call is short (?2s, much shorter than human chat pause) and the next LLM request will arrive soon reusing same context [PAPER FACT]. Eviction triggers two overheads: (a) **Turn-based eviction cost** ? recompute full prefill or reload from CPU DRAM (if LMCache offloading enabled) when next step starts [PAPER FACT]; (b) **Per-turn queueing delay** ? even with instant CPU reload, evicted program?s next request must wait in waiting queue behind ongoing prefills/decodes to free GPU memory; this bubble accumulates over turns, increasing job completion time (JCT) and breaking program continuity (Fig. 1) [PAPER FACT]. Fig. 4 shows InferCept still accumulates bubble despite reload savings. Additionally tool durations are long-tailed and variable (Fig. 5: slowest 10% of BFCL fetch_url accounts 52.5% total delay, SWE-Bench cd 94.1% [PAPER FACT]), so static preserve lacks robustness and can cause unbounded GPU occupation or deadlock if tool hangs [PAPER FACT].

## 2 Motivation [PAPER FACT]

- Agentic applications scaling to multi-turn: SWE-Bench traces average (10.9 turns, 2.1 std), BFCL v4 6.3 turns 2.3 std; tool time mean 925 ms 3550 std (SWE) and 1923 ms 2133 std (BFCL); tokens per program 70,126 mean 19,732 std (SWE) and 93,256 68,687 (BFCL) via GPT-5 collected 100 traces each [PAPER FACT] (Table 2).
- As steps increase, expected remaining tokens decrease (Fig. 3) ? later turns closer to finish, suggesting program-level FCFS or more turns ? shorter remaining time, approximating SRTF without clairvoyance [PAPER FACT].
- Failure of prior methods (?2.2):
  - **Fixed-workflow schedulers** (Teola, Alto, Parrot, Teola dataflow, Alto streaming) assume static DAGs, not dynamic ReAct graphs [PAPER FACT].
  - **No tool awareness** (Autellix PLAS, Tempo SLO): ignore variable tool durations and KV retention [PAPER FACT].
  - **Insufficient retention** (InferCept preserve/swap/evict based solely on reload cost vs GPU occupation; Pie programmable handlers but no policy; Ayo/Alto/Parrot static) [PAPER FACT]. InferCept rarely preserves when LMCache makes reload cheap, yet queueing delay persists every turn [PAPER FACT].
- Characteristics: many short tool calls ? large reuse benefit; per-turn queueing delay models require new algorithm; variable durations require bounded retention (TTL) to prevent waste [PAPER FACT].
- Need to balance benefit (prefill/reload + queueing delay savings if hit) vs cost (GPU memory ? TTL blocking other requests) under unpredictable durations [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **End-of-turn eviction vs reuse window mismatch:** [PAPER FACT] Engines evict assuming finished request is low priority, but agentic gap is short pause before reuse; eviction forces expensive re-prefill each turn.
2. **Per-turn queueing delay unmodeled:** [PAPER FACT] Even with CPU offloading (LMCache non-blocking fast reload), evicted KV loses GPU residency ? next request queued behind active batch; accumulation over many turns degrades program JCT (Fig. 1 red blocks, Fig. 4 bubbles).
3. **Long-tail variable tool durations:** [PAPER FACT] Fig. 5 shows extreme variance; fixed preserve pins indefinitely, wasting memory if tool is in tail; tail occupies significant total delay.
4. **Deadlock risk:** [PAPER FACT] Many pinned KVs can fully occupy GPU, blocking new scheduling loop if all memory pinned and no next request in waiting queue frees it (?5.2).
5. **Scheduling continuity break:** [PAPER FACT] Eviction breaks FCFS program order; requests with earlier program arrival scheduled after later ones due to waiting gap (?3.2, ?4.3).
6. **Memory vs benefit trade-off:** [PAPER FACT] Larger TTL increases hit probability but also blocks memory proportional to MemUsage/M_avg ? ? (cost model) [PAPER FACT].
7. **Limited predictability:** [PAPER FACT] Tool duration distributions per tool type unknown cold-start; need empirical CDF with history, global fallback, default exponential assumption [PAPER FACT].
8. **Existing cost models incomplete:** [PAPER FACT] InferCept only cost = reload vs occupation cost, missing OutofOrderCost = T/M * MemUsage * ? where ? is memoryfulness factor [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Continuum = KV Cache Time-to-Live (TTL) retention + utility-optimal TTL selection (benefit = CacheMissCost + OutofOrderCost vs Cost) + program-level FCFS scheduling with deadlock prevention [PAPER FACT].**

- **TTL mechanism:** [PAPER FACT] For each request generating a tool call, pin its KV cache in GPU with TTL ? = max duration to stay before auto-eviction. If next request arrives within ? ? immediate resume saving prefill/reload and queueing delay; if exceeds ? ? auto-evict to free memory, robust to mispredicted long tails [PAPER FACT] (Fig. 6 trade-off).
- **Utility model (?4.1, Table 3):** [PAPER FACT]
  - Cost(?,r) = MemUsage(r)/M * ? where M avg memory footprint [PAPER FACT]; approximates number of average requests blocked.
  - Benefit(r) = CacheMissCost(r) + OutofOrderCost(r) [PAPER FACT]
  - CacheMissCost = MemUsage * Prefill-Reload(r) / M ; Prefill-Reload is prefill time or reload time depending on offloading, profiled via quadratic fit [PAPER FACT]
  - OutofOrderCost = T/M * MemUsage * ? [PAPER FACT] where T avg queueing delay per unit context, ? = -Corr(k, N-k) memoryfulness factor (?=0 memoryless geometric N, ?=1 fully memoryful fixed N, ?<0 anti-memoryful long-tail) [PAPER FACT]. For ?=1, delay = T/M * MemUsage exactly waiting time; scaled by ? otherwise [PAPER FACT]. This term absent in InferCept is key.
- **Optimal TTL (?4.2 Eq.1/2):** [PAPER FACT] ?* = argmax_? P(?,f) * Benefit(r) - Cost(?,r) = argmax P(?,f)*(T*? + Prefill-Reload(r)) - ? after canceling MemUsage/M [PAPER FACT], where P(?,f) = empirical CDF of tool f durations: |{t ? ?}|/|S[f]| from historical records S[f] [PAPER FACT]; solved by enumerating unique durations + ?=0 [PAPER FACT].
  - Cold-start handling: threshold K=100; if |S| ? K use fixed T_default derived from Exp(1) tool duration + ?=1 optimum; else if |S[f]| ? K use global CDF P(?,f_any); else per-tool fine-grained [PAPER FACT]; T initialized 0, updated sliding window avg of evicted queue delays [PAPER FACT].
- **Scheduling priority (?4.3):** Multi-key tuple ordering: (1) preempted status first, (2) TTL status ? pinned within TTL prioritized over unpinned, (3) program-level arrival order FCFS within each category [PAPER FACT]; preserves continuity and approximates SRTF via FCFS given decreasing remaining time trend [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig. 7):** [PAPER FACT] Modular on vLLM (v0.10.2) with minimal scheduler loop change (~1k lines Python) [PAPER FACT]; client adds program_id to every request to identify multi-turn programs; requests enter existing scheduler loop plus thin Tool-Call Handler invoked on arrival/finish [PAPER FACT].
- **Tool Call Handler (?5.1, Alg.1):** [PAPER FACT] Separate class, extensible parsers.
  - *OnRequestArrive(r):* Q?Q?{r}; if seen program, (f,t) tool info from r ? S[f]?S[f]?{t} [PAPER FACT] ? records inter-request interval t_arrive^{p,i+1} - t_finish^{p,i} as tool execution time for future [PAPER FACT].
  - *Parsing:* Checks message block type for function_call per OpenAI schema example (id, call_id, name get_weather) [PAPER FACT]; SWE-Bench extracts bash block first word as tool name [PAPER FACT]; Appendix B covers more formats, easily extended via parser like Appendix A example [PAPER FACT].
  - *OnRequestFinish(r):* if last request ? free KV; else f?next tool after r, id?Program ID, P[id]?CalcTTL(r,S[f]) [PAPER FACT].
- **Scheduler Pin/Unpin (?5.2, Alg.1 Schedule()):** [PAPER FACT]
  - *Pin:* if not last step and CalcTTL returns ?* ?0 ? pin_request(request, ?*) recording (id ? current_timestamp+?*) in pinned_requests dict, skipping KV block free; passed to waiting queue to prioritize next request [PAPER FACT].
  - *Unpin:* each scheduling step scans pinned_requests, unpins if current_time > TTL and program_id not in waiting queue (prevents premature eviction if follow-up already arrived but not yet scheduled); plus proactive unpin when program last step finishes [PAPER FACT].
  - *Deadlock prevention:* If scheduling fails due to space contention and pinned_requests non-empty, iteratively unpin victims with latest program arrival time until first request fits, freeing KV and requeueing [PAPER FACT].
- **Offline Profile (?5.2):** [PAPER FACT] Per hardware+model pair (A100/H100/B200 ? Llama 8B/70B etc): (1) GPU-CPU bandwidth via avg CPU offloading throughput, (2) prefill vs context length quadratic fit over chunk sizes {1000,2000,4000,...max} [PAPER FACT]; profiling <10 minutes per pair [PAPER FACT]; remaining pages approximation uses full prefill with little error when memory contended [PAPER FACT].
- **Implementation hooks (?5.3):** [PAPER FACT] Three functions in vLLM scheduler: func_call_finish(tool,timestamp), update_tool_call_time(program_id,timestamp), set_up_ttl(request,tool) [PAPER FACT].
- **Offloading integration:** Works with LMCache 0.3.7 non-blocking CPU offloading (DRAM 100GB A100, 200GB per GPU B200/H100) and SSD extension (400G/800G) [PAPER FACT]; also CPU offloading disabled mode [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Average Job Completion Time (JCT) / average response time / per-job delay** for agentic programs (end-to-end program JCT, not per-request) [PAPER FACT]; figures report speedups.
  - **Throughput: Jobs per second (JPS) or Requests per second** under Poisson arrival [PAPER FACT]; paper states JPS lower than prior papers due to many LLM calls per program (~10+).
  - **P90 / P95 latency** tail [PAPER FACT].
  - **Pass rate** on SWE-Bench-Verified with time limit 15 min preemption (higher pass rate indicates fewer timeouts) [PAPER FACT].
- **Secondary:**
  - **Per-turn queueing delay bubble time** (Fig. 4) [PAPER FACT]
  - **Scheduling overhead ms** (Table 4) [PAPER FACT]
  - **Inference steps per minute** for RL rollout (Table 5) [PAPER FACT]
  - **Sensitivity across batch size, chunk size, turn counts, SSD size** [PAPER FACT]
- **Not traced numerically:** Token-level throughput not primary.

## 7 Baselines [PAPER FACT]

- **Vanilla vLLM 0.10.2 default** chunk size 2048 [PAPER FACT]; end-of-turn eviction, no offloading (baseline in Fig.8/13/14/16) [PAPER FACT].
- **CPU DRAM offloading:** vLLM 0.10.2 + LMCache 0.3.7 [PAPER FACT]; DRAM 100GB A100, 200GB per GPU B200/H100 [PAPER FACT]; applied on top of algorithms below [PAPER FACT].
- **Autellix:** PLAS algorithm (program-level attained service) implemented on vLLM, extended to CPU offloading as Autellix+ [PAPER FACT].
- **InferCept:** selectively preserve/swap/evict implemented on vLLM+LMCache, updated cost estimation for non-blocking LMCache [PAPER FACT]; only reload cost considered [PAPER FACT].
- **Pie:** programmable serving decomposed generation loop (no policy, requires manual design) ? listed in Table 1 comparison but not end-to-end benchmark [PAPER FACT].
- **Distributed inference (real agent Fig.12):** [PAPER FACT]
  - SGLang 0.5.5.post3 native cache-aware routing
  - Nvidia Dynamo 0.7.0.post1 1P1D PD Disaggregation
- **Internal comparisons:** Program FCFS only, Static TTL (fixed threshold from cold-start) in ablation Fig.16 [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Traces collected via GPT-5 for better correctness (base small models often fail):** [PAPER FACT]
  - **SWE-Bench:** mini-swe-agent (rank #5 on leaderboard Apr 13) running SWE-Bench [Jimenez 2310.06770]; requests within context window; 100 traces analyzed Table 2 [PAPER FACT].
  - **BFCL v4 Web Search:** Berkeley Function Calling Leaderboard Web Search category latest version, answering questions with web browsing tools; scaled down workload by 0.4 to fit ?100 requests in Llama-3.1 128k context [PAPER FACT].
  - **OpenHand:** multi-SWE-bench Go language example from official repo [PAPER FACT].
  - **Real evaluation:** 500 tasks SWE-Bench-Verified on Tensormesh H100 testbed via job distributor Poisson [PAPER FACT].
- **Arrival pattern:** Poisson distribution for program arrivals [PAPER FACT].
- **Turn statistics (Table 2):** SWE 10.9?2.1 turns, 925?3550 ms tool time, 70126?19732 tokens/program; BFCL 6.3?2.3 turns, 1923?2133 ms, 93256?68687 tokens [PAPER FACT].
- **Emulation:** Trace replay experiments using recorded traces with Poisson scaling; for turn scaling law (Fig.14) repeat trace 1? to 5? while inversely scaling token lengths to keep within window [PAPER FACT].
- **Models:** [PAPER FACT]
  - Llama-3.1-8B, Llama-3.1-70B, Gemma-3-12B (8-bit? [NOT REPORTED]), GLM-4.5 355B (mentioned abstract), GLM-4.5-fp8 for RL rollout on 8?H100 [PAPER FACT] (GLM-4.5 355B hardware [NOT REPORTED] exacts beyond B200 mentions).
- **Datasets for motivation Figs:** Same 100 traces per dataset for Fig.3/5 distributions [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Models & Hardware combos in evaluation:**
  - A100-SXM from Runpod (SWE/BFCL Llama 8B 1?A100, Gemma 12B 1?A100) [PAPER FACT]
  - B200 on-prem servers (Llama 70B 4?B200, Llama 8B 1?B200) [PAPER FACT]
  - H100 from AWS/Tensormesh on-prem (OpenHand Llama-8B 1?H100, real SWE-agent H100 testbed) [PAPER FACT]
  - Additional H100 mention for Llama-8B OpenHand (8?H100 for RL) [PAPER FACT]
- **Memory/Offloading:** LMCache DRAM 100GB for A100, 200GB per GPU for B200/H100 [PAPER FACT]; SSD extension 400G/800G beyond CPU [PAPER FACT]
- **Profiling hardware-specific:** Prefill curve & GPU-CPU bandwidth per hardware+model pair <10min [PAPER FACT]
- **Software:** vLLM, SGLang, Dynamo versions as in Baselines [PAPER FACT]
- **CPU/RAM interconnect details, PCIe topology, network for distributed:** [NOT REPORTED]
- **Batch size / chunk size defaults:** Chunk size 2048 vanilla; batch size varied Fig.13 sensitivity [PAPER FACT].

## 10 Main Results [PAPER FACT]

All from ?6.

- **Overall claims (Abstract & ?1 & ?6):** Continuum improves average JCT by **>8?** while improving throughput in abstract; ?1 states reduces delay 1.12? to 3.66? and throughput 1.10? to 3.22? across three hardware+model setups on multi-turn workloads; internal Tensormesh testbed up to **8.18?** delay reduction for real SWE-agent workloads [PAPER FACT].
- **Fig. 8 (no offload, average JCT):** Across 6 panels (SWE/BFCL ? Llama 70B 4?B200 / Llama8B 1?B200 / Llama8B 1?A100 / Gemma12B 1?A100) Continuum consistently best vs vLLM & Autellix; e.g., Llama-3.1-8B up to **2? reduction avg response time vs vanilla vLLM** (text ?6.2) [PAPER FACT].
- **Fig. 10 (with DRAM offloading CPU 200GB per GPU B200 or 100GB A100):** Continuum consistently outperforms CPU offloading baselines; Autellix gain diminished vs offload; InferCept still worse due to missing queueing term [PAPER FACT] (Fig.10 four panels).
- **Fig. 11 P90/P95 (Llama-8B single B200 200GB offload SWE trace):** Continuum better tail latency due to per-turn queueing reduction [PAPER FACT] (two subplots p90/p95).
- **Fig. 9 OpenHands Llama-8B 1?H100:** Continuum best on avg and P95, improvement more significant due to higher avg turn count [PAPER FACT].
- **Fig. 12 Real SWE-Agent distributed 500 tasks Tensormesh H100:** Continuum consistently lower avg delay when pass rate equal, actually higher pass rate than SGLang/Dynamo because baselines exceed 15 min time limit and are preempted as failure [PAPER FACT].
- **Fig. 13 Sensitivity (0.13 JPS):** Improvement stable across max batch sizes and chunk sizes 256?4096 [PAPER FACT].
- **Fig. 14 Turn scaling (0.13 JPS, 200GB offload SWE):** Baselines degrade as turns 1??5? (more tool calls), Continuum maintains stable low latency [PAPER FACT].
- **Fig. 15 SSD offloading 400G/800G SWE Llama-8B B200:** Continuum reduces avg delay beyond CPU offload for both SSD sizes [PAPER FACT].
- **Fig. 16 Ablation SWE/BFCL:** Program-level FCFS ? Static TTL ? Continuum (per-tool TTL) each incrementally improves; shows contribution of queueing term vs fixed threshold [PAPER FACT].
- **Table 4 Scheduling overhead:** No offload: vLLM 0.95 ms, Autellix 0.82 ms, Continuum **0.96 ms**; With CPU offload: vLLM 2.33 ms, Autellix 2.18 ms, InferCept 2.25 ms, Continuum **2.30 ms** ? single-digit ms negligible vs GPU execution [PAPER FACT].
- **Table 5 RL rollout OpenHands GLM-4.5-fp8 8?H100 throughput steps/min:** vLLM 93.4, ThunderAgent 114.8, **Continuum 144.9** [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Each LLM step returns clear tool invocation followed by gap before next step (ReAct sequential reason?tool?reason rhythm) [PAPER FACT]; parallel tool calls still sequential rhythm supported [PAPER FACT].
- Tool call duration distributions per tool type are stationary and learnable from history; empirical CDF approximates true P(?,f) [PAPER FACT].
- MemUsage/M and average queueing delay T are stable and estimable via sliding window [PAPER FACT]; waiting queue contains enough requests for blocking effect when retention needed [PAPER FACT].
- Approx full prefill time for remaining pages error small when memory contended [PAPER FACT].
- ? memoryfulness factor based on correlation between served vs remaining requests captures queueing benefit of preserving order [PAPER FACT].
- Program arrival order approximates SRTF because remaining time decreases with steps (Fig. 3) [PAPER FACT].
- Offline profiling quadratic prefill fit holds for all context lengths [PAPER FACT].
- Agents post-trained with tools before production, allowing training-time statistics collection [PAPER FACT] (mentioned ?4.2 end).

## 12 Author-Stated Limitations [PAPER FACT]

- **Appendix D Limitations and Future Work** (referenced but not fully excerpted HTML) and ?7 / Appendix C state: [PAPER FACT]
  - Only validated on ReAct-style tool-interleaving agents with linear control flow; not yet on speculative branches, asynchronous multi-agent coordination, context folding (non-linear workflows) ? requires future extension [PAPER FACT] (Appendix C.1).
  - Novel tool-calling styles (parallel tool calling, long-running function calls, chain-of-abstraction) not fully evaluated [PAPER FACT].
  - Model architecture beyond Transformer not explored (Mamba, state space models mentioned Appendix C.2) [PAPER FACT].
  - TTL is coarse-grained validity window; optimality depends on empirical CDF accuracy, may need learning-based prediction under drift [PAPER FACT] (inferred from discussion but tied to traditional TTL lineage).
  - Evaluation limited to Python tool calling (bash fetch_url, cd etc.) and not diverse domain variability beyond SWE/BFCL/OpenHand [AGENT INFERENCE ? paper notes more diverse challenge sets needed in D].
  - No explicit discussion of fairness under many tenants beyond program FCFS [UNVERIFIED ? not in excerpt].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Per-tool history sparsity:** For rare tools (long-tail fetch_url tail 52.5% by 10% slowest), S[f] ? K=100 triggers global fallback, losing per-tool tail specificity [AGENT INFERENCE].
- **? estimation window sensitivity:** T sliding average and correlation may be noisy under Poisson burstiness; ? computed offline from traces may shift online [AGENT INFERENCE].
- **Single-node focus:** Distributed case uses simple session-aware routing; distributed load balancing, PD disaggregation interaction with TTL not benchmarked beyond SGLang/Dynamo comparison [AGENT INFERENCE].
- **Memory overhead of pinned_requests:** With thousands concurrent programs each pinned several seconds, dict growth and scanning each Schedule() O(P) could add overhead not measured [AGENT INFERENCE].
- **Quantitative queueing model linearity:** Cost = MemUsage/M * ? assumes linear blocking by average request size, ignoring paged fragmentation and variable sequence lengths [AGENT INFERENCE].
- **No integration with prefix sharing:** Prefill-Reload assumes full prefill; sharing across programs (e.g., same system prompt) not modeled, could overestimate benefit [AGENT INFERENCE].
- **RL rollout metric limited:** Steps per minute measured against ThunderAgent 2602.13692, but not vs highly optimized vLLM with chunked prefill etc. [AGENT INFERENCE].
- **Privacy/isolation:** Pinning KV across programs requires isolation; no discussion of cross-program leakage or multi-tenant security [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. How to extend TTL model to parallel / asynchronous tool calls where tool gap is not single blocking but DAG of tools (Anthropic parallel calling)? Would need graph-aware TTL rather than per-tool scalar [AGENT INFERENCE]
2. Can TTL be learned end-to-end via reinforcement learning using reward = -JCT, adapting to drift without maintaining empirical CDFs? [AGENT INFERENCE]
3. How to combine Continuum?s queueing-aware cost with InferCept?s swap vs discard decision (CPU vs recompute) into unified three-action policy with TTL? [AGENT INFERENCE]
4. What is optimal program-level scheduling under variable turn counts: is FCFS truly near-optimal vs SRPT with remaining-turn prediction? How to estimate remaining turns online? [AGENT INFERENCE]
5. Could Continuum integrate with KVFlow?s Step Graph for workflows that are both tool-interleaved and agent-graph structured, yielding hybrid retention? [AGENT INFERENCE]
6. How to handle long-tail *anti-memoryful* workloads (?<0) where serving longer appears to reveal more work ? should scheduler switch frequently vs pin? What online detection for negative ?? [AGENT INFERENCE]
7. Would hardware-assisted TTL (e.g., CXL memory expiration via Beluga) allow zero-copy unpin without CPU scan, scaling to 100k programs? [AGENT INFERENCE]
8. How to incorporate SLOs or user priorities into utility (currently throughput-focused via avg JCT) for mixed chatbot/agent co-serving (Tempo scenario)? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **InferCept (Abhyankar et al., ICML24 / arxiv 2310.04836):** preserve/swap/discard KV on intercept based on reload vs occupation cost only, no per-turn queueing, no bounded TTL [PAPER FACT].
- **vLLM (Kwon et al., SOSP23) & SGLang (Zheng et al., NeurIPS24):** PagedAttention, continuous batching, chunked prefill 2048; base engines Continuum extends [PAPER FACT].
- **LMCache (Cheng et al., 2510.09665) & Cachegen:** CPU/SSD/Disk offloading layer for KV, used as non-blocking reload path [PAPER FACT].
- **Autellix (Luo et al. 2502.13965) PLAS:** Program-level attained service scheduling, assumes longer executed ? longer remaining, opposite to agentic Fig.3 [PAPER FACT].
- **Pie (Gim et al., SOSP25):** programmable generation loop handlers, delegates policy but gives no retention algorithm [PAPER FACT].
- **Ayo / Teola, Alto (Santhanam et al. MLsys24), Parrot (Lin et al. OSDI24):** Static workflow DAG optimization, streaming pipelined execution, semantic variables; assume deterministic graphs, not ReAct dynamic [PAPER FACT].
- **Tempo / ThunderAgent (Kang et al. 2602.13692):** SLO heterogeneous and RL rollout agent serving; ThunderAgent is concurrent RL work baseline 114.8 steps/min vs Continuum 144.9 [PAPER FACT].
- **Dynamo (NVIDIA) & SGLang cache-aware routing:** Distributed PD disaggregation baselines [PAPER FACT].
- **TTL lineage (DNS/CDN caches ? Jung 2003, Nishtala Facebook 2013, Cohen 2005):** Traditional TTL for stale bound & robustness under unpredictable fetch latencies; Continuum first to regulate LLM KV cache via TTL as function of predicted tool duration + scheduling delay + workload memoryfulness [PAPER FACT].
- **[AGENT INFERENCE] KVFlow (Pan et al. 2507.07400) workflow-aware eviction + prefetch:** Complementary agent-graph steps-to-execution vs Continuum?s tool-duration TTL; joint design could unify.
- **[AGENT INFERENCE] Beluga (Yang et al. 2511.20172) CXL switch pooled memory:** Could provide near-local secondary tier for TTL-evicted KVs with load/store semantics, reducing reload cost vs LMCache.
- **[AGENT INFERENCE] SCOPE/AttentionStore/InfiniGen:** Conversation prefix caching / HBM management that could be integrated with TTL benefit model.


---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
