# Paper Metadata

- **Title:** Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads [PAPER FACT]
- **Authors:** Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan — University of Chinese Academy of Sciences / ICT, CAS; Huawei Cloud (Hu intern) [PAPER FACT]
- **Venue:** arXiv preprint arXiv:2401.11181v1 [cs.DC], submitted 20 Jan 2024 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2401.11181 / https://arxiv.org/abs/2401.11181 / HTML https://arxiv.org/html/2401.11181v1 [PAPER FACT]
- **Code:** [NOT REPORTED] (paper text does not list public repository URL) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2401.11181 + https://arxiv.org/html/2401.11181v1 (v1, 20 Jan 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM inference serving must handle mixed downstream workloads (chat, summarization, creation) with vastly different prompt lengths (prefill) and generated lengths (decode) differing by >2 orders of magnitude (Fig1) [PAPER FACT]. Existing LLM deployment co-locates prefill and decode on same instances with continuous batching, ignoring phase characteristics [PAPER FACT]. This causes severe **interference** when requests with different characteristics run together: prefill-prefill, prefill-decode, decode-decode each measured to cause slowdown [PAPER FACT]. Naive static per-task provisioning is infeasible due to high infrastructure cost [PAPER FACT]. Need a distributed system that minimizes interferences without over-provisioning [PAPER FACT].

## 2 Motivation [PAPER FACT]

- **Phase characteristics:** Prefill is computation-heavy (compute-bound), quadratically scaling with prompt length; decode is memory-intensive, latency-critical, scaling sublinearly with generated tokens [PAPER FACT]. Fig2 shows prefill throughput flat after accelerator-saturate threshold, decode throughput rises with batch until memory bandwidth saturated [PAPER FACT].
- **Interference taxonomy measured (§2.2):**
  - *Prefill & Prefill:* Light prefill ~18 tokens (ShareGPT median) vs heavy prefill 512 tokens (saturate point on OPT-13B testbed) [PAPER FACT]. Light latency increases 2x with 7 concurrent light prefills, 8x with 63 [PAPER FACT]; >10x with heavy prefills [PAPER FACT]; heavy prefill also 3x slower with light prefills [PAPER FACT] (Fig3). When total tokens in batch > saturate threshold, latency spikes [PAPER FACT].
  - *Prefill & Decode:* Mixing in continuous batch: light decode (generates <100 tokens) per-iteration latency increases 5x with just one heavy prefill in same batch (Fig4a,b) [PAPER FACT]; light prefill needs >7 concurrent light decodes to degrade, but both can slow up to 2.5x (Fig4c,d) [PAPER FACT]; heavy decode similar [PAPER FACT].
  - *Decode & Decode:* Light decode 20-100 tokens vs heavy decode >512 tokens (very short prompts) [PAPER FACT]; batch 128 half heavy half light vs all light: throughput drops 16% and latency increases 23% (Fig5) [PAPER FACT].
- **Root cause:** Classic systems problems — adding compute-heavy jobs to saturated hardware, co-running batch + latency-critical jobs, unaware memory bandwidth/capacity contention and head-of-line blocking [PAPER FACT].
- **Mixed workloads common:** Conversation, summarization, writing datasets show bimodal distributions; system must handle all simultaneously [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Prefill-prefill interference (compute saturation).** [PAPER FACT] Accumulating prompt tokens beyond accelerator-saturate threshold (e.g., 512 for OPT-13B) keeps throughput flat but latency linearly grows → batching many prefills together wastes [PAPER FACT].
2. **Prefill-decode interference (batch vs latency-critical).** [PAPER FACT] Continuous batching mixes them per iteration; long prefill occupies GPU delaying decodes; many decodes increase memory pressure delaying prefill [PAPER FACT].
3. **Decode-decode interference (memory bandwidth/capacity hotspot).** [PAPER FACT] Without length-aware scheduling, heavy decodes (long generation, larger KV + bandwidth) co-located on same instance cause contention and HoL blocking, degrading overall throughput [PAPER FACT].
4. **No length awareness:** Scheduler unaware of decode length/resource usage; random placement leads to hotspots (some decode instances overloaded) [PAPER FACT].
5. **Disaggregation cost:** Transferring KV cache from prefill to decode instances over network can add overhead if not hidden; needs efficient network stack [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**TetriInfer: Carefully schedule and group requests by characteristics to battle interferences via three pillars [PAPER FACT].**

- **Pillar1 — Chunked Prefill (prefill-prefill):** Partition prompts into fixed-size chunks (ChunkSize = accelerator-saturate threshold) so accelerator always runs close to computation-saturated limit without over-saturation [PAPER FACT]. Pad last chunk to ChunkSize; maintain per-request last prefilled position; invoke LLM one chunk at a time [PAPER FACT]. Avoids latency penalty of oversized batches, analogous to Tetris packing [PAPER FACT].
- **Pillar2 — Disaggregate Prefill & Decode instances (prefill-decode):** Separate dedicated instances for prefill vs decode; each scales independently; transfer prefilled KV cache at chunk granularity to selected decode instance [PAPER FACT]. Instances are *virtual* — can be scaled up/down and **flip roles** via Instance Flip if load changes (§3.5) [PAPER FACT].
- **Pillar3 — Two-level scheduling with predicted resource usage (decode-decode):** Use LLM-based length predictor to speculate decode length range (bucketed), estimate resource usage bounds, then schedule: (a) inter-decode load balancing at prefill dispatcher (power-of-two → pick least-interference decode instance with enough resources), and (b) intra-decode local scheduling within each decode instance [PAPER FACT].
- **Hierarchical control:** Central control plane (global scheduler + cluster monitor) plus per-instance local schedulers [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig6b):** Four modules — centralized control plane, prefill instances, decode instances, length prediction model [PAPER FACT].
- **Control Plane (§3.2):**
  - *Cluster monitor:* Collects load stats from prefill/decode instances every 100 ms, aggregates decode load and broadcasts to all prefill instances; adds/removes/flips instances [PAPER FACT]; distributed without single point of failure [PAPER FACT].
  - *Global scheduler:* Maintains request status table (arrival, phase, SLA); forwards external requests to least-loaded prefill instance; collects streaming outputs from decode instances back to clients [PAPER FACT].
- **Prefill Instance (§3.3) — four sub-modules:**
  - *Prefill Scheduler (§3.3.1):* Maintains raw queue vs scheduled queue; three policies: FCFS (keep arrival order), SJF (ascending prompt length), LJF (descending) [PAPER FACT]; SJF/LJF estimate prefill time from token count [PAPER FACT]; PrefillSchedBatch limits #requests scheduled at once to avoid starvation (e.g., 20 queued with batch10 → two sorted batches) [PAPER FACT]; non-preemptive now, but chunking enables future preemptive SRTF [PAPER FACT].
  - *Length Predictor (§3.3.2):* Small LLM classification model (e.g., OPT-125M predicting for OPT-13B, ~10x faster) [PAPER FACT]; offline fine-tuning: take prompt-only dataset, send prompts to target LLM to generate, bucket response lengths by granularity (e.g., 100 → 0:0-200,1:200-400 etc.), train classifier; online parallel vs sequential modes tested — parallel mode chosen because >80% requests unaffected though 10% throughput hit under extreme stress (Fig17) [PAPER FACT]; granularity 200 tokens yields 74.9% accuracy [PAPER FACT]; predicts range, not exact, due to temperature/top-p variance [PAPER FACT].
  - *Chunked Prefill (§3.3.3):* Deterministic by accelerator+model — e.g., 512 tokens for OPT-13B on testbed [PAPER FACT]; slice/merge prompt tokens in scheduler order, pad final chunk with zeros, execute one chunk per forward [PAPER FACT]; length predictor still uses fixed-size batch (small model no clear threshold) [PAPER FACT].
  - *Dispatcher (§3.3.4):* Event-driven per prefilled chunk; three-step inter-decode algorithm: (1) categorize decode instances into α (enough resources, estimated via predicted range) and β (not), using broadcasted load; (2) power-of-two random pick two from α; (3) pick one with least interference (lowest heavy:light ratio, to spread heavy decodes) [PAPER FACT]; proven lowest total decoding time vs others Fig19 [PAPER FACT]; then transfer metadata + KV cache to selected decode instance. Transfer considerations: granularity (per-chunk vs per-request; chosen chunk-level to pipeline) and network stack [PAPER FACT].
- **Decode Instance (§3.4):**
  - Receives from any prefill instance; maintains local scheduler with three policies (details Fig18) for selecting decode requests to run with continuous batching but now only decodes [PAPER FACT]; uses predicted length for length-aware decisions to avoid hotspot co-location of heavy decodes [PAPER FACT].
  - Runs main LLM engine only for decode phase [PAPER FACT].
- **Instance Flip (§3.5):** Prefill/decode roles are virtual; if load shifts (e.g., many prefills vs many decodes), instances can flip to rebalance without physical redeploy [PAPER FACT].
- **Network Stack:** Python for most modules, C++ for KV transfer interfacing low-level APIs; mock emulated network bandwidth 200-300 Gbps/GBps for evaluation due to lacking high-end hardware (Fig9) [PAPER FACT].
- **Implementation base:** vLLM foundation for both prefill and decode instances [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Time-to-First-Token (TTFT):** Latency to generate first token (prefill) [PAPER FACT].
  - **Job Completion Time (JCT):** End-to-end per-request latency [PAPER FACT].
  - **Prefill latency / waiting time:** Avg prefill latency and avg waiting (queue) time [PAPER FACT].
  - **Perf/$ or perf per dollar:** Inference efficiency (throughput per dollar) [PAPER FACT]; resource usage (% less resources) [PAPER FACT].
- **Secondary:**
  - Throughput tokens/s vs batch size and token counts (Fig2) [PAPER FACT].
  - Interference slowdown factors (2x,5x etc.) [PAPER FACT].
  - Length predictor accuracy (74.9% at 200 granularity) and impact on throughput (Fig17) [PAPER FACT].
  - Scheduler comparison (SJF vs FCFS -7.8% waiting at batch16; chunked+FCFS -86.4% prefill latency vs vanilla vLLM) [PAPER FACT].
  - Network bandwidth sensitivity (200Gbps-300GBps) [PAPER FACT].
- **Workload-specific breakdowns:** Light prefill/heavy decode vs heavy prefill/heavy decode etc. Fig16 [PAPER FACT].
- **Not reported:** Energy, exact dollar cost, P99 tail, memory bandwidth absolute numbers [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **Vanilla vLLM [Kwon et al. 2023]:** Uses fixed batch size (not chunked), co-located prefill+decode with continuous batching; main comparison in end-to-end Fig16 [PAPER FACT].
- **Sarathi [Agrawal et al. 2023] (concept):** Concurrent chunked-prefill with prefill-decode-mixed chunks; TetriInfer contrasts as prefill-only chunks due to disaggregation [PAPER FACT] (discussion).
- **Different TetriInfer variants for ablation:** FCFS/SJF/LJF schedulers, chunked vs fixed batch, with/without length predictor, with/without disaggregation (Fig16,17,19) [PAPER FACT].
- **Network mock:** Vanilla vLLM vs TetriInfer under same emulated bandwidth [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Datasets (Fig1):**
  - Conversation: ShareGPT [35] — for light/heavy prefill/decode distribution examples [PAPER FACT].
  - Summarization: dataset [17] (e.g., LongBench-like) — long prompts short decodes [PAPER FACT].
  - Writing/creation: dataset [18] — short prompts long decodes [PAPER FACT].
  - Plus evaluation uses public ShareGPT dataset for end-to-end [35] [PAPER FACT].
- **Models:**
  - Target LLM: OPT-13B is primary (ChunkSize 512 example) [PAPER FACT]; also mentioned general LLM inference, but evaluation focuses on OPT family (implied via vLLM) [PAPER FACT].
  - Predict model: OPT-125M (and family) for length prediction, ~10x faster than OPT-13B [PAPER FACT].
- **Request types (Sec2.2):** Heavy/light prefill (18 tokens median vs 512 saturate) and heavy/light decode (<100 vs >512 generated) [PAPER FACT].
- **Evaluation workloads (Fig16):**
  - Light prefill + heavy decode, heavy prefill + light decode, mixed common, heavy prefill + heavy decode [PAPER FACT].
  - Prompt sizes 18, 512, distribution based, generation lengths 20-2000 [PAPER FACT].
  - Poisson arrivals, varying rates [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Testbed:** Real testbed with limited GPUs (exact count not numerically reported, but runs vLLM vs TetriInfer) [NOT REPORTED for exact GPU count/model] — authors state *since we cannot access high-end hardware, we implement a mock mechanism to emulate varying network bandwidth* (Fig9, Sec1) [PAPER FACT]; emulated bandwidth **200 Gbps to 300 Gbps (and 300 GBps mentioned)** connecting prefill/decode instances [PAPER FACT].
- **Accelerator-saturate example:** OPT-13B on their accelerator saturates at 512 tokens (Fig2) [PAPER FACT].
- **Precision:** [NOT REPORTED] exact precision (likely FP16 as vLLM default) [PAPER FACT — no explicit statement for evaluation precision].
- **Not detailed:** Exact GPU model (maybe A100/V100 but not stated for TetriInfer testbed), CPU, interconnect, cluster size [NOT REPORTED] except mocked network [PAPER FACT].

## 10 Main Results [PAPER FACT]

*Numbers from Abstract, Sec1,2,5, Figs2-5,16,17,19.*

- **Overall headline (Abstract):** TetriInfer uses **38% less resources** while lowering **average TTFT by 97% and average JCT by 47%** vs baselines [PAPER FACT].
- **Mixed common workload (Sec1, Fig16):** Improves **avg TTFT by 85% and avg JCT by 50%** vs vLLM [PAPER FACT].
- **Light prefill + heavy decode workload:** Improves **perf/$ by 2.4x** (Fig16) [PAPER FACT].
- **Heavy prefill + heavy decode:** **Marginal gain, overhead not offset** — TetriInfer not ideal, design overhead cannot be offset (Sec1, Fig16) [PAPER FACT].
- **Interference quantification (Sec2):**
  - Prefill&Prefill: light prefill +7 light → 2x, +63 light → 8x, + heavy → >10x; heavy + light → 3x (Fig3) [PAPER FACT].
  - Prefill&Decode: light decode +1 heavy prefill → 5x per-iteration decode latency (Fig4a,b); prefill +7 light decodes → 2.5x (Fig4c,d) [PAPER FACT].
  - Decode&Decode: batch128 half heavy half light vs all light → throughput -16% latency +23% (Fig5) [PAPER FACT].
- **Prefill scheduler microbenchmark (Sec3.3.1, Fig16):** **SJF lowers avg prefill waiting time by 7.8% vs FCFS at batch16**, more pronounced at larger batch [PAPER FACT].
- **Chunked prefill vs fixed batch (Sec3.3.3, Fig16):** **Chunked+FCFS lowers avg prefill latency by 86.4%** vs vanilla vLLM fixed batch [PAPER FACT].
- **Length predictor (Sec3.3.2, Fig17):**
  - Granularity 200 tokens → **74.9% accuracy** [PAPER FACT].
  - Parallel mode vs sequential: parallel does not affect main LLM for >80% requests, but -10% throughput under extreme stress; chosen parallel [PAPER FACT].
  - Small predict model (125M) ~10x faster than target (13B) [PAPER FACT].
- **Decode scheduling (Fig19):** Power-of-two + least-interference selection achieves **lowest total decoding time** vs random/round-robin etc. [PAPER FACT].
- **Sensitivity to Bandwidth (Fig9 mock):** TetriInfer maintains advantage across 200-300 Gbps/GBps emulated bandwidth; gains shrink at low bandwidth due to KV transfer overhead [PAPER FACT] (described).

## 11 Assumptions [PAPER FACT]

- LLM inference distinct phases: prefill compute-bound (quadratic with input length) and decode memory-bound (sublinear) [PAPER FACT].
- Prompt length distribution and decode length can be bucketed and predicted via small LLM classifier with ~75% accuracy suffices for scheduling [PAPER FACT].
- Accelerator has well-defined saturate threshold (ChunkSize) dependent on model hidden dim and accelerator capability (512 for OPT-13B) [PAPER FACT].
- vLLM-based implementation with continuous batching as reference; chunked prefill can be implemented without major kernel changes [PAPER FACT].
- Prefill time accurately estimable from token count (enables SJF/LJF) [PAPER FACT].
- Resources are disaggregated virtually; flipping roles is feasible quickly [PAPER FACT].
- Network between prefill/decode, even if emulated 200-300 Gbps, can be hidden by pipelining chunk transfers [PAPER FACT].
- Workload mix predictable enough that two-level scheduling spreads heavy decodes evenly [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Discussed in Sec1 and Sec3.3:
  - **Not ideal for heavy prefill + heavy decode workloads:** Room for improvement marginal, introduced overhead cannot be offset (Fig16) [PAPER FACT].
  - **Fine-tuned predictor per target model:** Need separate training per target LLM (OPT-13B etc.); granularity trade-off requires tuning [PAPER FACT].
  - **No online fine-tuning:** Work does not explore online adaptation of predictor; offline only (Fig8 caption) [PAPER FACT].
  - **No preemptive scheduling explored:** Only non-preemptive FCFS/SJF/LJF evaluated; shortest-remaining-time-first left for future (chunking enables it) [PAPER FACT].
  - **Small-model predictor uses fixed batch, not chunked:** Due to no clear saturate threshold for small models [PAPER FACT].
  - **Emulated network:** High-end hardware not available, uses mock bandwidth emulation → real RDMA/NVLink performance not measured [PAPER FACT].
  - Future work: improve prediction accuracy, explore preemptive policies [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Hardware realism:** Mocked network may underestimate real PCIe/RDMA latency and contention; no evaluation on actual multi-node NVLink/InfiniBand [AGENT INFERENCE].
- **Model scale narrow:** Only OPT-13B-scale evaluated; no 70B/MoE or GQA/MQA models where KV size differs and chunk threshold changes [AGENT INFERENCE].
- **Single-cluster size unknown:** No reporting of number of prefill/decode instances, GPUs, or scaling limit; scalability claim based on mock, not real large cluster [AGENT INFERENCE].
- **Prediction errors:** 74.9% accuracy → 25% mis-predicted ranges may still cause hotspots; no analysis of tail impact or misprediction correction [AGENT INFERENCE].
- **No TTFT SLO vs JCT tradeoff quantified per SLO tier:** Results aggregate averages; no P95/P99 SLO attainment like DistServe [AGENT INFERENCE].
- **No memory capacity evaluation:** For very long prompts (e.g., 32K) chunked prefill helps compute but still requires KV transfer of many chunks; not evaluated at long context [AGENT INFERENCE].
- **Flip overhead unmeasured:** Instance Flip time, state migration cost not reported [AGENT INFERENCE].
- **Baselines limited:** Only vanilla vLLM; no comparison to Splitwise, DistServe, DejaVu, Sarathi-Serve which also disaggregate [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. Can chunk size be auto-tuned online per batch composition vs static 512 for OPT-13B, especially for heterogeneous prompts? [AGENT INFERENCE]
2. How to improve length prediction beyond 74.9% — would larger predictor, prompt features, or online feedback close gap, and what is cost-benefit? [AGENT INFERENCE]
3. How does TetriInfer perform with modern GQA models (Llama-2-70B) where KV per token smaller and decode memory pressure lower? [AGENT INFERENCE]
4. Could preemptive chunked scheduling (SRTF) further reduce JCT beyond SJF, and what preemption overhead does it incur? [AGENT INFERENCE]
5. How to make Instance Flip live without dropping in-flight requests — need migration of KV/queues? [AGENT INFERENCE]
6. Does the power-of-two least-interference algorithm remain optimal under bursty arrivals or highly skewed heavy-decode ratios (e.g., 90% heavy)? [AGENT INFERENCE]
7. How to extend hierarchical to storage tier (CPU/SSD) as in InfiniGen or Mooncake for 1M context where decode instances cannot hold all KV? [AGENT INFERENCE]
8. What is the real perf/$ on heterogeneous pricing (spot, different GPU types) vs homogeneous mock? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **vLLM / PagedAttention [Kwon et al. SOSP23]:** Basis for TetriInfer implementation; fixed batch vs TetriInfer chunked; continuous batching [PAPER FACT].
- **Sarathi / Sarathi-Serve [Agrawal et al. 2024]:** Concurrent chunked prefill work but with prefill-decode-mixed chunks (hybrid batching) vs TetriInfer prefill-only chunks; both aim to keep accelerator saturated [PAPER FACT].
- **Orca [Yu et al. OSDI22]:** Iteration-level scheduling predecessor; continuous batching baseline that mixes phases causing interference [PAPER FACT].
- **DistServe [Zhong et al. 2401.09670] / Splitwise [Patel et al. 2311.18677] / DéjàVu [Strati et al. 2403.01876]:** Concurrent disaggregation of prefill/decode; DistServe adds goodput-optimized placement search, Splitwise focuses on heterogeneous cost, DéjàVu adds streaming + fault tolerance; TetriInfer distinguishes by chunked prefill + two-level predicted scheduling for decode hotspots [PAPER FACT].
- **Hierarchical KV systems:** CacheBlend, LMCache, Mooncake — disaggregated KV transfer beyond network mimicry (TetriInfer mock could use their RDMA) [AGENT INFERENCE].
- **SLO-aware scheduling (Pollux, Sia, Shepherd):** Goodput scheduling inspirations not directly evaluated but related [AGENT INFERENCE].
- **Long-context serving (LongNet, Gemini 1.5):** Motivates mixed workloads with extreme lengths >1M tokens, where chunking crucial [PAPER FACT].
- **Prediction models (e.g., AlpaServe predict):** Using small LLM to predict output length; TetriInfer fine-tunes OPT-125M classifier [PAPER FACT].

