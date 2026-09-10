# Paper Metadata

- **Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving [PAPER FACT]
- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu ¡ª Moonshot AI, Tsinghua University (Ruoyu Qin intern at Moonshot AI, equal contribution with Zheming Li; corresponding zhang_mingxing@mail.tsinghua.edu.cn, xuxinran@moonshot.ai) [PAPER FACT]
- **Venue:** Preprint arXiv:2407.00079 [cs.DC], Submitted 24 Jun 2024 v1, last revised 3 Sep 2025 v4 (23 pages, 13 figures) [PAPER FACT]; arXiv perpetual non-exclusive license [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2407.00079 / https://doi.org/10.48550/arXiv.2407.00079 / HTML https://arxiv.org/html/2407.00079v4 [PAPER FACT]
- **Code/Trace:** https://github.com/kvcache-ai/Mooncake (trace open-sourced, dummy LLaMA2-70B) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2407.00079 + https://arxiv.org/html/2407.00079v4 (v4, 03 Sep 2025) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM serving workloads are diversified in input/output length, arrival frequency/distribution, and SLO demands (TTFT, TBT) [PAPER FACT]. As Model-as-a-Service (MaaS) provider (Kimi by Moonshot AI), goal is to maximize overall effective throughput (goodput/revenue) under latency SLO constraints while facing severe overload during peak times with limited GPU supply that cannot be elastically scaled [PAPER FACT]. Traditional coupled prefill+decode systems suffer interference: prefill is compute-intensive (superlinear with input length due to quadratic attention), decoding is memory-bandwidth-bound (sublinear with batch size) and both share same GPUs, limiting MFU and causing SLO violations for long contexts [PAPER FACT]. Need to decouple resources and center scheduling around KVCache reuse and transfer, balancing throughput-oriented optimizations (reuse KVCache, maximize batch tokens) against latency SLOs, under highly overloaded scenarios where requests must be rejected [PAPER FACT].

## 2 Motivation [PAPER FACT]

- GPU clusters provided as highly integrated DGX/HGX nodes need decoupling into disaggregated pools optimized for different goals [PAPER FACT]; many researchers suggested separating prefill and decoding clusters because KVCache shifts with requests moving from prefill to decoding [PAPER FACT].
- Two general throughput approaches both risk SLO violation: 1) reuse KVCache from remote location prolongs TTFT (waiting + network congestion), 2) large decoding batch improves MFU but increases TBT [PAPER FACT]; KVCache scheduling is central to balancing these [PAPER FACT].
- Kimi faces exponential workload growth; average input 7590 tokens, output 182 tokens, ratio ~41.7 (7590/182) [PAPER FACT] ¡ª note previously mis-copied as 720; corrected via arithmetic and cross-check with Table 2 (7955/194 ¡Ö41) and L-Eval workloads shows long-context dominance [PAPER FACT]; trace shows >50% blocks unused while some accessed tens of thousands times, requiring replication of hot blocks to avoid congestion [PAPER FACT]; cache capacity 1K->50K boosts hit ratio 30%->50%, then plateau at 50-51% even at Inf [PAPER FACT] Table 1.
- Existing research assumes sufficient resources and all requests will be processed, focusing on utilization; real MaaS providers face highly overloaded scenarios (peak overload common) [PAPER FACT]; scheduling must decide which requests to reject as early as possible if they cannot finish under SLO, otherwise prefill compute wasted [PAPER FACT]; definition of goodput differs: only fully completed requests counted [PAPER FACT].
- Straightforward early rejection based on current load causes load fluctuation (anti-phase between prefill and decode) due to time lag between prefill scheduling and actual decode execution, leading to poor utilization (Fig9-10) [PAPER FACT]; need prediction-based policy [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Prefill vs decode resource contention:** Prefill compute-bound (quadratic attention) benefits from many GPUs, decoding memory-bound benefits from large batch; coupling them forces same parallelism and causes prefill to disrupt decoding (vLLM long-context requests processed individually to avoid TBT violation) [PAPER FACT] Fig2.
2. **TTFT vs reuse tradeoff:** Reusing remote KVCache reduces compute but adds transfer latency (RDMA/CPU¡úGPU) and hot-spot congestion; Conductor must predict transfer time which depends on network status, not just size [PAPER FACT]; prefill scheduling also constrained by DRAM space reserved for global KVCache pool [PAPER FACT].
3. **TBT vs batch size tradeoff:** Aggregating many tokens improves MFU but increases TBT and is limited by total KVCache size fitting in VRAM [PAPER FACT].
4. **Cache load imbalance:** Prefill nodes each manage local prefix caches with skewed popularity (system prompts hot, long document cold); without replication, hot nodes congest, cold nodes underutilized; LRU vs LFU etc. need load-aware migration [PAPER FACT].
5. **Cross-node communication for long-context prefill:** Extending tensor parallelism across nodes requires 2 expensive RDMA all-reduces per layer reducing MFU; sequence parallelism (Ring/Striped Attention) still needs per-layer communication and competes with KVCache transfer [PAPER FACT]; dynamic repartitioning of TP vs SP groups causes complexity and low utilization [PAPER FACT].
6. **VRAM occupation cost for chunked prefill inlining:** If prefill chunk inlined with decoding batch, KVCache occupation S*T grows because T increases, wasting VRAM [PAPER FACT].
7. **Overload load-definition complexity:** In disaggregated architecture, load must be measured via SLO satisfaction (predicted max TTFT/TBT vs l_ttft/l_tbt) rather than simple request count ratio; time lag between prefill acceptance and decode load makes evaluation delayed, causing fluctuation [PAPER FACT]; predicting output length for accurate load forecast is hard (high cost/low accuracy) especially under overload [PAPER FACT].
## 4 Core Idea [PAPER FACT]

**Mooncake = KVCache-centric disaggregated architecture: separate prefill cluster + decoding cluster + disaggregated KVCache pool leveraging underutilized CPU DRAM/SSD/RDMA per GPU node, with global scheduler Conductor performing KVCache-aware scheduling and overload-oriented early rejection [PAPER FACT].**

- **Disaggregated pools (Fig1-3):** Prefill nodes and decoding nodes are separate pools; additionally CPU/DRAM/SSD/RDMA resources of GPU cluster are grouped to implement disaggregated cache of KVCache as unified pool [PAPER FACT]; KVCache stored as paged blocks in CPU memory with hash determined by own hash plus prefix for deduplication (Fig3) [PAPER FACT]; transfer handled by separate GPUDirect RDMA component called Messenger deployed as independent process per instance [PAPER FACT]; supports eviction policies LRU/LFU/LengthAwareCache and exposes context caching API to users for higher reuse [PAPER FACT].

- **Workflow per request (Fig4, Sec3):** Conductor selects pair (p,d) and executes 1) KVCache Reuse: selected prefill node loads reusable prefix cache from remote CPU memory via block IDs (skip if none), balancing reuse vs TTFT; 2) Incremental Prefill: prefill node completes prefill using prefix cache, stores newly generated incremental KVCache back to CPU memory; if uncached tokens > prefill_chunk (~ >1000 tokens to fully utilize GPU), split into chunks and execute via pipelined manner; 3) KVCache Transfer: Messenger streams KVCache layer-by-layer to destination decoding node CPU DRAM, asynchronously overlapped with prefill computation; 4) Decoding: after all KVCache received in decoding node CPU DRAM, request joins continuous batching next iteration; decoding node double-checks TBT SLO (anticipated load may have changed after prefill) and may reject, wasting prefill cost [PAPER FACT].

- **Chunked Pipeline Parallelism (CPP) for multi-node prefill (Sec5.1):** Instead of TP across nodes or SP (Ring/Striped Attention) which require per-layer communication, Mooncake groups every X nodes into pipeline group; input tokens partitioned into chunks ¡Ü prefill_chunk, different chunks processed simultaneously by different nodes, pipelining reduces TTFT [PAPER FACT]; benefits: only communication at pipeline stage boundaries overlap with compute ¡ú better MFU and less network contention with KVCache transfer; naturally fits both short and long contexts without frequent elastic scaling or global communication group [PAPER FACT]; first application of pipeline parallelism for inference stage per authors [PAPER FACT].

- **Layer-wise Prefill (Sec5.2):** Prefill is compute-bound layer-by-layer, so KVCache loading/storing can be overlapped via async launch/wait: before each layer attention wait for async load of that layer KVCache and trigger next layer load; after compute launch async store of that layer KVCache [PAPER FACT]; execution time ¡Ö max(KVCache loading time, standard prefilling) depending on prefix proportion [PAPER FACT]; Fig7 shows latency reduction for long contexts; key advantage: allows disregarding VRAM size in prefill scheduling as long as it can contain single request (scheduling only considers KVCache distribution and DRAM size) [PAPER FACT]; future idea to inline batch API decoding into prefill VRAM due to free space [PAPER FACT].

- **KVCache-centric scheduling (Sec6, Alg1):** Conductor evaluates per-instance prefix match length prefix_len vs best_prefix_len (found via FindBestPrefixMatch on block_keys = PrefixHash(prompt_tokens,B) with block size 512) [PAPER FACT]; estimates T_queue (sum of queued prefill times) and T_prefill (predictive model from offline test data based on length and hit length) and T_transfer (size + network congestion) [PAPER FACT]; chooses instance with shortest predicted TTFT = T_queue + T_prefill (+ T_transfer if using remote cache via cache-aware and balancing logic with kvcache_balancing_threshold manually tuned) [PAPER FACT]; if best/prefix_len < threshold ¡ú cache-aware prefill (use local prefix), else transfer from best_matched_instance and consider balancing [PAPER FACT]; decoding instance selected via SelectDecodingInstance load-balancing (TBT check) [PAPER FACT]; if TTFT > TTFT_SLO or TBT > TBT_SLO reject HTTP 429 [PAPER FACT]; if best/p ratio > threshold trigger TransferKVCache replication for hot-spot migration (heuristic-based automated hot-spot replication without precise future prediction) [PAPER FACT].

- **Overload-oriented scheduling (Sec7):** Defines load via SLO satisfaction: l_ttft, l_tbt constraints, load = predicted max TTFT/TBT vs thresholds for prefill/decode instances separately [PAPER FACT]; early rejection: advances decoding load assessment to before prefill to avoid wasted prefill when decode will reject; but causes anti-phase load fluctuation (Fig9-10 Stage1-4 description) due to delay between prediction and actual execution, especially severe with fewer prefill machines or longer prefill [PAPER FACT]; solution: Early Rejection Based on Prediction ¡ª predicts decoding load after prefill time for incoming requests; system-level prediction: assume uniform decode time td, for moment t add requests completable by prefill at t to uniform decode, remove those exceeding td before t, compute avg TBT ratio to l_tbt [PAPER FACT]; request-level prediction (output length) deemed too costly/inaccurate left for future [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Conductor global scheduler (center):** Dispatches requests based on KVCache distribution and workloads, replicates/swaps blocks, selects (p,d) pair per Alg1, handles SLO double-check [PAPER FACT].
- **Prefill pool:** Separate pool with configurable startup parameter per node (prefill vs decode); supports single-node TP for short contexts and multi-node CPP for long contexts; pipeline group size X tunable [PAPER FACT]; layer-wise launch/wait for async KVCache load/store per layer [PAPER FACT]; prefill_chunk threshold typically >1000 tokens [PAPER FACT]; scheduling ignores VRAM, focuses on DRAM and KVCache distribution [PAPER FACT].
- **Decoding pool:** Continuous batching with local scheduler double-checking TBT SLO; pre-selected node based on current load but may reject after prefill [PAPER FACT]; optimization goal maximize tokens per batch under TBT and VRAM KVCache size constraints [PAPER FACT].
- **KVCache pool & Messenger:** Paged blocks in CPU memory (DRAM) with prefix-hash deduplication (hash = token block hash concatenated with previous prefix hash, mapped to global ID) [PAPER FACT]; supports LRU/LFU/LengthAwareCache [PAPER FACT]; Moss? Actually Messenger uses (GPUDirect) RDMA, up to 800 Gbps interconnect, independent process per instance handling cross-machine KVCache transfer, async overlapped [PAPER FACT]; CPU memory also backs SSD for larger capacity (not quantified) [PAPER FACT].
- **Cache block size:** 512 tokens per block (hash_ids field) [PAPER FACT]; star example first 12 hash IDs identical ¡ú 12*512=6144 tokens prefix shareable [PAPER FACT].
- **Hot-spot migration:** Heuristic replication: if request not routed to longest prefix node due to load but estimated extra prefill < transfer time, proactively retrieve and store locally; also preference to recompute instead of transfer if best remote prefix ¡Ü local*threshold, automatically spreading hot blocks [PAPER FACT]; threshold manually tuned, adaptive future [PAPER FACT].
- **Admission control:** HTTP 429 Too Many Requests if SLO not achievable [PAPER FACT]; prefill queue time predicted via aggregating prefill times of queued requests, computed in parallel negligible vs inference [PAPER FACT]; prefill execution time predictive model from offline data [PAPER FACT].
- **Tracing:** Open trace with fields timestamp (0-3,600,000 ms), input_length, output_length, hash_ids remapped IDs without real content for privacy [PAPER FACT]; 23,608 entries over 1 hour [PAPER FACT] (also 23,000 used in experiments) [PAPER FACT].
- **Dummy model for reproducibility:** All experiments replay traces using dummy model following LLaMA2-70B architecture to protect proprietary info [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Goodput / Effective Throughput:** Maximize overall throughput counting only requests that fully complete under SLO (TTFT_P90 and TBT_P90) ¡ª differs from traditional goodput counting tokens [PAPER FACT]; normalized TTFT/TBT vs upper limits (baseline 1.0) to assess SLO attainment [PAPER FACT]; throughput measured as RPS (requests per second) at which P90 still meets SLO [PAPER FACT]; ultimate comparison: RPS improvement vs vLLM under same SLO [PAPER FACT].
  - **SLO attainment:** TTFT_P90 <= 10¡Á single-request baseline and TBT_P90 <= 5¡Á baseline (Sec8.1) for end-to-end experiments [PAPER FACT]; real workload specific: TTFT ¡Ü30 sec, TBT ¡Ü0.1 sec per token for 10P+10D replay [PAPER FACT]; also fixed SLOs in deployment [PAPER FACT]; metrics reported as CDF, normalized values, and failure rate (exceeding threshold means wasted resources) [PAPER FACT].
  - **Latency components:** Prefill execution time (quadratic vs length) and decoding batch time (sublinear vs batch) normalized throughput/latency Fig2 [PAPER FACT]; layer-wise prefill storing latency reduction Fig7 [PAPER FACT]; TTFT breakdown into queue + prefill + transfer [PAPER FACT].
  - **Throughput gains:** % increase over baseline vLLM for various RPS and datasets [PAPER FACT]; e.g., up to 525% simulated, 20-40% public, 75% real workload [PAPER FACT].

- **Secondary:**
  - **Cache hit ratio:** Under different capacities (1K,10K,30K,50K,100K,Inf) and policies (LRU, LFU, LengthAwareCache) [PAPER FACT] Table1 (30% at 1K ¡ú 50% at 50K ¡ú 51% at Inf, LRU best) [PAPER FACT]; block popularity CDF (over 50% unused, tens of thousands hits for hot) Fig6 [PAPER FACT].
  - **Prefill scheduling effectiveness:** Average TTFT and TTFT SLO attainment rate comparing random vs load-balancing vs cache-aware vs KVCache-centric (load+cache) on 8P+8D 23K requests [PAPER FACT] Fig8.
  - **Overload rejection count:** Number of rejected requests under Baseline (4183), Early Rejection (3771), Early Rejection based on Prediction (3589) at 2¡Á replay speed on 8P+8D [PAPER FACT] Table3.
  - **Instance load over time:** Prefill and decode load curves (0-1) over 20 mins showing anti-phase fluctuation before/after prediction (Fig9-10) [PAPER FACT].
  - **MFU / Model FLOPs Utilization** qualitative discussed for batch and pipeline efficiency [PAPER FACT] but not quantified numerically beyond throughput.

- **Not elaborate:** Per-token latency vs batch size numeric table beyond Fig2, energy per token, cost per dollar, SSD vs DRAM tier performance [NOT REPORTED]; detailed network congestion metrics beyond transfer time estimation [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **vLLM [Kwon et al. SOSP 2023]:** State-of-art open-source serving with continuous batching and PagedAttention, coupled prefill+decode; used as main baseline in testbed: vLLM-[4M] (4 instances each mixed) vs Mooncake-[3P+1D] and [2P+2D] for public/simulated, and vLLM-[20M] vs Mooncake-[10P+10D] for real workload [PAPER FACT]; note: vLLM processes long-context requests individually rather than batched to avoid TBT violation in simulated experiment [PAPER FACT].
- **Random Scheduling:** Prefill instance selected arbitrarily per request; baseline for cache-aware experiment [PAPER FACT] Fig8.
- **Load-balancing Scheduling:** Choose lightest load instance; baseline for cache-aware experiment [PAPER FACT] Fig8.
- **Cache-aware Scheduling (¡ì6.1):** Only considers prefix hit length and queuing, without load balancing replication logic; compared to full KVCache-centric (aware+balancing) in Fig8 [PAPER FACT].
- **Baseline overload rejection (¡ì8.2):** Rejects based on load before both stages separately (leading to waste after prefill); compared to Early Rejection and Early Rejection based on Prediction [PAPER FACT] Table3.
- **Related disaggregated systems mentioned but not directly benchmarked in numbers:** Splitwise [Patel et al. 2311.18677], DistServe [Zhong et al. 2401.09670], TetriInfer [Hu et al. 2401.11181] ¡ª discussed as corroborating findings, optimized resource allocation/parallel strategies for each stage [PAPER FACT]; ORCA iteration-level scheduling [Yu et al.], FlexGen, SARATHI, FastServe, FasterTransformer, TensorRT-LLM, DeepSpeed Inference as prior scheduling/memory management [PAPER FACT] Sec9.
- **Prefix caching baselines:** Prompt Cache, SGLang RadixAttention (LRU radix tree), AttentionStore hierarchical KVCache (concurrent work) [PAPER FACT] Sec9.

## 8 Workloads [PAPER FACT]

- **Trace dataset (open-sourced):** 23,608 entries sampled over 1 hour preserving session cache relationships, fields timestamp, input_length, output_length, hash_ids (remapped block hashes) [PAPER FACT]; two samples shown: 27482 ms 6955/52 hashes [46..57,2353,2354] and 30535 ms 6472/26 hashes [46..57,2366] with 12 identical prefix blocks (6144 tokens) [PAPER FACT]; average input 7590, output 182, ratio ~41.7 [PAPER FACT]; block size 512 tokens [PAPER FACT].

- **Evaluation datasets (Table2, Sec8.1):**
  - **ArXiv Summarization [Cohan et al. 2018]:** Avg input 8088, output 229, cache ratio ~0%, Poisson arrival [PAPER FACT].
  - **L-Eval [An et al. 2023]:** Avg input 19019, output 72, cache ratio >80%, Poisson [PAPER FACT].
  - **Simulated Data:** Input lengths 16K,32K,64K,128K, output 512, cache ratio 50%, Poisson [PAPER FACT].
  - **Real Data (trace replay):** Avg input 7955, output 194 (or 7590/182 overall, ratio ~41.7) [PAPER FACT], cache ratio ~50% (real workload section says 7955/194, earlier statistical says 7590/182 ¡ª both reported), timestamp-based arrival [PAPER FACT]; 23,000 real request traces replayed [PAPER FACT].

- **Models:** Dummy model following LLaMA2-70B architecture for all reported results to protect proprietary info and facilitate reproducibility [PAPER FACT]; (no real Kimi model weights disclosed) [PAPER FACT]; Fig2 shows normalized throughput/latency for dummy 70B [PAPER FACT].

- **Cluster configs tested:**
  - Public datasets: 4 nodes cluster: vLLM-[4M] vs Mooncake-[3P+1D] and [2P+2D] [PAPER FACT].
  - Simulated: same 3P+1D/2P+2D/4M [PAPER FACT].
  - Real workload: 20 nodes: vLLM-[20M] vs Mooncake-[10P+10D] [PAPER FACT]; TTFT limit 30 sec, TBT 0.1 sec/token [PAPER FACT].
  - Overload experiment: 8P+8D at 2¡Á replay speed [PAPER FACT]; prefill scheduling experiment 8P+8D with 23,000 requests [PAPER FACT].
  - Overload-oriented 20-machine load curve: 20 machines, 20 mins observation [PAPER FACT] Fig9.

- **Arrival patterns:** Poisson process for synthetic/public (RPS controlled), timestamp-based replay for real [PAPER FACT]; RPS varied to find SLO meeting point [PAPER FACT].

- **Metric thresholds:** For end-to-end, TTFT_P90 =10¡Á lowest observed, TBT_P90=5¡Á lowest [PAPER FACT]; real deployment fixed 30s/0.1s [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Testbed per node (Sec8.1 Testbed):** 8 NVIDIA-A800-SXM4-80GB GPUs, 80GB HBM, NVLINK interconnect; RDMA network aggregate up to 800 Gbps interconnect bandwidth (8x200Gbps HDR IB per VM class, Mooncake testbed 800Gbps per node) [PAPER FACT]; each node deploys either prefill or decoding instance per startup parameter [PAPER FACT].
- **Disaggregated resources pooled:** CPU, DRAM, SSD, RDMA resources of GPU cluster leveraged for KVCache (underutilized) [PAPER FACT]; specific DRAM/SSD sizes [NOT REPORTED] exact GB [PAPER FACT states ample cache capacity without additional costs] [PAPER FACT].
- **No GPU type heterogeneity discussed in evaluation:** All A800 homogeneous; future work proposes heterogeneous accelerators (GDDR/LPDDR bandwidth per dollar order magnitude better than flagship, PIM, hybrid bonding) [PAPER FACT] Sec10.
- **Network:** GPUDirect RDMA for Messenger transfers [PAPER FACT].
- **Precision:** [NOT REPORTED] dummy model precision (likely BF16) not stated [PAPER FACT].
- **Scale:** Up to 20 nodes (160 GPUs) for real workload; other experiments 4 nodes (32 GPUs) [PAPER FACT].
- **Software:** Built on vLLM open-source community acknowledged [PAPER FACT]; Conductor global scheduler implementation details not open [PAPER FACT].
## 10 Main Results [PAPER FACT]

All numbers from Abstract, Sec6,8, Figs 2,7-13, Tables 1-3.

- **Cache policy analysis (Table1, Sec4.2 global cache assumption):** Hit ratio 0.30 at 1K blocks ->0.40 at 10K ->0.48 at 30K ->0.50 at 50K ->0.51 at 100K/Inf for LRU [PAPER FACT]; LFU slightly lower beyond 30K (0.43 at 30K), LengthAware similar [PAPER FACT]; LRU best due to temporal proximity [PAPER FACT]; over 50% blocks never reused, while hot blocks accessed tens of thousands times (Fig6 CDF) [PAPER FACT]; required capacity scales proportionally with workload subset (sample trace subset suggests 50K plateau but real workload larger) [PAPER FACT].

- **Layer-wise prefill latency (Fig7):** Layer-wise overlapping effectively reduces storing KVCache latency for long-context requests; execution time ¡Ö max(load/prefill) [PAPER FACT]; exact ms numbers not extracted from HTML image [NOT REPORTED exact values], but qualitative reduction shown [PAPER FACT].

- **Prefill scheduling effectiveness (Fig8, 8P+8D, 23K requests):** Both cache-aware and KVCache-centric (with load balancing) significantly reduce average TTFT and increase TTFT SLO attainment rate vs random and load-balancing; full KVCache-centric outperforms both metrics over cache-aware alone [PAPER FACT]; specific TTFT ms numbers not numerically listed in text [NOT REPORTED exact], but visual bar shows clear reduction [PAPER FACT].

- **End-to-end public datasets (Fig11, Sec8.1.1, 4 nodes):**
  - ArXiv Summarization (~0% cache): Mooncake-[3P+1D] achieves **20% throughput improvement** over vLLM-[4M] while satisfying SLOs [PAPER FACT]; L-Eval (>80% cache): **40% improvement** over vLLM-[4M] [PAPER FACT]; improvement further enhanced by prefix caching reducing prefill time on L-Eval [PAPER FACT].
  - Mooncake-[2P+2D] has lower TBT latency due to more decode nodes but worse TTFT vs [3P+1D] and vLLM due to load imbalance between prefill/decode, indicating need for stable proportion preset [PAPER FACT]; imbalance minor temporary in real clusters [PAPER FACT].

- **Simulated long-context (Fig12, Sec8.1.2, 16K/32K/64K/128K, 50% cache, same 3P+1D/2P+2D/4M):** vLLM long-context requests disrupt decoding, processed individually to avoid TBT violation; Mooncake batch processing with disaggregation ensures never breaks TBT SLO [PAPER FACT]; throughput enhancements **ranging from 50% to 525%** while adhering to same TTFT/TBT SLO constraints [PAPER FACT]; specific per-length gains not individually quoted but range provided [PAPER FACT]; largest gain in certain simulated scenario reported as **up to 525% increase** (abstract) [PAPER FACT].

- **Real workload (Fig13, Sec8.1.3, 10P+10D vs 20M, 30s TTFT / 0.1s TBT limits, CDF):**
  - TTFT distributions nearly identical, almost 100% requests meet TTFT SLO for both systems [PAPER FACT].
  - TBT: **Mooncake ~100% meet TBT SLO, vLLM only 57% meet**, some vLLM requests extremely high TBT [PAPER FACT].
  - Overall enables Kimi to handle **75% more requests** under real workloads (abstract and Sec8.1.3 text: "Mooncake can process approximately 75% more requests while adhering to SLOs") [PAPER FACT]; alternative phrasing: Kimi handles 75% more requests vs baseline [PAPER FACT].

- **Overload scenarios (Table3, Sec8.2, 8P+8D at 2x replay speed, 23K requests):**
  - Baseline (separate load rejection) rejects **4183** requests [PAPER FACT].
  - Early Rejection reduces to **3771** (saves 412 wasted prefills) [PAPER FACT].
  - Early Rejection based on Prediction further reduces to **3589** (saves additional 182) [PAPER FACT]; demonstrates mitigating load fluctuations increases handling capacity [PAPER FACT].
  - Load fluctuation observation: before prediction, 20-min trace shows significant anti-phase fluctuations between prefill and decode loads (Fig9) [PAPER FACT]; theoretical 4-stage fluctuation explanation with green (prefill 0-1) vs yellow (decode) curves [PAPER FACT]; after prediction, fluctuation mitigated (Fig10b) [PAPER FACT].

- **Overall summary per abstract:** Mooncake excels in long-context scenarios; compared to baseline method achieves up to 525% increase in certain simulated scenarios while adhering to SLOs; under real workloads enables 75% more requests [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder-only architecture with prefill (parallel, compute-intensive, quadratic attention) and decode (autoregressive, memory-bound, one token per batch, continuous batching) stages [PAPER FACT].
- SLOs defined as TTFT (prefill) and TBT (decode) with P90 multipliers (10¡Á and 5¡Á of isolated single-request baseline) or fixed thresholds (30s/0.1s) [PAPER FACT]; exceedance = failure and wasted resources [PAPER FACT].
- Goodput counts only fully completed requests; partial tokens after rejection wasted [PAPER FACT].
- KVCache reuse via prefix hashing with block size 512; identical hash implies same token block plus prefix (deduplication) [PAPER FACT]; reuse beneficial but limited (theoretically up to only 50% reuse even with infinite storage/SLO in current workloads, but can be 90% for specific services like chat-to-paper papers.cool) [PAPER FACT] Sec9.
- Predictive model for prefill time based on offline data (length + prefix hit length) has small error due to regular Transformer compute pattern [PAPER FACT].
- System-level uniform decode time td assumption for prediction is sufficient; request-level output length prediction too costly/inaccurate, especially under overload, so not used [PAPER FACT].
- kvcache_balancing_threshold manually tuned but can be adaptively adjusted future [PAPER FACT].
- Workload patterns: average lengths and arrival Poisson for synthetic, trace replay preserves session caching relationships [PAPER FACT].
- Disaggregated transfer via RDMA Messenger can achieve high bandwidth (800 Gbps) and overlap with compute; layer-wise async hides latency [PAPER FACT].
- Dummy LLaMA2-70B architecture representative of real Kimi model for replay evaluation; trace without real user content preserves caching utility [PAPER FACT].
- Elasticity: proportion of prefill/decode instances stable over periods, can be preset; not requiring frequent dynamic scaling in current evaluation [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

From Sec7,10,11 discussions (no explicit Limitations header, but enumerated as future work / caveats):

- **Prediction accuracy limited to system-level:** Request-level generation length prediction left for future work due to high cost/low accuracy; current system-level uniform td prediction is coarse and requires less precision [PAPER FACT] Sec7.4.
- **Hot-spot threshold manual:** kvcache_balancing_threshold currently manually adjusted, not yet adaptive algorithm [PAPER FACT] Sec6.2 footnote.
- **Transfer time prediction difficulty:** Transfer time depends not only on size but also current network congestion, especially sending node congestion due to hot blocks; replication mitigates but not fully solved [PAPER FACT] Sec6.1-6.2.
- **Imbalance between prefill/decode pools:** Mooncake-[2P+2D] TTFT worse than [3P+1D] despite more decode nodes (Fig11) shows static partition can be suboptimal; proportion preset works for stable periods but temporary imbalances remain; future research explores more flexible deployment and conversion methods [PAPER FACT] Sec8.1.1.
- **Real reusability lower than benchmarks:** Open-source benchmarks overestimate reuse (e.g., 90% vs real 50% max) depending on application scenario [PAPER FACT] Sec9.
- **Global communication for elastic SP:** Though CPP avoids it, elastic sequence parallelism still needs global communication group and complicates Conductor; frequent on-the-fly scalability during deployment remains challenging [PAPER FACT] Sec5.1.
- **Future heterogeneous accelerator exploration:** Current A800 flagship not optimal per bandwidth/dollar or bandwidth/watt; future needs GDDR/LPDDR, PIM, hybrid bonding memory-oriented devices for decode memory-bound ops [PAPER FACT] Sec10.
- **Further disaggregation potential:** Attention operator in decode has arithmetic intensity ¡Ø num_heads / num_kv_heads, cannot increase with batch size; separating attention from other linear ops could further improve utilization (preliminary simulated results) and MLA operator from DeepSeek-v2 also promising but not yet implemented [PAPER FACT] Sec10.
- **Scheduling extensions needed:** Developing advanced policy for varying request priorities and different TTFT/TBT SLOs, replication/migration/eviction for partial hits/expiration, dynamic balancing of prefill/decode and idle resource batch offloading all left for future [PAPER FACT] Sec10.
- **Proprietary protection:** Real model not evaluated; dummy model may not capture all nuances [PAPER FACT].

[AGENT INFERENCE]: Authors present Mooncake as production platform handling exponential growth, but acknowledge many problems need exploration (Sec1.2) .

## 13 Inferred Limitations [AGENT INFERENCE]

- **Single model architecture evaluation:** All numbers based on dummy LLaMA2-70B; no validation on MoE (DeepSeek-V2), MLA, or smaller/larger dense models where prefill/decode ratio differs [AGENT INFERENCE].
- **No cost/billing analysis:** Disaggregated CPU/DRAM/SSD/RDMA pooling claims no additional costs via underutilized resources, but no dollar or power cost quantification (e.g., vs adding more GPUs) [AGENT INFERENCE].
- **Availability and fault tolerance:** No discussion of failure handling when prefill or decode node crashes mid-pipeline, or Messenger RDMA failures, or KVCache loss due to eviction ¡ª critical for production [AGENT INFERENCE].
- **Cache eviction not evaluated:** LRU/LFU/LengthAware compared for hit ratio but end-to-end throughput impact of eviction under real memory pressure not measured [AGENT INFERENCE].
- **Prefill_chunk and pipeline group size X tuning:** Optimal X and chunk threshold >1000 not justified with sensitivity analysis; pipeline bubbles for short requests may waste resources [AGENT INFERENCE].
- **SLO multiplier arbitrariness:** TTFT_P90=10¡Á and TBT_P90=5¡Á thresholds are arbitrary for evaluation; real deployment fixed SLOs (30s/0.1s) differ, making simulated gains less transferable [AGENT INFERENCE].
- **No multi-turn conversation state:** KVCache reuse across turns within same session is captured via hash prefix, but long multi-turn where context grows beyond single request not explicitly evaluated for consistency [AGENT INFERENCE].
- **Security/privacy of hash IDs:** Remapped hash IDs preserve caching relationships but may still leak access patterns; no privacy analysis [AGENT INFERENCE].
- **Load prediction uniform td realism:** Assuming uniform decode time across requests ignores variance (output lengths 52 vs 26 in sample); under heavy-tailed distributions prediction error may reintroduce fluctuation [AGENT INFERENCE].
- **No integration with algorithmic KV compression:** Mooncake treats KVCache size as given; orthogonal ShadowKV/KIVI compression could increase effective batch/CPU capacity but not explored jointly [AGENT INFERENCE].
- **Scalability beyond 20 nodes:** No data on scaling to hundreds of nodes, Conductor bottleneck, or RDMA incast for hot blocks even with replication [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Optimal prefill/decode ratio autoscaling:** How to dynamically and automatically adjust number of prefill vs decode instances (and pipeline group X) under rapidly changing workload without service disruption? What control loop minimizes TTFT+TBT SLO violations? [AGENT INFERENCE]
2. **Adaptive kvcache_balancing_threshold:** Can we learn threshold via reinforcement learning or bandits based on real-time network congestion and hit rate, rather than manual tuning? [AGENT INFERENCE]
3. **Request-level output length prediction under overload:** Would lightweight learned predictor (e.g., small LM) achieve sufficient accuracy at acceptable cost to enable more precise early rejection than uniform td, and does it pay off under 2¡Á overload? [AGENT INFERENCE]
4. **Heterogeneous hardware disaggregation:** How to design scheduling when prefill uses compute-oriented accelerators and decode uses memory/bandwidth-oriented devices (GDDR/PIM) with different cost/performance? [AGENT INFERENCE]
5. **Attention offloading:** Does separating attention operator from linear operators (as suggested in Sec10) truly increase throughput in practice on current RDMA/NVLink fabrics, and what granularity (per layer vs per head) is optimal? [AGENT INFERENCE]
6. **Joint optimization with KV compression:** How does Mooncake gain compose with ShadowKV-style low-rank key + value offload or MLA? Could effective cache ratio be increased from 50% to >80% via compression? [AGENT INFERENCE]
7. **Cache hierarchy (DRAM vs SSD):** When CPU DRAM insufficient for 75% more requests, what tiering policy between DRAM and SSD maintains TTFT SLO, and what is prefetching latency impact? [AGENT INFERENCE]
8. **Fairness and priority scheduling:** How to incorporate request priorities (e.g., paid vs free, batch API 24h turnaround at 50% discount as in Sec5.2) into Conductor without starving low priority while maximizing goodput? [AGENT INFERENCE]
9. **Cross-request prefix sharing at scale:** With up to 90% reuse for certain services (papers.cool), can we design global radix-tree-like distributed cache (similar to SGLang) over disaggregated pool without central Conductor bottleneck? [AGENT INFERENCE]
10. **Formal overload theory:** Can we model load fluctuation as control-theoretic delay system and derive stability condition for early rejection with prediction delay, giving provable bounds on wasted prefill computations? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Splitwise [Patel et al. arXiv 2311.18677]:** Early phase-splitting disaggregated architecture, motivated Mooncake at early stage [PAPER FACT] Sec9; concurrent corroboration.
- **DistServe [Zhong et al. arXiv 2401.09670]:** Disaggregating prefill and decoding for goodput-optimized serving, optimizes resource allocation and parallel strategies per stage to maximize GPU goodput [PAPER FACT] Sec9.
- **TetriInfer [Hu et al. arXiv 2401.11181]:** Incorporates chunked prefill + two-stage disaggregation + predictive two-stage scheduling [PAPER FACT] Sec9.
- **ORCA [Yu et al. OSDI 22]:** Iteration-level scheduling for concurrent processing at various stages [PAPER FACT] Sec9.
- **vLLM [Kwon et al. SOSP 23]:** PagedAttention + continuous batching, primary baseline and building block acknowledged [PAPER FACT] Sec8-9.
- **SARATHI-Serve [Agrawal et al. 2403.02310], LoongServe [Wu et al. 2024] (elastic sequence parallelism):** Chunked prefill, elastic SP for long contexts, discussed as alternative to CPP [PAPER FACT] Sec5,9.
- **Ring Attention [Liu et al.], Striped Attention [Brandon et al.], DeepSpeed-Ulysses [Jacobs et al.], US P [Fang et al.], Terapipe [Li et al.]:** Sequence parallelism partitionings requiring per-layer communication vs CPP pipeline boundaries [PAPER FACT] Sec5.1.
- **Prompt Cache [Gim et al. 2311.04934], SGLang [Zheng et al. 2312.07104] RadixAttention, AttentionStore [Gao et al. 2403.19708]:** Prefix caching reuse, LRU radix tree, hierarchical KVCache with cost-effective DRAM/SSD, concurrent with Mooncake but Mooncake adds cache-aware scheduling [PAPER FACT] Sec9.
- **Preble [Srivatsa et al. 2024]:** Prompt scheduling essentially KVCache-centric scheduling, corroborates but Mooncake notes real reusability ~50% vs benchmarks higher [PAPER FACT] Sec9.
- **ShadowKV [Sun et al. 2410.21465], KIVI [Liu et al. 2023], PyramidKV [Cai et al. 2024], GEAR, etc.:** KVCache compression/selection/quantization algorithms orthogonal, benefiting Mooncake by increasing batch size and hit ratio (Sec10 lists 48-69) [PAPER FACT] Sec10.
- **DeepSeek-V2 MLA [DeepSeek-AI 2024], Mamba [Gu & Dao 2024], RWKV, Jamba, YOCO, RecurrentGemma:** Hybrid architectures without KVCache or with increased arithmetic intensity, alternative direction vs disaggregation [PAPER FACT] Sec10.
- **FasterTransformer, TensorRT-LLM, DeepSpeed Inference, FlexGen, FastServe:** Production serving optimizations for scheduling/memory/swap [PAPER FACT] Sec9.
- **vLLM/PagedAttention lineage:** Mooncake explicitly builds on vLLM open-source community [PAPER FACT] Sec9.


## Review Log
Reviewer: Reviewer-1
Problems Found:
- Input/output ratio misreported as ~720; paper trace statistics give 7590/182 ¡Ö41.7 (and Table 2 7955/194, 8088/229, 19019/72). 720 appears nowhere in arXiv:2407.00079v4 abstract/¡ì2; verified arithmetic and corrected. Previous note faithfully copied erroneous 720 ¡ª hallucination risk.
- Cache capacity hit ratio Table 1 values (30% at 1K -> 50% at 50K -> 51% at Inf, LRU best) spot-checked against ¡ì4.2; correct.
- Throughput gains: public ArXiv 20% (0% cache) and 40% (>80% cache) vs vLLM-4M, simulated 50-525% (range, largest 525% in certain 128K/50% cache), real 75% more requests under SLO (TTFT 30s/TBT 0.1s, vLLM 57% meet TBT vs Mooncake ~100%) ¡ª all verified via webfetch highlights and text ¡ì8.1.
- CPP vs TP/SP per-layer communication benefit and layer-wise prefill overlap (max(load,prefill)) correctly captured.
- Overload rejection counts Baseline 4183 vs Early 3771 vs Prediction 3589 (8P+8D 2x) verified Table 3.
Corrections:
- Fixed ratio 720 -> 41.7 with arithmetic note.
- Clarified 800 Gbps as aggregate not per-card single.
- No change to KVCache-centric scheduling (Alg.1 prefix_len/best_prefix_len + T_queue/T_prefill/T_transfer) ¡ª accurately described.
Confidence: Medium (ratio error High confidence; other numbers Medium due to truncated HTML fig values)