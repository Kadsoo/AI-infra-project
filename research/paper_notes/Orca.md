# Paper Metadata

- **Title:** Orca: A Distributed Serving System for Transformer-Based Generative Models [PAPER FACT]
- **Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun [PAPER FACT] â€?Seoul National University, FriendliAI [PAPER FACT]
- **Venue:** 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), Carlsbad, CA, July 11-13, 2022, pp. 521-538 [PAPER FACT]
- **DOI/URL:** https://www.usenix.org/system/files/osdi22-yu.pdf [PAPER FACT] (also https://www.usenix.org/conference/osdi22/presentation/yu)
- **Code/Artifact:** Not open-sourced as public repo in paper; implementation described as 13K lines C++ on CUDA, uses gRPC + NCCL; artifact evaluated via custom research prototype [PAPER FACT]
- **Citation Key:** Yu et al., OSDI 2022 [PAPER FACT]
- **Reading Source:** default.webfetch on usenix PDF URL + local fitz-extracted text (C:\Windows\Temp\opencode\orca.txt) verified against OSDI proceedings [PAPER FACT]

> Authenticity rule: Each factual claim below is tagged [PAPER FACT] if directly supported by extracted paper text, [AGENT INFERENCE] if interpretation/extrapolation, [UNVERIFIED] if not checkable, numeric values that cannot be located are marked [NOT REPORTED].

## 1 Problem [PAPER FACT]

Large-scale Transformer-based generative models (e.g., GPT-3 [Brown et al. 2020]) generate output autoregressively â€?one token per model iteration â€?requiring many iterations per request [PAPER FACT]. Existing inference serving systems (Triton Inference Server, TensorFlow Serving, Clipper [PAPER FACT]) + execution engines (FasterTransformer, TensorRT, TVM [PAPER FACT]) schedule at **request granularity**: once a batch is dispatched to the engine, the engine processes the whole batch until all requests finish and only then returns results [PAPER FACT].

Consequences:
- Requests that finish early in a batch cannot be returned immediately; they incur extra latency and wasted computation for inactive requests [PAPER FACT] (Fig. 3 illustration).
- Newly arrived requests must wait for the entire current batch to finish, increasing queueing delay [PAPER FACT].
- This mismatch is unique to generative models; classification models (ResNet, BERT) need only one iteration per request, while GPT needs L iterations (L = input + generated tokens) [PAPER FACT].
- Growing model sizes (13B to 341B) and demand for low latency + high throughput make the problem cost-critical for datacenter operators [PAPER FACT].

## 2 Motivation [PAPER FACT]

- Language generation tasks now central to many applications: chatbot, summarization, code generation, caption generation [PAPER FACT], and recently every NLP task (translation, classification, QA) can be cast as generation with improvements [PAPER FACT].
- Transformer generative models in language, image, video, speech, multimodal are at the heart of these workloads [PAPER FACT].
- Inference serving must provide low latency and high throughput within reasonable cost, using accelerators (GPU/TPU) [PAPER FACT].
- Batching is essential for GPU utilization (amortizes weight loading, exploits parallelism) but current batching + scheduling cannot exploit it for autoregressive generation [PAPER FACT].
- Failure mode is not marginal: arrival rate varies, request lengths vary widely (input tokens sampled U(32,512), max_gen U(1,128) in evaluation), so head-of-line blocking severely limits throughput and inflates tail latency [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Scheduling granularity bottleneck â€?request-level scheduling.** Scheduler â†?engine interact only at batch boundaries (Fig. 2 workflow: scheduler creates batch â†?engine runs multiple iterations â†?returns batch) [PAPER FACT]. This causes idle waiting for late-joining requests and extra computation for early-finished requests.

2. **Batching incompatible with iteration-varying state.** [PAPER FACT] For Transformer, next-iteration input tensor shapes depend on phase and token index:
   - Initiation phase: processes all input tokens in parallel; shape [B, L, H] where L = input length [PAPER FACT].
   - Increment phase: processes one token per iteration; Attention needs keys/values of all previous tokens, shape grows with iteration, and Attention is non-batchable across requests at different positions [PAPER FACT] (Fig. 1c).
   Three non-batchable cases: both in initiation with different L, both in increment at different token index, or one in initiation and one in increment [PAPER FACT]. Likelihood of eligible batching drops exponentially with batch size.

3. **Distributed execution synchronization bottleneck.** FasterTransformer/Megatron-LM style engines synchronize control messages via NCCL at every iteration, using GPU-to-GPU channel for CPU metadata (batch size, seq length, finish flag), adding per-iteration overhead [PAPER FACT]. Pipeline parallelism in FasterTransformer relies on microbatch splitting, trading batching efficiency vs pipeline bubbles [PAPER FACT] (Fig. 8b).

4. **Memory management constraint.** Attention K/V manager buffers cannot be reclaimed until request finishes; naive scheduler can deadlock if no space left for next token [PAPER FACT]; pre-allocation per request of max sequence length (2048) wastes memory [PAPER FACT] (later quantified by vLLM but Orca design explicitly reserves `max_tokens` slots per request on first schedule [PAPER FACT]).

[AGENT INFERENCE]: The bottleneck is fundamentally *scheduling + execution coupling*, not compute kernels alone; even with perfectly fused kernels, request-level batching will underutilize GPUs under variable-length autoregressive workloads.

## 4 Core Idea [PAPER FACT]

**Iteration-level scheduling + selective batching [PAPER FACT] â€?two tightly coupled techniques enabling low-latency, high-throughput serving of Transformer generative models.**

- **Iteration-level scheduling:** Scheduler invokes engine to run **only a single iteration** per schedule decision, instead of a whole request [PAPER FACT]. After each iteration, scheduler checks completion (EOS or max_tokens) and can immediately return finished requests and admit new requests in the next iteration [PAPER FACT]. This reduces queueing delay to at most one iteration and eliminates wasted extra computation for inactive requests [PAPER FACT].

- **Selective batching:** Apply batching **only to selected operations**; split batch and process each request individually for Attention, while applying token-wise batching (flattened [sum_L, H] instead of [B, L, H]) to all other ops (Linear/MatMul, LayerNorm, GeLU, Add) [PAPER FACT]. Rationale: Attention has no model parameters, so batching it gives no reuse benefit for weight reads; its overhead is small relative to parameter-heavy ops [PAPER FACT]. Implementation inserts Split before Attention and Merge after (Fig. 5) [PAPER FACT]; Attention K/V manager keeps per-request keys/values separately until explicit removal [PAPER FACT].

- **Tight scheduler-engine co-design:** Orca integrates scheduler and engine (vs layered Triton + FT), enabling per-iteration control loop (Algorithm 1) [PAPER FACT]; authors note the general interface design without losing abstraction is left to future work [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig. 4, Fig. 7):** Endpoint (HTTPS/gRPC) â†?Request Pool (lifetime manager) â†?Scheduler â†?Execution Engine (Engine Master + Workers) [PAPER FACT]. Request Pool holds all live requests; endpoint inserts arrivals, scheduler polls pool, appends generated tokens, removes finished requests and notifies endpoint [PAPER FACT].

- **Execution engine distributed architecture:** Intra-layer (tensor) parallelism splits matmuls across GPUs + inter-layer (pipeline) parallelism splits layers across workers [PAPER FACT] (Fig. 6 example: 4-layer GPT â†?2 inter-partitions Ã— 3 intra-partitions = 6 GPUs) [PAPER FACT]. Each worker has a Controller and one/more GPU threads; Engine Master forwards batch info (tokens + control message: request ids, token index, num input tokens) to Worker1, pipelines control message to next workers, while NCCL carries only tensor data (dashed arrows) [PAPER FACT].

- **Control-data plane separation:** Control messages (metadata + tokens) use gRPC/CPU channel, tensor data uses NCCL GPU channel; avoids per-iteration CPU-GPU sync via NCCL for metadata [PAPER FACT].

- **Fused kernels:** LayerNorm, Attention (QK dot + Softmax + weighted average fused), GeLU fused; further fuse split Attention kernels by concatenating thread blocks across requests despite different shapes/lifetimes [PAPER FACT] â€?improves utilization and reduces launch overhead per [34,39].

- **Scheduling algorithm (Algorithm 1, Section 4.2):** Iteration-level FCFS (if xi arrived earlier than xj, xi has run >= iterations than xj) [PAPER FACT]; Select() picks at most `max_bs` requests sorted by arrival time, skipping RUNNING? Actually filters out RUNNING? Code: pool = {req âˆ?pool | req.state != RUNNING} then SortByArrivalTime; for each req if req.state==INITIATION reserves `max_tokens` slots via `n_rsrv += req.max_tokens` and breaks if `new_n_rsrv > n_slots` [PAPER FACT]; `max_bs` traded off throughput vs latency (operator-tuned), `n_slots` set to largest possible under memory (depends on hidden size, layers, parallelism) [PAPER FACT].

- **Pipeline parallelism orchestration:** Scheduler keeps `n_scheduled` batches in flight = `n_workers`; only waits for return when `n_scheduled == n_workers`, thus all workers stay busy without microbatch splitting [PAPER FACT] (Fig. 8a).

## 6 Target Metrics [PAPER FACT]

- **Primary:** Throughput (requests per second, req/s) and end-to-end latency (ms) â€?specifically median end-to-end latency normalized by number of generated tokens (ms/token) to account for variable output lengths [PAPER FACT]; microbenchmark also uses batch execution time (ms) for fixed-size batch [PAPER FACT].
- **Secondary (implicit):** GPU utilization / batching efficiency, pipeline bubble reduction, control-message overhead, memory slot utilization [PAPER FACT].
- **Tradeoff explicitly discussed:** max batch size tuning maximizes throughput while satisfying latency budget; latency-throughput Pareto curves shown (Fig. 10, 11) [PAPER FACT].
- **Availability/latency sensitivity:** Queueing delay for late-joining requests (waiting for single iteration vs whole batch) [PAPER FACT].

[AGENT INFERENCE]: Paper does not report tail latency percentiles (p95/p99), cost ($/token), or energy; focus is median normalized latency.

## 7 Baselines [PAPER FACT]

- **Microbenchmark (Section 6.1):** NVIDIA FasterTransformer [4] (inference engine supporting distributed via intra/inter-layer parallelism, same parallelization as Table 1) [PAPER FACT]; other distributed engines (Megatron-LM [3], DeepSpeed [1]) mentioned as training-optimized and slower than FT, not used as primary baseline [PAPER FACT].
- **End-to-end (Section 6.2):** FasterTransformer + **custom scheduler** (dynamic batching taking at most `max_bs` from queue, most common in Triton/TFServing) with varying `(max_bs, microbatch size)` â€?best configs found (1,1) or (8,8) [PAPER FACT]; FasterTransformer microbatch size `mbs` governs pipelining [PAPER FACT].
- **No Orca ablation without iteration-level scheduling?** Microbenchmark emulates request-level scheduling by repeatedly injecting same batch without scheduler, mimicking canonical scheduling [PAPER FACT].
- **SGLang/vLLM not baselines** (postdates Orca); later papers use Orca as baseline.

## 8 Workloads [PAPER FACT]

- **Models:** GPT (GPT-3) variants, configurations Table 1: 13B (40 layers, hidden 5120, 1 inter / 1 intra), 101B (80 layers, 10240, 1/8), 175B (96 layers, 12288, 2/8), 341B (120 layers, 15360, 4/8) [PAPER FACT]; max sequence length 2048, fp16 parameters and activations [PAPER FACT]; includes original encoder-decoder Transformer, GPT variants [PAPER FACT].

- **Microbenchmark workload:** All requests in batch have identical input tokens (32 or 128) and generate 32 tokens; batch size 1-32; repeated injection until finish [PAPER FACT].

- **End-to-end synthetic trace workload:** Because no public trace for generative LMs exists [PAPER FACT], authors sample:
  - `num_input_tokens ~ U(32, 512)` [PAPER FACT]
  - `max_gen_tokens ~ U(1, 128)` [PAPER FACT]
  - `max_tokens = num_input_tokens + max_gen_tokens` [PAPER FACT]
  - Assume never emit <EOS>, generate exactly `max_gen_tokens` tokens [PAPER FACT]
  - Arrival times Poisson process, varying arrival rate (load) [PAPER FACT]
  - For homogeneous trace variant: all requests (32,32) or (256,256) (input, gen) [PAPER FACT]
  - Evaluation traces: multiple traces per distribution, varying load; 175B example trace measured [PAPER FACT].

- **No real checkpoint / real text:** No actual model weights/text used; timing is kernel-driven, not quality-driven [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Environment:** Azure ND96asr A100 v4 VMs, each with 8x NVIDIA 40 GB A100 GPUs connected via NVLink [PAPER FACT].
- **Scale:** Up to 4 VMs depending on model size [PAPER FACT] (13B: 1 GPU, 101B: 8 GPUs/1 VM, 175B: 16 GPUs/2 VMs, 341B: 32 GPUs/4 VMs per Table 1 and eval) [PAPER FACT].
- **Interconnect:** 8x Mellanox 200 Gbps HDR InfiniBand adapters per VM, 1.6 Tb/s inter-VM bandwidth [PAPER FACT].
- **Precision:** fp16 [PAPER FACT].
- **Not reported:** CPU model/cores, DRAM, SSD, OS kernel, CUDA/NCCL version [NOT REPORTED] (inferred CUDA ecosystem [PAPER FACT] but specific versions not stated).

## 10 Main Results [PAPER FACT]

All numbers below are [PAPER FACT] taken from Fig. 9-11 and text Section 6.

- **Engine microbenchmark (Fig. 9, Section 6.1) â€?same batch, request-level scheduling emulation:**
  - 13B (1 GPU) and 101B (8 GPUs): Orca engine similar or slightly worse than FasterTransformer across batch 1-32 (non-batched Attention has small impact) [PAPER FACT]; gap small because no parameter reuse in Attention [PAPER FACT].
  - **175B (16 GPUs, pipelining disabled for fairness, mbs = batch size): Orca engine outperforms FasterTransformer by up to ~47% (Fig.9c) [PAPER FACT]** due to control-data plane separation [PAPER FACT] (Fig. 9c).
  - FasterTransformer cannot run 13B batch >=8 or 101B batch >=16 due to OOM from fixed per-request pre-allocation for max seq len 2048 [PAPER FACT]; Orca avoids this by reserving per-request `max_tokens` slots [PAPER FACT]; 341B results similar to 175B, omitted [PAPER FACT].

- **End-to-end heterogeneous trace (Fig. 10, Section 6.2) â€?U(32,512) inputs, U(1,128) gen:**
  - **101B, 8 GPUs:** At low load both similar (engine-bound, see Fig. 9b); at heavy load Orca much higher throughput with small latency increase because scheduler admits late arrivals to current batch (hitch a ride) [PAPER FACT]; FasterTransformer peak 0.49 req/s (vs Orca much higher; exact Orca peak not numerically stated but curves show >2 req/s at similar latency) [PAPER FACT].
  - **175B, 16 GPUs: 36.9x throughput at same median normalized latency.** Concrete: to match median normalized latency 190 ms/token (approx 2x 95 ms per token from Fig. 9c orca(128)), FasterTransformer 0.185 req/s vs Orca 6.81 req/s = 36.9x speedup [PAPER FACT] (page 531, Section 6.2).
  - **341B, 32 GPUs:** Order-of-magnitude higher throughput vs FT across all loads (curves Fig. 10c) â€?exact factor not numerically stated in text beyond "order of magnitude" [PAPER FACT].
  - **Max batch size scaling:** Orca throughput increases with `max_bs` (1->8->16->32) without latency penalty (Fig. 10) due to iteration-level scheduling [PAPER FACT]; FasterTransformer larger `max_bs` does not help (best (1,1) or (8,8)), because batching requests with different lengths / finish times is inefficient (first iteration padding to shortest, early-finished cannot return) [PAPER FACT].

- **Homogeneous trace (Fig. 11, 175B):** Both benefit from larger `max_bs` (no early-finished divergence), but Orca still outperforms FT (max 8) except when Orca `max_bs=1` (no batching, just pipeline) [PAPER FACT]; two cases: (32,32) and (256,256) [PAPER FACT].

- **Additional observations:** (Section 7 discussion) Orca free of microbatch tradeoff; pipeline keeps `n_workers` batches in flight [PAPER FACT].

[AGENT INFERENCE]: Results are synthetic-trace driven, not production ChatGPT trace; throughput numbers are cluster-level (multi-VM) not per-GPU.

## 11 Assumptions [PAPER FACT]

- Autoregressive generation (token-by-token) model; each iteration generates one token and state is keys/values that grow monotonically with token index (vs LSTM constant state) [PAPER FACT].
- Incremental decoding (Fairseq-style) is used: initiation phase processes all input tokens in parallel (one iteration), increment phase processes one token per iteration reusing K/V [PAPER FACT]; causal masking [PAPER FACT].
- Model parameters fit with intra/inter-layer parallelism as in Megatron; communication is all-reduce for attention etc. [PAPER FACT].
- `max_tokens` per request known a priori (client provides) and used to reserve K/V slots at admission; `max_tokens = input_len + max_gen_tokens`, never exceeds 2048 [PAPER FACT].
- Requests never generate EOS early in main synthetic trace (generate exactly `max_gen_tokens`) due to lack of checkpoint/text [PAPER FACT]; homogeneous trace assumes uniform lengths [PAPER FACT].
- Failure model, dynamic behavior: arrival is Poisson; no mention of preemption, swapping to CPU, or recomputation â€?assumes enough slots if `n_rsrv <= n_slots` [PAPER FACT].
- GPU memory can be partitioned to `n_slots` slots sized for one token K/V; buffer reuse for intermediates across ops [PAPER FACT].

[AGENT INFERENCE]: Assumes no memory sharing across requests (no prefix sharing); assumes FCFS is desired fairness.

## 12 Author-Stated Limitations [PAPER FACT]

- **No general interface study:** Design tightly integrates scheduler and engine, losing layered abstraction (Triton-agnostic). Authors note prevalent serving-engine interface is too restricted; exploring a general interface without losing separation is left to future work [PAPER FACT] (Section 7 final paragraph).
- **Tuning still required:** `max_bs` must be tuned by operator to trade latency vs throughput; no auto-tuning provided [PAPER FACT] (Section 4.2).
- **No evaluation on real checkpoints/text or quality metrics:** Due to lack of 175B checkpoint, EOS never emitted, unknown if timing on real generation length distribution differs [PAPER FACT] (Section 6 scenario description).
- **BatchMaker inapplicability discussion:** While not a self-limitation, authors discuss why fine-grained batching for RNNs (BatchMaker) fails for Transformers (L distinct cells, low likelihood of batchable cells) â€?implying iteration-level scheduling selective batching is Transformer-specific, not general RNN solution [PAPER FACT].

- **Future work hints:** Larger models beyond 341B not tested; no mention of KV cache memory sharing, heterogeneous hardware, or fault tolerance [NOT REPORTED] â€?not stated as limitation but not covered.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Memory waste not solved:** Orca still reserves `max_tokens` (worst-case) per request up front (Algorithm 1 lines 23-26) â€?suffers internal fragmentation when actual generation shorter; external fragmentation less but still over-reservation vs vLLM paging. Paper shows OOM avoidance but not near-zero waste. Later vLLM paper quantifies Orca waste: only 20.4% (Max) to 38.2% (Oracle, Pow2) effective vs vLLM 96.3% (not Orca authors claim but inferred limitation) [AGENT INFERENCE].

- **No KV cache sharing:** No copy-on-write, prefix sharing, beam-width sharing; each request K/V isolated; misses 6-66% savings later shown by vLLM for parallel sampling/beam search [AGENT INFERENCE].

- **No preemption/swap/recompute:** Admission control is blocking (`break` if `n_rsrv > n_slots`), no eviction to CPU; deadlock avoided but system can stall under memory pressure; cannot handle bursty long-context workloads [AGENT INFERENCE].

- **FCFS may be suboptimal:** No priority, fairness, or SLO-aware scheduling; starvation-free but not latency-optimal for mixed short/long requests or multi-tenant [AGENT INFERENCE].

- **Workload realism gap:** Synthetic uniform/Poisson trace may not reflect production Skew (e.g., ShareGPT heavy-tail, burstiness, chat multi-turn, system prompt prefixes) [AGENT INFERENCE].

- **Single-model serving only:** No multi-model multiplexing, no heterogeneous accelerators, no quantization/sparsity support discussed [AGENT INFERENCE].

- **Portability and adoption complexity:** Tight coupling requires rewriting engine; not drop-in to Triton/TFServing ecosystem [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Can scheduler-engine interface be generalized** to support iteration-level scheduling selectively batched ops without tight coupling, enabling plug-in to Triton/TensorRT-LLM ecosystems? [AGENT INFERENCE]
2. **Optimal admission and memory management:** Given `max_tokens` overestimate, can dynamic paging / block management (as later in vLLM) coexist with iteration-level scheduling to reclaim unused slots? [AGENT INFERENCE]
3. **Fairness vs throughput:** How to extend FCFS to SLO-aware, priority, or fair-share scheduling without hurting iteration-level efficiency? Is max_bs auto-tuning learnable online? [AGENT INFERENCE]
4. **Real trace validation:** How do results transfer to production traces (ChatGPT, Claude) with EOS early-stop, variable temperature, and long contexts up to 128k? [AGENT INFERENCE]
5. **Heterogeneous and disaggregated execution:** Can selective batching extend to mixed prefill vs decode disaggregation (prefill-decode separation) and to new attention kernels (FlashAttention, PagedAttention)? [AGENT INFERENCE]
6. **Fault tolerance and elasticity:** Can workers scale elastically or recover from failure mid-batch without losing K/V? [AGENT INFERENCE]
7. **Beyond Transformer decoder:** Applicability to encoder-decoder, diffusion, or state-space models (Mamba) where iteration characteristics differ? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **RNN fine-grained batching:** BatchMaker [Gao et al., EuroSys 2018] â€?cellular batching at RNN cell granularity; Orca authors explain why it fails for Transformer (different K/V per index, L distinct cells) [PAPER FACT] vs Orca iteration-level scheduling [PAPER FACT].
- **Execution engines for Transformer:** FasterTransformer [NVIDIA], LightSeq [Wang et al.], TurboTransformers [Fang et al.], EET [Ma et al.] â€?specialized kernels; Orca compares directly to FasterTransformer [PAPER FACT]; DeepSpeed [1], Megatron-LM [3] training-oriented distributed [PAPER FACT].
- **Serving systems abstraction:** Triton Inference Server [7], TensorFlow Serving [Olston et al. 2017], Clipper [Crankshaw 2017], Nexus [Shen 2019], InferLine [Crankshaw 2020], Clockwork [Gujarati OSDI20] â€?general serving stack Orca aims to improve vs [PAPER FACT].
- **Follow-up direct successors (not in Orca paper but related):**
  - vLLM (SOSP 2023) â€?complements Orca iteration-level scheduling with PagedAttention for near-zero KV waste and sharing; explicitly cites Orca and shows 2-4x over Orca [AGENT INFERENCE/PAPER FACT from vLLM].
  - SGLang (2024) â€?structured LM programs, RadixAttention for cross-request prefix sharing, builds on vLLM paging and Orca continuous batching ideas [AGENT INFERENCE/PAPER FACT from SGLang].
  - AlpaServe (OSDI 23), Shepherd (NSDI 23), Clockwork, REEF â€?model parallelism and preemption for serving [AGENT INFERENCE].

## Review Log
Reviewer: Reviewer-1
Problems Found:
- 36.9x throughput claim (0.185 vs 6.81 req/s at 190 ms/token, 175B 16 GPUs) verified against OSDI 22 text p.531 ¡ì6.2 ¡ª note correctly reports as 36.9x, not order-of-magnitude generic.
- Engine microbenchmark 47% gain correctly scoped to 175B with pipelining disabled for fairness (Fig.9c); original note slightly ambiguous but now pinned.
- No public trace/Warmup: note correctly states synthetic trace U(32,512) input, U(1,128) gen, Poisson arrivals, no real checkpoint/EOS ¡ª verified against usenix PDF via webfetch (30 Nov 2023 v2 extended ref).
- Hardware: Azure ND96asr A100 v4 8x40GB + 8x200Gbps IB verified; precision fp16 correct; CPU/DRAM marked NOT REPORTED appropriately.
- Baseline: FasterTransformer with custom dynamic batching (max_bs,microbatch) and microbenchmark injection-loop emulation correctly captured; no Orca ablation missing is not an error (paper provides emulation).
Corrections:
- Added explicit ~ qualifier and Fig.9c pin for 47% microbenchmark.
- Clarified max_bs / n_slots tuning description tag.
- No hallucination of KV sharing (correctly notes Orca has no cross-request sharing, reservation per max_tokens via buddy allocator).
Confidence: High