# Paper Metadata

- **Title:** DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving [PAPER FACT]
- **Authors:** Foteini Strati (MSR/ETH Zurich), Sara Mcallister (MSR/CMU), Amar Phanishayee (Microsoft Research), Jakub Tarnawski (Microsoft Research), Ana Klimovic (ETH Zurich) [PAPER FACT]
- **Venue:** arXiv preprint arXiv:2403.01876v1 [cs.DC], submitted 4 Mar 2024 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2403.01876 / https://arxiv.org/abs/2403.01876 / HTML https://arxiv.org/html/2403.01876v1 [PAPER FACT]
- **Code:** [NOT REPORTED] (paper does not list public repository URL) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2403.01876 + https://arxiv.org/html/2403.01876v1 (v1, 04 Mar 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Distributed LLM serving is costly and underutilizes accelerators due to three challenges in stateful pipeline-parallel inference [PAPER FACT]. LLM KV cache makes inference stateful (must persist across token steps) [PAPER FACT]; large models (100s GB for 2K seq Fig1) require pipeline + tensor parallelism across many GPUs [PAPER FACT]. Existing systems suffer: (1) pipeline bubbles from bimodal latency (prompt processing compute-bound vs token generation memory-bound) causing GPUs idle [PAPER FACT]; (2) GPU memory overprovisioning by preallocating KV for all microbatches while only one microbatch uses cache at a time [PAPER FACT]; (3) long recovery on failure/preemption — loss of KV cache forces recomputation from scratch, stalls pipeline [PAPER FACT]. No efficient fault-tolerant pipeline serving exists [PAPER FACT].

## 2 Motivation [PAPER FACT]

- **Bimodal latency:** Prompt processing time scales with input size and is compute-bound; token generation time with KV cache is nearly constant and memory bandwidth-bound [PAPER FACT]. Fig2 shows prompt 1.4x to 106x higher than per-token generation (batch8, prompt1000, details Appendix A) [PAPER FACT]. With pipeline parallelism this creates bubbles: Fig3 4-stage example with prompt 2x token (actually up to 106x) shows stalls at Stage1 waiting for Stage4 prompt completion; early stopping of microbatches worsens bubbles when new microbatch's prompt disturbs token steps [PAPER FACT].
- **Memory overprovision:** FasterTransformer preallocates GPU memory for KV of all microbatches to avoid dynamic allocation [PAPER FACT]; but pipeline processes microbatches round-robin, only one active at a time per stage → memory wasted, limiting batch size and model size that can fit [PAPER FACT].
- **Statefulness & failures:** Failures common in large GPU deployments (Meta 50% jobs fail within 16 minutes [Eisenman 22], Microsoft hardware/software failures [Jeon 19]) [PAPER FACT]. Fig4 toy GPT2-1.5B 500 prompt +500 gen, failure after 250 tokens → baseline restart (reprocess prompt + regenerate 250 tokens) increases E2E latency by 1.89x [PAPER FACT]. Pipeline dependency causes cascading idle/timeouts [PAPER FACT].
- **Prompt size growth:** User prompts growing (LongNet 1B tokens) → prompt KV transfer bottleneck if disaggregation attempted [PAPER FACT].
- **Opportunity:** Fast KV cache streaming library can enable disaggregation (separate prompt/token pipelines), microbatch swapping (CPU pool + on-demand GPU), and replication (remote CPU/persistent storage) with minimal overhead [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Pipeline bubbles from prompt-token mixing (challenge 2.2.1).** [PAPER FACT] Y (prompt time per microbatch) >> t (per-token time); processing both in same pipeline leaves stages idle; early exit of microbatches introduces mismatched work, further bubbles [PAPER FACT].
2. **Overprovisioned GPU KV (challenge 2.2.2).** [PAPER FACT] D stages need D microbatches in-flight, each needs M GB; naive allocates D*M on GPU but only M used at a time → wastes (D-1)*M per stage [PAPER FACT].
3. **Failure recovery amplification (challenge 2.2.3).** [PAPER FACT] KV lost on GPU failure; no existing pipeline system checkpoints KV, so all in-flight microbatches restart, redundant compute grows with generated tokens (1.89x example) [PAPER FACT].
4. **Streaming overhead:** KV cache is non-contiguous (Fig6: per-token updates across layers are small scattered regions; key 6D, value 5D tensors with head/batch/seq/layer dims) [PAPER FACT]; naive cudaMemcpy per chunk is expensive; remote transfer via NCCL/MPI/Boost also needs orchestration [PAPER FACT].
5. **Scheduling complexity:** Different prompt/token pipeline depths, batch sizes, and pipeline splits require splitting/merging KV cache chunks at source/destination [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**DéjàVu: Use a unified versatile KV cache streaming library (DéjàVuLib) to implement three synergistic optimizations for pipeline-parallel serving: prompt-token disaggregation with principled resource partition, per-microbatch CPU-GPU swapping, and asynchronous KV replication for fault tolerance [PAPER FACT].**

- **DéjàVuLib primitives:** stream_out/stream_in (high-level: find destinations/sources, handle split/merge based on pipeline depths/batch sizes) → scatter/gather (chunk non-contiguous regions to contiguous transfers) → flush/fetch (copy contiguous chunk via CUDA locally or NCCL/MPI/Boost remotely) [PAPER FACT]. Background CPU thread + CUDA streams parallelize transfers with compute [PAPER FACT].
- **Three optimizations for streaming:** (1) Buffered copies — aggregate many small token updates in temporary GPU buffer then single copy, reusing buffers [PAPER FACT]; 95x improvement over baseline of many cudaMemcpy [PAPER FACT]; (2) Layer-by-layer prompt streaming — stream each layer as computed, pipeline microbatch i streaming with microbatch i+1 compute (like wait-free backprop) [PAPER FACT]; (3) Token streaming parallelization — stream token step i while step i+1 compute; for pipeline, parallelize microbatch i step j streaming with microbatch i+1 step j compute [PAPER FACT]; together +1.4x beyond buffered copies [PAPER FACT]; overall slowdown within 2% vs no streaming (Sec5.1, Appendix D) [PAPER FACT].
- **Disaggregation:** Split D machines into Dp prompt-processing pipeline and Dt token-generation pipeline; each pipeline hosts more layers (D/Dp or D/Dt) increasing per-machine work but eliminating bubbles [PAPER FACT]; KV transferred prompt→token via DejaVuLib; planner computes optimal Dp/Dt to balance throughput and meet memory constraints [PAPER FACT].
- **Microbatch swapping:** Store D*M KV on CPU, only 2*M (or M if D==2) on GPU; before processing microbatch x, swap in its KV (prefetch) while swapping out previous microbatch's delta; transfer fully hidden if transf_i <= t [PAPER FACT].
- **Replication fault tolerance:** Each worker x streams its KV delta to worker (x+1)%N (next stage) asynchronously; also tracks (x,j,t) replication watermark to controller via heartbeat messages; on failure, three-step recovery repopulates lost caches and resumes from last replicated (j,t) [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Overall architecture (Fig5):** Centralized Controller coordinates workers and clients; each DéjàVu Worker has Cache Manager that knows pipeline config (depths, batch sizes, prompt vs token mode) and invokes DéjàVuLib [PAPER FACT]; Controller handles request dispatch and failure detection via periodic heartbeats [PAPER FACT].
- **DéjàVuLib integration (Fig6,7):** Built on FasterTransformer (chosen for tensor+pipeline support; vLLM lacked pipeline at time) [PAPER FACT]. Handles 6D key /5D value tensors; manages layer and seq dimensions splitting. Supports local (CUDA) and remote (NCCL, MPI, Boost) transports [PAPER FACT].
- **Prompt-token disaggregation (§4.2.1):**
  - Workers categorized P-worker (prompt) and T-worker (token) [PAPER FACT].
  - Transfer path: GPU → local CPU → remote CPU, then load to GPU when token worker ready; handles differing pipeline depths / batch sizes via stream_out splitting/merging [PAPER FACT].
  - **Resource planner (Eq1-6):** Given D machines, M GB per machine, L layers, W_i param per layer, C_i prompt KV per layer, K_i token KV per layer.
    - Prompt pipeline: M >= Pn*(C0+W0) → Pn <= floor(M/(C0+W0)), Dp >= ceil(L*(C0+W0)/M) [Eq1] [PAPER FACT].
    - Token pipeline: M >= Tn*W0 + Dt*(Ci+Ki) → Dt >= L*W0/(M - L*(C0+K0)) [Eq2 approx] [PAPER FACT].
    - Throughput model: Y=prompt time per microbatch with D machines, t=per-token time, N= tokens generated. Non-disaggregated inverse throughput Ic = Y+N*t + (D-1)*(Y-t)/D [Eq3] [PAPER FACT]; Disaggregated It = N*D*t/Dt, Ip = m*D*Y/Dp (m>=1 streaming overhead) [PAPER FACT]; Idis = max(It,Ip). Optimal when It=Ip → Dt = D*N*t/(m*Y+N*t) [Eq5], Dp = D*m*Y/(m*Y+N*t) [Eq6] [PAPER FACT]; Benefits if Y/t > (D-1)/(D*(2-m)-1) and m in [1,2) [Eq4] [PAPER FACT]; larger N favors more token machines, larger Y/t favors more prompt machines [PAPER FACT].
  - Simulator-based planner (Appendix B) also searches batch size and data-parallel pipelines to minimize makespan [PAPER FACT].
- **Microbatch swapping (§4.2.2, Fig9):** For D stages, D microbatches in-flight; CPU holds D*M, GPU double-buffer 2*M. Pipeline: when Stage4 processes microbatch1 step T1_1, swap in microbatch2 and swap out microbatch1 delta (x+1)%N and (x-1)%N rule [PAPER FACT].
- **Failure handling (§4.2.3, Fig10):** Each worker x replicates KV delta to x+1%N; receiver thread stores replica; controller tracks watermark (x,j,t) per worker. On failure of x: (1) x+1%N sends replica of x back to x, (2) x-1%N sends its own KV to x (repopulate lost replica at x), (3) controller finds (j,t) that was not yet replicated (e.g., 1C not replicated before fail), (4) broadcasts resume point to all stages, Stage1 replays activations from (j,t) [PAPER FACT]; heartbeat detection → stop all workers → repair [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary throughput/latency:**
  - Normalized latency = E2E latency / #generated tokens (seconds/token), median/ CDF, vs request rate (open-loop Poisson) [PAPER FACT]; used to find sustainable throughput [PAPER FACT].
  - System throughput (requests/s or tokens/s) at given rate/latency SLO; speedup factors reported [PAPER FACT].
  - Makespan (total time to complete trace) and normalized cost (hourly VM cost) (Appendix B) [PAPER FACT].
- **Pipeline bubble / utilization:** Inverse throughput formulas Ic, It, Ip, qualitative bubble overhead (Y/t ratio) [PAPER FACT].
- **Memory:** GPU memory required for various LLMs with 2K seq (Fig1); allocation D*M vs 2*M [PAPER FACT]; batch size feasibility [PAPER FACT].
- **Fault tolerance:** Cumulative latency of microbatch when failure at step 1200 (Fig14), request completions over time with failures at 600/1200/1800 sec (Fig15), increase factors 1.91x baseline vs 1.24x DejaVu, runtime 1.16x shorter with failures [PAPER FACT].
- **Microbenchmarks:** DéjàVuLib streaming slowdown within 2% for local SSD and remote CPU (single batch, prompt500 gen500) [PAPER FACT]; optimization breakdown: buffered copies 95x over baseline, further 1.4x from layer/pipeline optimizations (Fig11) [PAPER FACT]; Appendix E formal condition for swapping benefit: 2*N*t >= sum max(t, transf_i) where transf_i = i*B*Ci / pcie bw [PAPER FACT].
- **Secondary:** Prompt vs per-token time scaling vs batch size and prompt length (Fig16-19 Appendix A, up to 106x) [PAPER FACT]; planner best config tables (Tables2-5) [PAPER FACT].
- **Not reported:** Energy per token, dollar cost absolute, TTFT vs TPOT isolation, P99 tail [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **FasterTransformer [NVIDIA 2023b]:** State-of-art supporting tensor+pipeline; preallocates KV for all microbatches [PAPER FACT]; original does not allow early-finishing requests → batch stuck until last microbatch done at last stage [PAPER FACT].
- **Modified FasterTransformer (baseline in evaluation):** Patched to allow microbatch-level scheduling — whenever microbatch completes at any stage, next microbatch can replace it [PAPER FACT]; used for fair pipeline comparison [PAPER FACT].
- **Baseline (Tensor+Pipeline) in planner (Appendix B):** All D machines in single pipeline with tensor parallel per stage; varies microbatch size b, picks shortest makespan [PAPER FACT].
- **Baseline-DP (Tensor+Pipeline+Data Parallel):** D machines as d pipelines each depth D/d; parallel pipelines serving subsets; varies d and b [PAPER FACT].
- **Other references not directly benchmarked:** vLLM (PagedAttention, but no pipeline at time), FlexGen (single-GPU offload), SpotServe (preemption via 30-sec grace period, not sudden failure), Orca, Sarathi, H2O etc. discussed in Related Work [PAPER FACT].
- **No-streaming ablation:** Single-batch no streaming vs DéjàVuLib to SSD/remote CPU to measure overhead [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models:** GPT2-1.5B for failure toy [PAPER FACT]; OPT family [Zhang 22] (OPT-13B, 30B, 66B) and BLOOM-176B (176B) for main evaluation [PAPER FACT]; also HuggingFace versions adapted for FasterTransformer, half-precision (fp16) [PAPER FACT]; also simulated OPT-30B on V100 [PAPER FACT].
- **Datasets / Traces:**
  - LMSys-chat-1m [Zheng 23] real conversation trace for #generated tokens distribution; prompt size fixed 1000 tokens for Fig12, tokens per microbatch sampled from LMSys assuming equal within microbatch [PAPER FACT].
  - Synthetic homogeneous requests for isolation: prompt 500 gen500 (microbenchmark), prompt500 gen1000 (failure test), prompt1000 sweep (prompt/token time Fig2,16-19) [PAPER FACT].
  - Batching: batch size 1-8 for prompt/token time sweeps; batch size 4/8/16/32 sweep in planner; microbenchmark batch of requests prompt500 gen500 [PAPER FACT].
- **Request arrival:** Open-loop Poisson with varying rates, single client submitting [PAPER FACT].
- **Failure injection:** Single stage failure at token step 1200 for latency CDF (Fig14) and repeated failures at 600/1200/1800 sec for completion trace (Fig15) [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Cloud VMs for main evaluation (Sec5):**
  - 2x A100-80GB per VM, inter-VM network 40 Gbps (for OPT-66B, BLOOM) [PAPER FACT]; also mentioned earlier as 2 GPUs per stage with tensor model parallel within stage [PAPER FACT].
  - VMs with V100-16GB, inter-VM 32 Gbps (for OPT-30B simulation variants) [PAPER FACT]; V100 PCIe3 x16, A100 PCIe4 x16 [Wikipedia 23] [PAPER FACT].
- **Planner simulations (Appendix B):** Models V100-16GB and A100-80GB, 1-4 GPUs per machine (varying tensor parallel degree), 1-16 machines, evaluated makespan/cost [PAPER FACT].
- **Software:** FasterTransformer base, CUDA streams, background CPU thread, NCCL/MPI/Boost for remote [PAPER FACT].
- **Not detailed:** Exact CPU model, DRAM per VM (beyond GPU memory M), OS, CUDA version [NOT REPORTED].

## 10 Main Results [PAPER FACT]

*Numbers from Sec5, Figs11-15,16-19, Appendix B,E.*

- **Microbenchmark overhead (Sec5.1, Fig11, Appendix D Fig27):** DéjàVuLib streaming to local SSD or remote CPU adds **within 2% slowdown** for single-batch prompt500 gen500 without pipeline [PAPER FACT]; Breakdown: **Buffered copies 95x improvement** over naive many cudaMemcpy; layer-by-layer + token parallelization adds **1.4x further** [PAPER FACT]; prompt streaming overhead negligible when many tokens generated [PAPER FACT].
- **Prompt-token disaggregation throughput (Sec5.2.1, Fig12):** With LMSys sample, Poisson arrivals:
  - OPT-66B: **up to 1.88x higher sustainable throughput** (lower median normalized latency) vs modified FasterTransformer baseline [PAPER FACT].
  - BLOOM-176B: **up to 2x higher throughput** [PAPER FACT]; abstract summarises up to 2x [PAPER FACT].
  - Benefits grow with larger prompt sizes (larger Y/t → larger bubbles baseline, inequality Eq4) despite larger KV to stream; streaming fully hidden [PAPER FACT].
- **Planner simulations (Appendix B, Figs20-23, Tables2-5):** Across D=1-16 machines, tensor 1-4 GPUs, A100/V100, OPT-66B/BLOOM:
  - Baseline makespan decreases with more machines but **cost (normalized) increases** due to sublinear scaling; early stops cause sudden makespan spikes at 6,10,14 machines (Fig24) [PAPER FACT].
  - Baseline-DP outperforms single pipeline Baseline by **2.29x** [PAPER FACT].
  - DéjàVu disaggregated gives **4.2x and 2.22x shorter makespan** vs Baseline and Baseline-DP respectively when same #machines [PAPER FACT]; also better cost efficiency up to certain scale [PAPER FACT].
- **Microbatch swapping throughput (Sec5.2.2, Fig13, Appendix E Figs28-31):** For each model/GPU config, enabling swapping allows batch size **2*B** vs B without swapping, increasing throughput **up to 1.8x** [PAPER FACT]; abstract matches [PAPER FACT]. Condition derived: swapping beneficial when transf_i = i*B*Ci / pcie bw does not exceed t (10s-100s ms per token step) [PAPER FACT]; with larger batch or long sequences (e.g., N 1200/2000) overhead can dominate → not beneficial [PAPER FACT]; enables larger models that would not otherwise fit [PAPER FACT].
- **Failure handling (Sec5.2.3, Figs14-15):**
  - Toy GPT2-1.5B example: failure at token250 → **1.89x E2E latency increase** baseline (reprocess prompt + 250 tokens) [PAPER FACT].
  - OPT-66B 4-stage pipeline, homogeneous 500 prompt +1000 gen, failure at step1200: **baseline cumulative latency increase 1.91x**, **DéjàVu increase 1.24x** (Fig14) [PAPER FACT]. Text also states **microbatch latency reduction 1.54x vs non-fault-tolerant** (abstract) [PAPER FACT].
  - Timeline with failures at 600,1200,1800 sec: baseline restarts all active microbatches from scratch; DejaVu resumes from last replicated (j,t) → **1.16x shorter runtime** [PAPER FACT].
  - Single failure overall claims up to **1.54x** microbatch latency reduction vs non-fault-tolerant (abstract, Sec1) [PAPER FACT].
- **Prompt/token time characterization (Appendix A Figs16-19):** Prompt processing vs per-token ~ **1.4x to 106x** ratio depending on model (OPT-13B/66B/BLOOM) batch size 1-8 and prompt length 250-2000 [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder with KV cache; prompt phase compute-bound matrix-matrix, token phase memory-bound [PAPER FACT].
- GPU memory must hold model weights + active microbatch KV; other microbatches can reside on CPU/remote [PAPER FACT].
- Tensor parallelism limited to single node (requires fast interconnect); pipeline parallelism required for cross-node scaling [PAPER FACT].
- Pipeline stages exchange activations; D microbatches in-flight keep pipeline full [PAPER FACT].
- Prompt time Y and per-token time t approximately constant for given model/batch/prompt size; used in throughput formulas [PAPER FACT].
- Replication to neighbor stage (x+1)%N possible with bandwidth to hide; heartbeat detection latency small vs token time [PAPER FACT].
- Streaming overhead factor m can be kept in [1,2) via optimizations; otherwise disaggregation not beneficial [PAPER FACT] (Eq4).
- Request arrivals Poisson; tokens per microbatch sampled from LMSys but assumed equal within microbatch for controlled experiment [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- No explicit Limitations section; Discussion in Conclusion/Related Work implies:
  - Built on FasterTransformer; integration with iteration-level scheduling (Orca/vLLM) is *work in progress* (not yet integrated) [PAPER FACT].
  - Disaggregation planner assumes known D, M, model characteristics and stable Y/t and trace distribution; short-term dynamic shift not discussed [PAPER FACT].
  - Swapping benefit conditional on PCIe bandwidth vs t; for very large batch or very long sequences, transfer not hidden and swapping may hurt (Appendix E explicitly formalizes this boundary) [PAPER FACT].
  - Replication uses extra network (neighbor CPU/memory) but overhead claimed negligible; not evaluated for concurrent multiple failures or network partition [PAPER FACT].
  - Evaluation limited to 4-stage pipeline, 2 GPUs per stage, homogeneous prompt1000; heterogeneous prompts not varied in main throughput figure beyond planner simulation [PAPER FACT].
  - Comparison focuses on FasterTransformer pipeline; no direct comparison with vLLM pipeline (non-existent) or DistServe/Splitwise concurrent works (acknowledged as concurrent) [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Scale gap:** Main end-to-end results use 4 machines (8 GPUs) for 66B/176B; larger clusters (32+ GPUs) not measured physically, only simulated via planner [AGENT INFERENCE].
- **No heterogeneous or spot evaluation:** Promised cost reduction with heterogeneous GPUs (A100/H100) not quantified, unlike Splitwise which specifically evaluates power/cost [AGENT INFERENCE].
- **Iteration-level batching miss:** Unlike Orca/vLLM continuous batching where batch composition changes each iteration, DejaVu processes microbatches round-robin with fixed batch; may be less flexible under dynamic arrivals [AGENT INFERENCE].
- **Replication cost model:** Extra memory for replica (another D*M on neighbor) and network bandwidth for per-token replication not reported as absolute GB/s; could double memory footprint in fault-tolerant mode [AGENT INFERENCE].
- **Single model serving:** No multi-model or LoRA multiplexing; KV streaming library may need extension for prefix sharing [AGENT INFERENCE].
- **Failure model simple:** Only single stage fail-stop; no study of concurrent failures, stragglers, or network partitions; recovery time not separated into detection vs copy [AGENT INFERENCE].
- **Planner needs profiling:** Y, t, Ci, Wi must be profiled per model/GPU; not auto-learned online [AGENT INFERENCE].
- **No end-to-end cost/$ or energy:** Throughput gain reported, but not perf/$ or perf/W vs heterogeneous baseline [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. Can DéjàVuLib be ported to vLLM/PagedAttention with non-contiguous paged KV to eliminate preallocation entirely while keeping swapping? [AGENT INFERENCE]
2. How does disaggregation perform with iteration-level continuous batching and chunked prefill (Sarathi-Serve) where prompt and token already mixed per iteration? [AGENT INFERENCE]
3. What is the optimal replication degree vs overhead: k=1 neighbor vs k=2 for tolerating simultaneous failures, and how does it affect watermark tracking? [AGENT INFERENCE]
4. Could KV streaming be extended hierarchically to tiered storage (GPU→CPU→NVMe→remote DRAM) for 1M context where even CPU cannot hold D*M? [AGENT INFERENCE]
5. How to dynamically adapt Dp/Dt when workload distribution shifts (e.g., chat vs summarization mix) without redeploying pipelines? [AGENT INFERENCE]
6. Would combining swapping with quantization (KIVI, GEAR) or eviction (H2O, Scissorhands) further increase effective batch size beyond 2x? [AGENT INFERENCE]
7. How does heartbeat timeout and detection latency affect tail latency SLO at 99th percentile during failure? [AGENT INFERENCE]
8. Can the same streaming library support disaggregated prefill/decode across heterogeneous clusters (e.g., prefill on H100, decode on cheaper A10) to optimize perf/$ as Splitwise suggests? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **FasterTransformer [NVIDIA 2023b], TensorRT-LLM:** Industry LLM inference with model/pipeline parallel; baseline for DejaVu pipeline [PAPER FACT].
- **vLLM / PagedAttention [Kwon 23]:** Dynamic KV allocation and swap of individual requests to CPU under pressure; DejaVu contrasts by swapping whole microbatches for pipeline and using DejaVuLib explicit streaming [PAPER FACT].
- **FlexGen [Sheng 23]:** Single-GPU offload with swapping of weights/cache; DejaVu targets pipeline multi-node with microbatch granularity [PAPER FACT].
- **Orca [Yu et al. OSDI22]:** Iteration-level scheduling allowing mixed prompt/token batches; DejaVu notes bubble problem remains and integration is future work [PAPER FACT].
- **Sarathi [Agrawal 23]:** Chunked-prefill + piggyback to mitigate prompt-token interference; concurrent to disaggregation, but still colocated [PAPER FACT].
- **Splitwise [Patel et al. 2311.18677] & DistServe [Zhong et al. 2401.09670]:** Concurrent disaggregation proposals. Splitwise focuses on power/cost with heterogeneous GPUs via simulation (limited pipeline); DistServe focuses on goodput-optimized disaggregation with parallelism search and placement; DejaVu distinguishes by pipeline-bubble focus and DejaVuLib + swapping + fault tolerance [PAPER FACT].
- **SpotServe [Miao 23]:** Serves LLMs on spot instances using 30-sec grace period to migrate KV; cannot handle sudden failure vs DejaVu per-token replication [PAPER FACT].
- **H2O [Zhang 23] / LESS / Scissorhands:** KV eviction/compression reducing cache size; orthogonal, can combine [PAPER FACT].
- **Megatron-LM, PipeDream [Narayanan 19/21]:** Pipeline training parallelism foundations [PAPER FACT].
- **Follow-up hierarchical memories (not in paper) [AGENT INFERENCE]:** Mooncake (KVCache-centric disaggregation + chunked pipeline), LMCache, ShadowKV extend disaggregation beyond two phases and integrate RDMA [AGENT INFERENCE].

