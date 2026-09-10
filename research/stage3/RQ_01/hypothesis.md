# Research Question

> When does P/D disaggregation lose to colocated execution because KV movement and burst dynamics dominate?
>
> — quoted verbatim from `knowledge/stage3_candidates.md`, RQ-1 (line 20), including its failure scenario: "cross-node/low-bandwidth placement, fragmented KV, mixed prompt lengths, and a burst that turns a cache-affine P/D pairing into a queue hotspot."

# Revision History (post-adversarial-review)

> Changes applied 2026-08-27 after `design_review.md` (critic verdict: require rework). Each change is traceable to a review issue:
>
> - **R1 (critic #2, self-contradiction):** the ≤5% P50-TTFT conjunct was removed from the Primary Hypothesis (it was arithmetically incompatible with the ≥30% transfer-fraction claim at 25 Gbps). Replaced by an aggregate-throughput equality guard that is *reported*, not a hypothesis conjunct.
> - **R2 (critic #1/#A, unreachable kill lines):** H1 kill lines re-derived to be reachable on the stated hardware: measured P95 transfer fraction <30% at 25 Gbps 8k–13k prompts (the claim's own threshold), or a 25 Gbps-vs-NVLink P95 TTFT gap <25% of the NVLink value. The old lines (<10% fraction, <10% gap) required >130 Gbps on a 25 Gbps link and were physically unreachable.
> - **R3 (critic #3, H2 untestable):** H2 is re-scoped to isolate the queueing mechanism at transfer ≈ 0 (NVLink placement, where H1's transfer term is negligible). The 25 Gbps transfer enters H1 (transfer materiality) and H3 (joint condition) only. The load-balanced discriminator now runs at the same (NVLink) bandwidth so a spurious kill branch cannot fire.
> - **R4 (critic #5, SLO arithmetic):** SLO tuple changed to TTFT P95 ≤ 4.0 s (feasible for P/D at the matched rate); the double-allocation rerun is pre-registered as a first-class cell.
> - **R5 (critic #6, hidden variables):** colocated dispatch = FCFS + join-shortest-queue; vLLM commit pinned; node-identity balance required for the joint cells.

# Motivation

The P/D value premise rests on two measurements, both conditional in the corpus. DistServe reports KV transmission as **<0.1% of total latency** and **>95% of requests under 30ms transmission despite 25 Gbps cross-node bandwidth** — but only because its low-affinity placement algorithm forces prefill and decode stages of the same request to co-locate on one node and transfer over NVLink (600 GB/s). The same note quantifies the exposed cost: a 512-token request on OPT-66B moves **1.13 GB of KV**, requiring **90 Gbps to hide at 10 rps** while the physical cross-node link offers 25 Gbps. Splitwise achieves **<7% of prompt time** with layer-wise async transfer (~8ms A100 / ~5ms H100 non-overlapped), but only with MSCCL++ one-sided put; its serialized path costs **+64% second-token latency** and up to **3% E2E**, and its heterogeneous variant (H100→A100) explicitly assumes InfiniBand may need to be replaced by RoCE/Ethernet with **10× lower bandwidth**. Both systems hide transfer by construction; neither reports the unhidden case.

The same corpus shows the transfer path is fragile and implementation-dependent across a wide range. FlowKV measures a **13k-token prompt whose NCCL/PagedAttention KV transfer occupies ~1/4 of end-to-end latency**, driven by fragmented (L,2,B,H) block layout causing **23,469→1 NCCL call reduction** (96.8% single-machine, 92% cross-machine) and reports Mooncake's RDMA path **failing outright at 10k/100** input/output, with vLLM-Disagg at **1.737s transfer**. Beluga's RDMA characterization finds **~75% of a 10.55µs transfer is synchronization overhead** (2.68µs actual data movement), a **sglist limit of 30 entries against 128 non-contiguous chunks** per Qwen-32B block, sparse 16-token loads at **211µs vs 5260µs** (95.9% reduction), and a Mooncake-style RDMA baseline at **76.8s TTFT with 16-token blocks vs 13.0s at 256-token blocks**. Transfer fraction in this corpus spans from <0.1% (DistServe, NVLink-co-located) to ~25% of E2E (FlowKV, NCCL) to failure — exactly the conditioning recorded in assumption A5 (Medium confidence: "true for NVLink intra-node, fragile cross-node/heterogeneous").

The burst side of the claim is equally conditional. Mooncake's Conductor predicts **TTFT = T_queue + T_prefill + T_transfer**, where transfer time depends on **network congestion, not just size**; its trace shows **>50% of cache blocks never reused while hot blocks are accessed tens of thousands of times**, its **kvcache_balancing_threshold is manually tuned**, and under overload its prefill and decode pools enter **anti-phase load fluctuation** (mitigated only by prediction-based early rejection: rejections 4183→3771→3589). DistServe's own M/D/1 model **Avg_TTFT = D + R·D²/(2(1−R·D))** has a queue term that diverges as utilization approaches 1 — a burst pushes a cache-affine node past the offline-fitted operating point that DistServe assumes is "stable over hours/days" (A6: fragile for tail/overload). Mooncake additionally shows a static P/D partition can be suboptimal even in steady state: **[2P+2D] yields worse TTFT than [3P+1D] despite more decode nodes**. None of these systems varies burstiness and bandwidth jointly; all fix one side.

The colocated alternative is not a strawman. Sarathi-Serve's chunked prefill plus token-budgeted stall-free batching bounds P99 TBT without any KV movement, delivering **2.6×–5.6× capacity over vLLM/Orca at equal SLOs** (tau=512 strict / 2048 relaxed, ~25% chunking overhead at 512), and its authors explicitly note that disaggregation "requires KV migration and underutilizes prefill GPU memory" while leaving the quantitative crossover "for future work". DistServe's authors concede the mirror image: resource-constrained scenarios favor simpler colocated deployment. Both sides of the boundary are asserted, neither is measured against the other under a matched harness.

Stage 2B's synthesis makes this the highest-value conditional cell: bottleneck B5 classifies P/D as an architecture that trades prefill/decode interference (B5) against placement and data-movement cost (B4); tension T4 (P/D vs data movement) lists the breakdown condition as "cross-node/low-bandwidth placement, fragmented KV, long prompts, rapid load changes, or insufficient transfer overlap"; tension T7 (throughput vs tail SLOs) adds "high arrival rate, a mix of prompt lengths, aggressive batching, a tighter percentile target". Fragility table row 2 ("a prefill/decode phase split remains beneficial") and row 4 ("arrival/output/reuse distributions are stable enough") are both marked worth testing, review section 7 item 1 retains "P/D architecture versus transfer/queue/topology crossover" as the primary research space, and the opportunity matrix's high-signal cell 1 is exactly "P/D disaggregation × P95/P99 × bandwidth — test whether traffic burst and topology reverse a P/D placement decision."

This stage therefore asks a purely existential question: does the failure phenomenon exist at all, and at what measurable thresholds? It proposes no placement policy, no routing heuristic, no new algorithm — only a matched experiment in which two documented configurations (DistServe-style P/D with Mooncake-style cache-affine routing; Sarathi-style colocated chunked prefill) are compared across a bandwidth × burstiness × prompt-length matrix. A null result is as informative as a positive one: it would shrink the research space and end the architectural debate without inventing a solution (stage2_final_review §9).

# Primary Hypothesis

Under cross-node placement at ≤25 Gbps interconnect with mixed prompt lengths (1k-class and 8k–13k-class) and bursty arrivals (inter-arrival-time coefficient of variation CV ≥ 2), the KV movement plus cache-affine queueing mechanism (transfer time + queue time on the TTFT critical path) becomes ≥ 30% of E2E TTFT at P95, causing the P/D configuration's P99 TTFT to exceed 1.5× the colocated chunked-prefill baseline at equal GPU allocation and equal SLO, while aggregate throughput at equal load stays within 5% of the baseline (the throughput-equality clause is a reported guard, not a hypothesis conjunct).

This is negatable: if either the ≥30% transfer+queue fraction or the ≥1.5× P99 TTFT gap fails to appear at the stated conditions, the hypothesis is false as stated.

# Sub-Hypotheses

**H1 — KV transfer becomes material under cross-node/low-bandwidth placement with long prompts.**
With prompts ≥ 8k tokens and cross-node transfer at ≤ 25 Gbps without NVLink co-location, the measured transfer fraction of E2E TTFT reaches ≥ 30% at P95 (≥ 15% at P50) for the P/D configuration, whereas the same configuration at NVLink-class placement keeps the P95 transfer fraction ≤ 5%. Kill condition (revised R2, reachable on stated hardware): at 25 Gbps with 8k–13k prompts and matched load, the measured P95 transfer fraction stays < 30% of E2E TTFT (measured with the corrected prefill calibration), or the P95 TTFT difference between 25 Gbps and NVLink placements is < 25% of the NVLink value — transfer is then not material enough to justify the "KV movement dominates" mechanism.

**H2 — A burst turns the cache-affine P/D pairing into a queue hotspot (tested at transfer ≈ 0).**
At NVLink placement (transfer negligible, isolating the queueing mechanism) with CV = 2–3 arrivals and cache-affine longest-prefix pairing, the queue time (T_queue + transfer wait) reaches ≥ 20% of E2E TTFT at P99 and P99 TTFT reaches ≥ 1.5× the colocated baseline, while at CV = 1 the same pairing keeps P99 TTFT ≤ 1.1× colocated (affinity pays in steady state). Kill condition (revised R3): at CV = 3 at NVLink, cache-affine P/D shows P99 TTFT ≤ 1.1× colocated with P99 queue fraction < 15% of E2E TTFT; or load-balanced P/D routing at CV = 3 at the same bandwidth reproduces the ≥ 1.5× tail, which would mean the hotspot is a burst-load artifact, not a pairing artifact. (The 25 Gbps transfer term is deliberately excluded from H2's test bandwidth; it is H1's and H3's mechanism.)

**H3 — Under the joint condition, the colocated/chunked baseline overtakes P/D.**
At the H1 condition (25 Gbps) combined with the H2 condition (CV ≥ 2, cache-affine routing), the colocated chunked-prefill baseline attains ≥ 100% of the P/D configuration's SLO-compliant goodput at equal GPU count, or P/D needs ≥ 1.25× the colocated GPU allocation to reach parity; at the benign condition (NVLink, CV = 1) the ordering is reversed with P/D at ≥ 1.3× colocated goodput. Kill condition: at the joint condition, P/D sustains ≥ 100% of colocated goodput at equal GPU allocation with P99 TTFT ≤ 1.1× colocated — the baseline never overtakes, and observable consequence 3 of RQ-1 fails.

Decomposition justification: the three hypotheses are the three independently measurable mechanisms of the failure scenario (transfer materiality, queue-hotspot formation, and the resulting baseline overtake). Each can be killed without killing the others, and each maps to a distinct instrument (transfer fraction, queue fraction, goodput/GPU ratio). Merging them would make the experiment unable to localize which mechanism fails.

# Independent Variables

- **IV1 — Placement/bandwidth (H1, H3):** three levels: (a) intra-node NVLink-class (600 GB/s), (b) cross-node 100 Gbps, (c) cross-node 25 Gbps (DistServe testbed class; FlowKV's cross-node hints at 25–50 Gbps). P/D stages never co-located on the same node at levels (b)/(c); NVLink co-location constraint (DistServe Alg2) is off at (b)/(c) and on at (a).
- **IV2 — Arrival burstiness (H2, H3):** inter-arrival time CV = 1 (Poisson; the corpus default used by DistServe/Splitwise/Mooncake/Sarathi), CV = 2, CV = 3 (ON/OFF burst with fixed mean rate).
- **IV3 — Prompt length mix (H1, H2):** unimodal short (median ~1k, Splitwise conversation class) vs mixed bimodal (1k-class + 8k–13k-class; FlowKV's 13k example, Mooncake's 7590-token average).
- **IV4 — P/D routing policy (H2, H3):** cache-affine longest-prefix pairing (Mooncake-style, choose instance with shortest predicted TTFT) vs load-balanced pairing. Both are documented corpus policies; this is a measured comparison, not a proposed design.

Nothing else is varied: output length distribution, request count, load mean, and trace content are fixed across all cells.

# Dependent Variables

- **TTFT P50/P95/P99** — the primary anchor: transfer and queue time land inside TTFT (H1/H2/H3 thresholds).
- **TPOT P50/P95/P99** — decode-side manifestation of the failure scenario (mixed-batch degradation, weight-duplication HBM pressure); distinguishes decode degradation from prefill-path effects.
- **Transfer time fraction of E2E TTFT, P50/P95** — instrumented on the critical path (inside the TTFT interval), not wall-clock overlap; H1's direct anchor.
- **Queue time fraction of E2E TTFT, P95/P99** — T_queue plus transfer wait at the selected P/D pair; H2's direct anchor.
- **SLO goodput (max sustained RPS at attainment ≥ 95%)** — H3's comparison metric.
- **GPU allocation (GPUs per configuration)** — H3's resource-equivalence term; also exposes P/D's 2× weight-duplication effect on batch capacity.
- **Cache hit rate (prefix match ratio)** — not an anchor; a guard metric to detect routing artifacts (see Confounders).

# Control Variables

- **Model:** Llama-3.1-8B-Instruct (FlowKV's model; fits 2 GPUs; GQA keeps KV sizes realistic). Rationale: the corpus contains transfer measurements for this class; larger models would conflate parallelism effects with the tested mechanism.
- **GPU type:** A100-SXM4-80GB, NVLink intra-node (DistServe/Splitwise testbed class).
- **Precision:** FP16.
- **Serving framework:** one engine (vLLM-lineage, identical kernels and scheduler code); the P/D configuration differs from colocated only in phase split and placement — no third-party system substitution across cells.
- **Request set:** the same 5,000-request trace, replayed in identical order to every cell; per-cell post-hoc verification that input and output length distributions match (mean within 1%).
- **Random seed:** fixed for length sampling, arrival generation, and KV block allocation order (allocation order changes fragmentation, FlowKV's 23,469-call phenomenon).
- **Max batch size / token budget:** colocated tau fixed at one value (e.g., 1024 tokens, Sarathi class); P/D prefill batch cap and decode batch cap fixed.
- **SLO tuple:** fixed before measurement — TTFT P95 ≤ 4.0 s, TPOT P95 ≤ 0.15 s, attainment ≥ 95% (revised R4: P/D-feasible at the matched rate so the "both fail" outcome remains diagnostic); if both configurations violate the SLO at the joint condition, the double-allocation rerun (2× GPUs for both) is a pre-registered first-class cell, not an exception.
- **Tokenizer and generation:** fixed tokenizer; greedy decoding or fixed temperature + fixed seed; max output tokens fixed at 200 (Mooncake's real-workload output class, ratio ~41.7).
- **Network:** dedicated links; background traffic verified below 2% of link capacity during runs; NIC class and throttle method (netem/tc) recorded and verified with a netperf probe per cell (revised R5).
- **Colocated dispatch:** colocated cells dispatch arrivals via FCFS + join-shortest-queue among the two instances (pre-registered; any other dispatch changes colocated P99 by construction — revised R5).
- **Engine version:** vLLM commit pinned and recorded (disaggregated prefill/decode is version-sensitive experimental code — revised R5).
- **Node identity:** the joint-condition cells (25 Gbps, D on node B) are balanced by rerunning them with D on node A, so machine-level identity is not confounded with placement (revised R5).

# Confounders

- **Batching/chunk-size differences:** colocated chunk size (tau) vs P/D prefill batch size change per-iteration latency and queueing independently of transfer. Detect: report token counts per iteration; rerun the critical cells at ±1 tau level and require the observed ordering to be monotone across tau.
- **Kernel compilation / CUDA graph capture:** first-call JIT inflates early requests asymmetrically. Neutralize: 200-request warmup per configuration before the measurement window.
- **GPU clocks and power state:** thermal/clock drift changes both configs' latency. Neutralize: lock clocks (`nvidia-smi -lgc`), log power draw, discard the first 10% of the measurement window.
- **Request length imbalance across conditions:** if one configuration receives longer prompts, all TTFT comparisons are void. Detect: KS-test the input/output length distributions per cell; require mean length within 1% and distributional match, or rerun the cell.
- **CPU/control-plane bottleneck:** the global scheduler (Mooncake Conductor-class) or the transfer control path (Beluga: ~75% of 10.55µs is sync overhead) can cap throughput or inflate latency without any P/D mechanism. Detect: report scheduler CPU utilization (< 60% required) and break transfer time into data movement vs synchronization; if CPU-bound, rerun on more cores before any conclusion.
- **Scheduler policy differences:** FCFS vs priority rules change tails independently of the phase split. Neutralize: identical FCFS dispatch semantics in both configurations; policy is a control, never a variable.
- **Network contention:** external traffic depresses realized bandwidth (Mooncake: transfer time depends on congestion, not size). Neutralize: dedicated NIC/VLAN; measure realized bandwidth with a netperf-class probe during each cell and record it with the results.
- **Cache-affinity routing artifacts:** hit-rate differences (P/D's prefix cache vs a cache-less colocated baseline) masquerade as queue effects. Detect: report hit rate per cell; if the P/D hit rate exceeds the colocated baseline's by > 10 points, enable the equivalent prefix cache in the colocated baseline (Mooncake: real reuse ≈ 50%, benchmarks claim 90%; Beluga: HBM hit ratio 14.6%) and rerun.
- **KV fragmentation:** allocation order and block layout change transfer call counts (FlowKV: 23,469→1; Beluga: sglist 30 vs 128 chunks). Neutralize: fixed allocation seed; instrument per-request NCCL/RDMA call counts as a sanity metric.
- **Weight duplication:** P/D holds two full model copies, shrinking HBM for KV and decode batches; this can lower throughput via capacity, not transfer. Detect: report HBM utilization and KV capacity per cell; if utilization differs by > 10 points at equal load, attribute the effect through the GPU-allocation dependent variable, not the transfer mechanism.
- **Cache warm/cold state:** first-run (populate) vs pre-populated (hit) scenarios differ fundamentally (Beluga evaluates both). Neutralize: run both scenarios in every cell and draw conclusions only within a matched scenario.

# Expected Observation If True

- **H1:** transfer fraction of E2E TTFT (P95) rises from ≤ 5% at NVLink placement to ≥ 30% at 25 Gbps for 8k–13k prompts; P95 TTFT of P/D at 25 Gbps is ≥ 1.5× its own NVLink value; absolute transfer time moves from the ~0.04–0.06s class (FlowKV's optimized single-machine 8k transfer, 0.0447s) into the ~1.3–2.0s class (vLLM-Disagg 1.34s at 8k; FlowKV's Mooncake baseline failing at 10k).
- **H2:** at NVLink placement (transfer ≈ 0), switching arrival CV from 1 to 3 with cache-affine routing raises the P99 queue fraction of E2E TTFT from < 10% to ≥ 20% and P99 TTFT from ≤ 1.1× colocated to ≥ 1.5×; the load-balanced P/D cell at CV = 3 stays ≤ 1.2× colocated, isolating the pairing mechanism (the analogue of Mooncake's anti-phase fluctuation and its 2P+2D-vs-3P+1D reversal, now measured under controlled burstiness and free of transfer confounding).
- **H3:** at the joint condition (25 Gbps, CV = 3, mixed prompts, cache-affine), colocated goodput ≥ P/D goodput at equal GPUs, or P/D requires ≥ 1.25× GPUs for parity; at the benign cell (NVLink, CV = 1) the ordering is reversed with P/D ≥ 1.3× colocated goodput — the crossover is a measured crossing, not an extrapolation.

# Falsification Condition

- **H1 is dead** if, at 25 Gbps cross-node with 8k–13k prompts and matched load, the measured P95 transfer fraction of E2E TTFT remains < 30% (measured with the corrected prefill calibration), or the P95 TTFT gap between 25 Gbps and NVLink placements is < 25% of the NVLink value (revised R2 — both lines reachable on the stated hardware). KV movement then never becomes material enough in the tested topology range, and the first leg of RQ-1's failure scenario does not exist.
- **H2 is dead** if, at NVLink placement with CV = 3 and cache-affine routing, P99 TTFT ≤ 1.1× colocated and the P99 queue fraction < 15% of E2E TTFT; or if load-balanced routing at CV = 3 at the same bandwidth reproduces the ≥ 1.5× tail, since the hotspot would then be a burst-load artifact rather than a cache-affine pairing artifact, and no placement decision could avoid it (revised R3).
- **H3 is dead** if, at the joint condition, P/D attains ≥ 100% of colocated SLO goodput at equal GPU allocation with P99 TTFT ≤ 1.1× colocated — the colocated baseline never overtakes within the tested range, and RQ-1's observable consequence 3 fails even though mechanisms H1/H2 may hold.

# Ambiguous Outcome

- **Both configurations violate the SLO at the joint condition:** this indicates resource-allocation insufficiency (both under-provisioned), not a mechanism difference. Follow-up: double the GPU allocation for both configurations and rerun; conclude only if an ordering emerges at matched allocation.
- **Transfer fraction ≥ 30% but P99 TTFT gap < 1.1×:** the transfer is material but hidden by overlap (Splitwise's layer-wise async, <7% prompt time, is the prototype). The 30% fraction must be measured on the TTFT critical path; if the observed traffic is off-path, the follow-up is to serialize transfer (Splitwise's serialized-vs-layer-wise contrast, +64% vs +16.5% second token) to expose the hidden cost. H1's consequence is void until the serialized check is done.
- **Both configurations meet the SLO at all tested loads, with P/D tails merely higher:** the phenomenon is then a cost question, not an SLO question. Follow-up: raise load until one configuration fails, then re-examine; concluding from a range where neither fails would be uninformative.
- **Hit-rate mismatch drives the reversal:** if P/D's cache affinity produces hits the colocated baseline cannot (no prefix cache), the result is a routing artifact, not a P/D-vs-colocated mechanism. Follow-up: enable the same prefix cache in the colocated baseline and rerun the critical cells.
- **Results invert between cache-populate and cache-hit scenarios:** hit-state is the operative condition. Follow-up: restrict all claims to the matched scenario and treat the other as a separate regime.
- **Queue fraction rises but goodput stays equal:** the burst is absorbed by scheduler headroom rather than hitting the pairing. Follow-up: bound the queues (capacity-limited scheduler) before drawing any mechanism conclusion.
