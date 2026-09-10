# experiment_plan.md — RQ-1 Stage 3A: Minimum-Cost Kill Sequence

# Preamble — Decision Pipeline

| Gate | Experiment | Cost tier | Kills | Runs only if |
|---|---|---|---|---|
| E1 | Transfer-materiality cost model + trace analysis (no GPU) | 1 — offline trace analysis | **H1** | always (first) |
| E2 | Calibrated event-driven queue simulator (no GPU) | 2 — simulation | **H2** | E1 completes (uses E1's calibrated constants) |
| E3 | KV-transfer microbenchmark (3× A100, no serving) | 4 — small microbenchmark | **H1** (hardware-confirmed) | E1 survived H1 |
| E4 | vLLM-lineage harness: P/D vs colocated, real serving | 5–6 — existing harness, minimal scheduler-policy hook | **H1, H2, H3** | E1 AND E2 both survived |
| E5 | GPU-allocation sweep for H3 parity | 5–6 | **H3** (≥1.25× leg) | E4 supported H3 |

No experiment tests all hypotheses at once as a first step. A negative verdict at any gate stops the pipeline: the failure scenario decomposes into transfer materiality (H1) and queue-hotspot formation (H2), each independently killable without GPU serving; only the joint overtake (H3) requires real serving, and only after both mechanisms survived. The P/D configuration (DistServe-style, 1 prefill + 1 decode instance) and the colocated baseline (Sarathi-style chunked prefill, tau = 1024) are EXISTING configurations; cache-affine vs load-balanced routing are experimental conditions drawn from Mooncake's documented policies, never "our proposal".

**Hypothesis contract (locked, verbatim thresholds — used below; revision history in hypothesis.md):**
- **H1:** ≥8k prompts, cross-node ≤25 Gbps, no NVLink co-location → P95 transfer fraction ≥ 30% of E2E TTFT (≥ 15% at P50); same config at NVLink-class keeps P95 fraction ≤ 5%. Kill (revised, reachable): P95 fraction < 30% measured at 25 Gbps with corrected prefill calibration, or P95 TTFT gap(25 Gbps vs NVLink) < 25% of NVLink value.
- **H2:** at NVLink placement (transfer ≈ 0, isolating queueing): CV 2–3 + cache-affine longest-prefix pairing → queue time (T_queue + transfer wait) ≥ 20% of E2E TTFT at P99 and P99 TTFT ≥ 1.5× colocated; at CV = 1 same pairing keeps P99 ≤ 1.1× colocated. Kill: at CV = 3 (NVLink) cache-affine P99 ≤ 1.1× colocated with P99 queue fraction < 15%; or load-balanced routing at CV = 3 at the same bandwidth reproduces the ≥ 1.5× tail.
- **H3:** at joint condition (25 Gbps, CV ≥ 2, cache-affine) colocated ≥ 100% of P/D SLO-compliant goodput at equal GPU count, or P/D needs ≥ 1.25× colocated GPUs for parity; benign (NVLink, CV = 1) reversed: P/D ≥ 1.3× colocated goodput. Kill: at joint condition, P/D ≥ 100% colocated goodput at equal GPU with P99 TTFT ≤ 1.1× colocated.
- **SLO (revised):** TTFT P95 ≤ 4.0 s, TPOT P95 ≤ 0.15 s, attainment ≥ 95%; double-allocation rerun pre-registered as a first-class cell on "both fail".

---

# Experiment ID
E1

# Target Hypothesis
**H1** — KV transfer becomes material under cross-node/low-bandwidth placement with long prompts (P95 transfer fraction ≥ 30% / P50 ≥ 15% at 25 Gbps, 8k–13k prompts; ≤ 5% at NVLink; kill if P95 fraction < 10% or the 25 Gbps-vs-NVLink P95 TTFT gap < 10% of NVLink value).

**E1 is allowed to KILL H1.** It cannot confirm H1 (confirmation needs real measurement, E3/E4); its verdicts are: "H1 killed" (kill conditions hold across the plausible parameter range), "H1 survives — proceed to E3/E4", or "H1 undecidable from corpus constants — narrow band where it flips, must go to E3 first" (not expected; see Purpose).

# Purpose
Distinguish whether H1 is arithmetically possible at all given the corpus's own transfer measurements, before any GPU time is spent. The transfer-materiality claim is a cost-model claim: transfer bytes × per-byte realized cost vs available bandwidth, with all components already measured in the corpus for this exact model class. E1 computes the P95/P50 transfer fraction of E2E TTFT from architecture constants (KV bytes/token) and corpus-measured transfer/prefill/queue times, sweeps the uncertain parameters, and checks whether the H1 threshold or the H1 kill condition is reached in the SLO-feasible regime. It eliminates the alternative explanation "the corpus numbers are idiosyncratic / not representative of the stated IV1(c) cell" by testing robustness of the verdict across the whole plausible calibration band. NOTE (review fix R4): E1 is an *arithmetic-support* gate — its kill lines are reachable, but a survival here means only "the mechanism is plausible; hardware confirmation happens in E3/E4".

# System
**Trace analysis + analytic cost model** (a Python script; no GPU, no serving framework, no simulator). The cheapest sufficient system because H1 is a closed-form materiality question:
- T_transfer = KV_bytes(prompt, model) / realized_bandwidth, with realized bandwidth anchored to FlowKV Table 3 (same model class, Llama-3.1-8B): cross-machine vLLM-Disagg 8,000/100 = 1.346 s, 10,000/100 = 1.737 s; single-machine 1.338 s; Mooncake RDMA 8,000/100 = 2.029 s (fails at 10k). These are the unhidden-path numbers the hypothesis targets.
- KV_bytes from closed-form architecture constants: Llama-3.1-8B-Instruct FP16 GQA = 2 (K+V) × 32 layers × 8 kv-heads × 128 head-dim × 2 B = 131,072 B = 128 KiB per token → 1.0 GiB at 8,192 tokens, 1.59 GiB at 13,000, 128 MiB at 1,024. Cross-check vs corpus: DistServe OPT-66B MHA = 1.13 GB per 512-token request (2.2 MB/token); the 8B-GQA/66B-MHA factor is (8/66)×(8/96 kv-heads)×2 ≈ 0.020 → predicted ≈ 45 KiB/token vs 128 KiB actual — consistent within factor ~3 (the residual gap is expected: FP16 vs the corpus's accounting and per-layer metadata; the sweep's bandwidth factor covers it). Arithmetic corrected per review fix R10.
- T_prefill from compute model (2·N·tokens FLOPs; A100 312 TFLOPS, 50% MFU): 2·8e9·8192 = 1.31e14 FLOPs / 1.56e14 = **0.84 s at 8k**, 1.33 s at 13k, 0.10 s at 1k. Calibration corrected per review fix R4: the previous 0.103 ms/token anchor came from Splitwise's *8-GPU (TP8)* table scaled as if single-GPU; the FLOPs formula is authoritative here. Uncertainty band: T_prefill(8k) ∈ [0.5, 0.9] s, (13k) ∈ [0.8, 1.5] s, (1k) ≈ 0.1 s (sweep covers MFU 30–60%).
- T_queue from DistServe's M/D/1 (Avg_TTFT = D + R·D² / (2(1−R·D))) with D = T_prefill, utilization ρ ∈ {0.3, 0.5, 0.6, 0.7}, P95 queue ≈ 2.5–3× mean wait (M/D/1 tail).
- Overlap: 0% by definition — H1 measures transfer on the TTFT critical path, not wall-clock overlap; the overlap-hidden case (Splitwise < 7% prompt time, ~8 ms A100 layer-wise) is modeled only for the NVLink cell (its <7% / 8 ms constants bound the ≤ 5% leg).
- Sync/fragmentation overhead folded into realized bandwidth via the FlowKV anchors (23,469→1 NCCL-call phenomenon; Beluga: ~75% of a 10.55 µs 16-KB transfer is synchronization, sglist 30 vs 128 chunks).

# Workload
The fixed 5,000-request trace (control variables): 70% short-class (median ~1,024 tokens, Splitwise conversation class, median 1,020) + 30% long-class (8k–13k, lognormal mean ~9k; anchored to FlowKV's 13k example, Mooncake avg 7,590, Sarathi arxiv median 7,059); output length fixed at 200 tokens (Mooncake output class, ratio ≈ 41.7); arrival CV ∈ {1, 2, 3} at a fixed mean rate; 60% of long-class requests share a common hot prefix (Mooncake: hot blocks accessed tens of thousands of times; real reuse ≈ 50%). E1 reads the trace's length distribution and arrival times only — it never replays it.

# Hardware
None. CPU-only script, minutes of compute. Why not less: cannot be less — this is the no-GPU tier by construction.

# Instrumentation Required
The trace-derived inputs: per-class input-length statistics (long-class mean/P95), output lengths, inter-arrival-time CV, hot-prefix share. The model's computed state per cell: T_transfer, T_prefill, T_queue (mean and P95), E2E TTFT P50/P95, transfer fraction P50/P95, and the 25 Gbps-vs-NVLink P95 TTFT gap — recorded for every cell of the uncertainty sweep. The verdict flag (survive/kill) per H1 branch.

# Baseline
NVLink-class placement (IV1 level a) within the same model and same load: T_transfer(NVLink) = 1.0 GiB / 600 GB/s = 1.7 ms nominal, bounded above by Splitwise's non-overlapped 8 ms layer-wise constant. Fair by construction: identical KV bytes, identical prompt classes, identical queue/prefill model; only the placement leg differs, which is exactly the kill-branch-2 comparison the hypothesis demands (P95 TTFT gap 25 Gbps vs NVLink, normalized by the NVLink value).

# Experimental Conditions
Uncertainty sweep over the four parameters that are not corpus-pinned (all others fixed at corpus values), 4×3×3×2 ≈ 72 model cells:
- A. Realized bandwidth factor f ∈ {0.2, 0.5, 1.0} of 25 Gbps nominal (corpus anchor f ≈ 0.24 from FlowKV 1.0 GiB/1.346 s).
- B. Utilization ρ ∈ {0.3, 0.5, 0.7}.
- C. T_prefill scale ∈ {0.5×, 1×, 2×} of the compute-model estimate (absorbs MFU/chunking uncertainty).
- D. Prompt class: long-class only (8k–13k, per H1) vs full mixed trace (fraction over all requests, sanity only).
A cell pair (25 Gbps / NVLink) is evaluated at every combination.

# Success Criterion
Exactly the hypothesis thresholds, evaluated as: the model supports H1 if at the corpus-anchored point and across the sweep the 25 Gbps 8k–13k cell shows P95 transfer fraction ≥ 30% of E2E TTFT (and P50 ≥ 15%), while the NVLink cell keeps P95 fraction ≤ 5% at matched load. Note: success here is provisional (model-supported, measurement-pending — the claim is confirmed only at E3/E4); E1's formal output is the robustness region.

# Falsification Criterion
Exactly the hypothesis kill condition (revised R2 — reachable lines), evaluated over the sweep: if at 25 Gbps with 8k–13k prompts and matched load the computed P95 transfer fraction < 30% of E2E TTFT for all plausible (f, ρ, prefill-scale) points, OR the computed P95 TTFT gap between 25 Gbps and NVLink placements is < 25% of the NVLink value — then H1 is dead and the pipeline stops (E3/E4 never run). Operational note: with the corrected prefill calibration (0.84 s at 8k) and the corpus anchor f ≈ 0.24 (1.35 s transfer), the fraction ≈ 55% — support is reachable at the anchored point; a kill requires realized bandwidth ≥ ~16 GB/s (transfer ≤ ~0.36 s for 8k), physically possible on a 25 Gbps link only with near-perfect fragmentation-free transfers, so a kill verdict is meaningful and decisive. If the sweep's verdict flips across the band, the verdict is "narrow band → E3 required" (pre-registered).

# Estimated Complexity
Very Low

# Expected Runtime
minutes–hours (script + trace parsing + 72-cell sweep)

---

# Experiment ID
E2

# Target Hypothesis
**H2** — A burst turns the cache-affine P/D pairing into a queue hotspot (CV 2–3 + cache-affine longest-prefix pairing → queue time ≥ 20% of E2E TTFT at P99 and P99 TTFT ≥ 1.5× colocated; CV = 1 keeps P99 ≤ 1.1× colocated; kill if at CV = 3 cache-affine P99 ≤ 1.1× colocated with P99 queue fraction < 15%, or if load-balanced routing at CV = 3 also reproduces the ≥ 1.5× tail).

**E2 is allowed to KILL H2.** It cannot confirm H2 (confirmation happens at E4); its verdicts are "H2 killed", "H2 survives — proceed to E4", or "H2 undecidable — proceed to E4 with a flagged risk".

# Purpose
Distinguish whether the queue-hotspot mechanism is a *pairing* artifact (cache-affine concentration of arrivals on one P/D pair, which a placement/policy decision controls) or a *burst-load* artifact (any routing would produce the ≥ 1.5× tail at CV = 3). This is a pure stochastic-process question: given calibrated service-time distributions and a routing policy, does arrival concentration on one pair push that pair's M/D/1 utilization toward 1, making its queue diverge exactly as DistServe's queue term predicts? The load-balanced condition is the discriminator that decides which of the two kill branches applies. E2 also removes the alternative "the hotspot is an artifact of my choice of service-time distribution" by sweeping that distribution over the corpus-anchored range. CRITICAL SCOPE (review fix R3): all H2-verdict cells run with transfer set to the NVLink value (mean 8 ms, Splitwise's non-overlapped constant) — H2 tests the queueing mechanism in isolation; the 25 Gbps transfer is H1's/H3's mechanism and does not enter H2 verdicts (a load-balanced discriminator at 25 Gbps would fire the kill branch spuriously because *every* request pays transfer).

# System
**Simulator** — a custom event-driven, discrete-time queueing simulator (single CPU machine, no GPU). Modeled:
- Arrival generator: Poisson (CV = 1) and ON/OFF two-state Markov-modulated arrivals at fixed mean rate (CV = 2 and CV = 3, targets matching IV2).
- P/D side: N_P prefill queues + N_D decode queues arranged in pairs; a conductor routing request → (p,d) pair per policy (cache-affine longest-prefix: choose the pair whose prefill instance holds the longest prefix match / shortest predicted TTFT — Mooncake Alg.1's T_queue + T_prefill (+ T_transfer) scoring; vs load-balanced: least-utilized instance). Prefill service times drawn per-request from a distribution calibrated to E1's corrected T_prefill model (0.5–0.9 s by prompt class, see E1).
- Transfer time: drawn from the NVLink distribution (mean 8 ms — Splitwise's non-overlapped constant) for ALL H2-verdict cells (conditions A–E below). The 25 Gbps transfer distribution (8k: lognormal mean 1.35 s at realized f ≈ 0.24) is used ONLY in the E4 joint cells for H3, never in E2's verdict cells.
- Colocated baseline side: two independent 1-GPU Sarathi-style chunked-prefill queues, token budget tau = 1024, service time = chunked-prefill time including ~25% chunking overhead at tau = 512–1024 (Sarathi Fig. 14), same arrivals.
- TTFT = T_queue + T_prefill + T_transfer + transfer_wait (wait for the paired decode instance's transfer engine / decode-side admission — Mooncake's "transfer time depends on congestion, not just size").
- NOT modeled: GPU kernel execution, NCCL/SM contention, actual memory fragmentation, real network — acceptable for a kill test because H2 is a claim about queue formation under a pairing policy, fully determined by the arrival process and the service/transfer distributions given the policy; this is exactly the fidelity DistServe uses for placement search (its event-driven simulator is < 2% error vs the real system, Table 2), and the H2 mechanism (Mooncake's anti-phase fluctuation, 2P+2D-vs-3P+1D reversal under a static partition) is a scheduling-phenomenon claim of the same class.

# Workload
The same fixed 5,000-request trace as E1 (identical order, fixed seed): 70% short-class (median ~1k) + 30% long-class (8k–13k), output 200, 60% of long-class share one hot prefix (this is what makes cache-affine concentration non-trivial), arrival CV ∈ {1, 2, 3} at a fixed mean rate chosen (pre-registered from a 3-point probe) so that the colocated baseline attains ≥ 95% SLO at TTFT P95 ≤ 4.0 s (revised SLO) — the matched-SLO operating point. Per cell: 5,000 requests, mean rate held fixed. All cells run with NVLink transfer (mean 8 ms).

# Hardware
One CPU-only machine (8+ cores for the sweep). None less is possible only in the sense that a laptop suffices; none more is needed — no GPU, no serving system.

# Instrumentation Required
Per-request state log: arrival time, routing decision (pair id), T_queue (prefill queue wait), T_prefill (sampled service), transfer wait (queueing for transfer engine/decode admission), T_transfer, TTFT total, prompt class, prefix-match length. Per-instance time series (per scheduling quantum): queue length, utilization, pair arrival share (concentration metric: max-pair share of arrivals), hotspot flag (ρ > 0.9 sustained over > 5 s). Aggregates per cell: P50/P95/P99 TTFT, P99 queue fraction = (T_queue + transfer wait)/TTFT, ratio of P99 TTFT to colocated P99, mean throughput, hot-pair arrival concentration, load-balanced-cell P99.

# Baseline
Colocated chunked prefill (2 instances, 1 GPU each) under identical arrivals and mean rate — fair because H2's ratios (≥ 1.5×, ≤ 1.1×, < 15%) are defined against colocated at matched load, matched request set, and matched GPU count (2 P/D GPUs vs 2 colocated GPUs).

# Experimental Conditions
Differ ONLY in IV2 (burstiness) and IV4 (routing policy); fixed: trace, mean rate, model service distributions, NVLink transfer distribution (H2 scope, review fix R3):
- A: CV = 1, cache-affine pairing (H2's steady-state leg — expects P99 ≤ 1.1× colocated).
- B: CV = 3, cache-affine pairing (H2's burst leg — expects ≥ 1.5× and ≥ 20% queue fraction).
- C: CV = 3, load-balanced pairing (the discriminator: if ≥ 1.5× persists here, the hotspot is a burst-load artifact → kill branch 2).
- D: CV = 2, cache-affine (intermediate threshold check).
- E (sanity, same cost class): CV = 3, cache-affine, N_P doubled to 2 (hotspot should dissolve if it is pairing-concentration-driven; a negative result here flags the mechanism as load-total, not concentration).
Each condition additionally swept over service-time scale ∈ {0.5×, 1×, 2×} and transfer mean ∈ {0.7×, 1×, 1.4×} of the NVLink anchor (12 runs per condition, 60 runs total, minutes each).

# Success Criterion
Exactly the H2 thresholds: in condition B, P99 queue fraction (T_queue + transfer wait) of E2E TTFT ≥ 20% at P99 AND P99 TTFT ≥ 1.5× the colocated baseline; in condition A, P99 TTFT ≤ 1.1× colocated (affinity pays in steady state); in condition C, P99 TTFT ≤ 1.2× colocated (isolating the pairing mechanism — the analogue of Mooncake's 2P+2D-vs-3P+1D reversal under controlled burstiness). Robust across the swept service/transfer scales.

# Falsification Criterion
Exactly the H2 kill conditions: if in condition B the cache-affine P/D shows P99 TTFT ≤ 1.1× colocated WITH P99 queue fraction < 15% of E2E TTFT — OR if condition C (load-balanced routing at CV = 3, same NVLink bandwidth) reproduces the ≥ 1.5× tail — then H2 is dead and the pipeline stops before any GPU serving. Note the logic: the first branch means no pairing concentration forms a hotspot; the second branch means no placement decision could avoid the hotspot.

# Estimated Complexity
Low

# Expected Runtime
hours (single day at most; 60 runs × 5,000 events on a single CPU)

---

# Experiment ID
E3

# Target Hypothesis
**H1** (hardware-confirmed leg). E1 established the transfer-materiality region from corpus constants; E3 replaces the two corpus-extrapolated parameters (realized 25 Gbps bandwidth incl. fragmentation/sync overhead; NVLink leg cost) with measurements on the actual E4 testbed, and recomputes the H1 fraction against E1's queue+prefill model.

**E3 is allowed to KILL H1** — this is the cheapest point where H1 can be killed on real hardware, since the kill requires no serving system: only the transfer path and the E1 latency model. When E3 survives, its measured constants are E4's step-0 calibration inputs (same testbed; noted per review fix R7).

# Purpose
Eliminate the residual alternative explanations in E1: (a) "the FlowKV/Mooncake transfer numbers are not reproducible on our testbed" (e.g., newer vLLM/kernels, different fragmentation, different NIC), and (b) "the NVLink leg is actually slow enough that the 25 Gbps-vs-NVLink gap is < 10% of NVLink value" (kill branch 2). E3 measures per-call overhead (Beluga: ~75% of a 10.55 µs transfer is sync), fragmented vs contiguous transfer cost (FlowKV: 23,469→1 NCCL calls; Beluga: sglist 30 vs 128 chunks), realized sustained bandwidth, and control-plane CPU cost — and recomputes the H1 P95 fraction and NVLink gap.

# System
**Microbenchmark** (custom script driving NCCL/RDMA-class transfers; no serving, no scheduler). This is tier 4 (small microbenchmark, single- to two-GPU) and the cheapest sufficient system because H1's object of study is the bare transfer path, not the serving system; the serving-specific terms (queue, prefill) are already modeled in E1 and calibrated to the same hardware in E4.

# Workload
Synthetic transfer payloads sized to the real KV layout of Llama-3.1-8B FP16: 128 MiB (1,024-token class), 1.0 GiB (8,192-token class), 1.59 GiB (13,000-token class); two layouts per size: (i) contiguous buffer (best case), (ii) paged-block layout with block ids in the corpus's fragmented order — 512 blocks of 2 MiB at 8k — forcing the FlowKV-style (L,2,B,H) multi-call pattern. Per-call overhead probe: 16 KB transfers in a ping-pong loop (Beluga protocol: 10.55 µs total / 2.68 µs data). Contention: 1, 2, 4, 8 concurrent transfers. Background traffic probe: netperf-class measurement of the link with and without background load (< 2% requirement per control variables).

# Hardware
3× A100-SXM4-80GB: 2 on node A (NVLink 600 GB/s pair) + 1 on node B, dedicated 25 Gbps cross-node link (the E4 testbed; if the lab link is faster, throttle with tc and verify realized ≈ 25 Gbps with a netperf probe before each run). Requires the NVLink pair (for the IV1(a) leg) and the cross-node pair (for IV1(c)); not less because both H1 legs must be measured on the same hardware for the gap ratio to be fair. No serving GPUs, no additional nodes.

# Instrumentation Required
Per run: payload bytes, block count, NCCL/RDMA call count (FlowKV's 23,469-class metric), call-size histogram, total wall time, per-call sync vs data time (sync overhead fraction), realized bandwidth (GiB/s), CPU utilization of the control-plane process (Beluga's CPU-driven-path check; < 60% required per confounders), concurrency level, background traffic level.

# Baseline
The NVLink leg (IV1 level a: intra-node pair on node A) measured with identical payloads, layout, and concurrency as the 25 Gbps leg — the fair baseline for kill branch 2 (P95 TTFT gap normalized by NVLink value), because only the placement leg differs.

# Experimental Conditions
- A: 25 Gbps cross-node, contiguous layout, 1 concurrent transfer (best-case realized bandwidth).
- B: 25 Gbps cross-node, fragmented paged layout, 1 concurrent transfer (the hypothesis's unhidden-path case; corpus predicts ~0.75 GiB/s effective).
- C: 25 Gbps cross-node, fragmented, 4 concurrent transfers (congestion/contention case — Mooncake's "transfer time depends on congestion").
- D: NVLink intra-node, fragmented layout, 1 and 4 concurrent (the baseline leg).
- E: 16 KB ping-pong overhead probe on both legs (Beluga protocol class).
- F (exploratory, review fix R6): 100 Gbps cross-node, fragmented layout, 1 and 4 concurrent — records the crossover direction for bandwidth; NOT gated on any hypothesis threshold (H1 thresholds apply only at 25 Gbps).
Each condition: 20 repetitions at each payload size, medians and P95 reported.

# Success Criterion
Exactly the H1 thresholds, recomputed: measured 8k/13k transfer times on the 25 Gbps leg, plugged into E1's TTFT model at the corpus-anchored queue/prefill terms, yield P95 transfer fraction ≥ 30% of E2E TTFT (P50 ≥ 15%) for the ≥ 8k class, and the NVLink leg yields ≤ 5% P95 fraction — with measured transfer times landing in the corpus class (0.5–2.1 s at 8k; cf. FlowKV vLLM-Disagg 1.346 s, Mooncake 2.029 s). Measured per-call sync overhead ≥ 50% of per-call time (Beluga: ~75%) corroborates the fragmentation mechanism.

# Falsification Criterion
Exactly the H1 kill conditions (revised R2 — reachable lines), recomputed with measured constants: if at 25 Gbps with 8k–13k payloads the measured P95 transfer fraction of E2E TTFT stays < 30% (computed with E1's corrected prefill model), OR the P95 TTFT gap between 25 Gbps and NVLink placements (recomputed via E1's model with measured legs) is < 25% of the NVLink value — H1 is dead and the pipeline stops. With the corrected prefill calibration, a measured fraction < 30% requires realized transfer ≥ ~16 GB/s on the 25 Gbps link — implausible with fragmented paged transfers but physically possible, so the kill is meaningful. Measured transfer times landing in the corpus class (0.5–2.1 s at 8k) with sync overhead ≥ 50% per-call corroborate the mechanism; measured times < 30% of the corpus anchors but above the kill line are recorded as "H1 narrowed" and E4 proceeds with the measured constants.

# Estimated Complexity
Low–Medium (driver script + tc throttling + netperf verification; no serving bring-up)

# Expected Runtime
hours

---

# Experiment ID
E4

# Target Hypothesis
**H1** (confirmatory, on real TTFT), **H2** (confirmatory, on real system), **H3** (first test of the joint consequence).
- H3 thresholds: at the joint condition (25 Gbps, CV ≥ 2, cache-affine) colocated attains ≥ 100% of P/D's SLO-compliant goodput at equal GPU count, OR P/D needs ≥ 1.25× the colocated GPU allocation for parity; at the benign condition (NVLink, CV = 1) the ordering is reversed with P/D ≥ 1.3× colocated goodput. Kill: at the joint condition P/D sustains ≥ 100% of colocated goodput at equal GPU allocation with P99 TTFT ≤ 1.1× colocated.

**E4 is allowed to KILL H1, H2, and H3.** A kill of any of them here stops the pipeline (E5 runs only if H3 survived E4).

# Purpose
Confirm the two mechanisms end-to-end and test the joint overtake on a real system: (i) real transfer fraction inside the TTFT critical path at 25 Gbps vs NVLink (H1), (ii) real queue-hotspot formation under CV = 3 with cache-affine routing, discriminated by the load-balanced cell (H2), (iii) the colocated-goodput crossover at equal GPU allocation (H3). This is the first experiment where the E1/E2 predicted outcomes are tested against a real serving system; it also records the primary hypothesis's throughput-equality guard (aggregate throughput at equal load within 5% of colocated — reported, not a conjunct, per revision R1).

# System
**vLLM** (vLLM-lineage engine, one codebase for both configurations — control variables: identical kernels and scheduler code; P/D differs from colocated only in phase split and placement). P/D mode: vLLM's disaggregated prefill/decode deployment (DistServe-style 1P+1D orchestration, FCFS, prefix cache on — Mooncake-class reuse) with KV moved over the link via NCCL on the UNOPTIMIZED path (vanilla vLLM-Disagg transfer class, i.e., no FlowKV-style shape remap or MSCCL++ layer-wise machinery — the hypothesis targets the unhidden case; the optimized path would only be a later, explicitly-labeled condition). Colocated mode: Sarathi-style chunked prefill (tau = 1024, token-budgeted hybrid batches) in the same engine. Routing policies (cache-affine longest-prefix vs load-balanced) are a small scheduler hook — the ONLY modification, implemented as an experimental condition, not a system proposal. This is tier 5–6 (existing benchmark harness + minimal modification), the cheapest sufficient serving test: no new system is built; both configurations are documented existing ones, and no GPU budget is spent before E1/E2 survived.

# Workload
The fixed 5,000-request trace (identical order, fixed seed, per-cell KS-verified length distributions, mean within 1%): 70% short (median ~1k) + 30% long (8k–13k), output 200 tokens, 60% of long-class share a hot prefix (enabling cache-affine hits), arrival CV ∈ {1, 3}, mean rate fixed at the E2-determined matched-SLO operating point (both configurations attain ≥ 95% at TTFT P95 ≤ 4.0 s — revised SLO; the rate is P/D-feasible so "both fail" remains diagnostic, review fix R4). Both cache-populate and cache-hit (pre-populated) scenarios run per cell (Beluga's two scenarios; conclusions drawn only within a matched scenario). 200-request warmup per cell before the measurement window; first 10% of window discarded. Colocated cells dispatch via FCFS + join-shortest-queue (pre-registered control). Node-identity balance: the 25 Gbps cells are rerun with D on node A (review fix R5).

# Hardware
The E3 testbed: node A with 2× A100-SXM4-80GB (NVLink 600 GB/s) + node B with 1× A100-80GB, dedicated 25 Gbps cross-node link (realized bandwidth verified with a netperf probe per cell, background < 2%). Every cell uses exactly 2 GPUs: P/D cells = 1P+1D (P and D co-located on node A via NVLink for the IV1(a) cells with DistServe Alg2 constraint ON; P on node A / D on node B, Alg2 OFF, for IV1(c) cells); colocated cells = the same 2 GPUs as independent 1-GPU chunked-prefill instances behind an FCFS dispatcher. Not less: H1's NVLink-vs-25 Gbps contrast and H3's equal-GPU-allocation comparison both require this shape; a single-GPU setup cannot host a P/D split at all.

# Instrumentation Required
Per request: id, arrival/prefill-start/prefill-end/transfer-start/transfer-end/decode-start/first-token timestamps (transfer and queue time instrumented INSIDE the TTFT interval, on the critical path, not wall-clock overlap), TTFT, TPOT (P50/P95/P99), block ids and per-request transfer bytes, NCCL/RDMA call count per request (fragmentation sanity metric), cache hit rate (prefix match ratio — guard metric), routing decision (pair id). Per cell: SLO goodput (max sustained RPS at attainment ≥ 95%), P50/P95/P99 TTFT and TPOT, transfer fraction P50/P95 (≥ 8k subpopulation), queue fraction P95/P99, GPU allocation (GPUs per config), HBM utilization and KV capacity (weight-duplication guard: > 10-point utilization difference must be attributed via the allocation DV, not the transfer mechanism), scheduler CPU utilization (< 60% required), realized link bandwidth per cell (netperf), power draw (clocks locked via nvidia-smi -lgc).

# Baseline
Colocated chunked prefill (2× 1-GPU instances, tau = 1024) on the same 2 GPUs, same trace, same SLO tuple, same mean rate — fair by construction (matched GPU allocation, matched load, matched request set, identical engine lineage). Hit-rate guard: if P/D's hit rate exceeds colocated's by > 10 points, enable the equivalent prefix cache in the colocated baseline and rerun the affected cells (confounder rule).

# Experimental Conditions
Differ ONLY in IV1 (placement), IV2 (CV), IV4 (routing); everything else fixed:
- A: P/D NVLink co-located (Alg2 ON), CV = 1, cache-affine — H1's ≤ 5% fraction cell and H3's benign cell.
- B: P/D 25 Gbps cross-node, CV = 1, cache-affine — isolates bandwidth from burst (H1 leg).
- B2 (exploratory, review fix R6): P/D 100 Gbps cross-node, CV = 1, cache-affine — records the crossover direction; not gated on hypothesis thresholds.
- C: P/D 25 Gbps cross-node, CV = 3, cache-affine — the joint condition (H2 + H3 leg).
- D: colocated chunked prefill, CV = 1 and CV = 3 — the baseline at both burstiness levels (same 2 GPUs).
- E: P/D 25 Gbps, CV = 3, load-balanced — H2's discriminator (already tested at NVLink in E2; here at 25 Gbps it is a reported cell, NOT an H2 verdict cell, because transfer confounds the queueing claim — review fix R3).
- F (conditional, pre-registered): if in C the transfer fraction ≥ 30% but the P99 TTFT gap < 1.1× (overlap hiding the cost — Ambiguous Outcome 2), run C' with SERIALIZED transfer (Splitwise's serialized path: +64% second-token class) to expose the hidden cost; H1's consequence is void until this check is done.

# Success Criterion
Exactly the hypothesis thresholds: **H1** — P95 transfer fraction ≥ 30% of E2E TTFT (≥ 15% at P50) for ≥ 8k prompts at 25 Gbps (B/C), ≤ 5% P95 at NVLink (A). **H2** — in C, P99 queue fraction (T_queue + transfer wait) ≥ 20% of E2E TTFT and P99 TTFT ≥ 1.5× colocated (D); in A's CV = 1 pair (and B), P99 ≤ 1.1× colocated; in E, P99 ≤ 1.2× colocated (reported cell). **H3** — in the joint condition (C vs D at CV = 3), colocated attains ≥ 100% of P/D's SLO-compliant goodput at equal GPU count; at the benign condition (A vs D at CV = 1), P/D attains ≥ 1.3× colocated goodput. Primary-hypothesis consistency guard (revised R1): aggregate throughput at equal load within 5% of colocated in C vs D — reported as a guard; if the gap exceeds 5%, it is recorded with the result and does not by itself void the P99 claim.

# Falsification Criterion
Exactly the kill conditions, checked in order: **H1** — in B/C, P95 transfer fraction < 30% of E2E TTFT, or the P95 TTFT gap between 25 Gbps (B) and NVLink (A) placements < 25% of the NVLink value (revised R2). **H2** — in C, cache-affine P/D shows P99 TTFT ≤ 1.1× colocated WITH P99 queue fraction < 15% of E2E TTFT; or E (load-balanced at CV = 3) reproduces the ≥ 1.5× tail at the NVLink cell from E2 (burst-load artifact — no placement decision could avoid it). **H3** — in the joint condition, P/D sustains ≥ 100% of colocated SLO goodput at equal GPU allocation with P99 TTFT ≤ 1.1× colocated. Any single kill stops the pipeline. Outcome bands pre-registered (review fix R9): fraction ≥ 30% → H1 supported; < 30% → H1 falsified; gap ≥ 25% → supported; < 25% → falsified; "neither supported nor killed" is not an allowed outcome for H1/H2 at E4. If BOTH configurations violate the SLO at the joint condition, the double-allocation rerun (2× GPUs for both, pre-registered first-class cell) runs before any conclusion.

# Estimated Complexity
Medium

# Expected Runtime
multi-day (harness bring-up + ~12 serving runs of 5,000 requests × 2 scenarios × 2 burstiness levels, plus the conditional serialized cell)

---

# Experiment ID
E5

# Target Hypothesis
**H3** — the GPU-allocation leg: "or P/D needs ≥ 1.25× the colocated GPU allocation to reach parity." E5 quantifies the exact multiplier only if E4 supported H3 (colocated ≥ 100% of P/D goodput at equal GPUs). If E4 killed H3 (P/D ≥ 100% goodput at equal GPUs with P99 ≤ 1.1×), E5 never runs.

**E5 can KILL H3's allocation leg**: if P/D reaches ≥ 100% of colocated SLO-compliant goodput (with P99 TTFT ≤ 1.1× colocated) at a GPU allocation strictly below 1.25× of colocated's, then the "needs ≥ 1.25×" branch of H3 is false; if additionally it does so at equal allocation, H3 is dead entirely (already decided at E4 — E5 only resolves the residual branch).

# Purpose
Resolve the remaining disjunction in H3: E4 establishes whether colocated ≥ P/D goodput at equal GPUs. E5 measures whether the deficit is eliminated by adding P/D GPUs, and at what multiplier — pinning the parity cost of disaggregation (2× weight duplication plus phase-split efficiency loss) in GPU-equivalent terms. This distinguishes "colocated wins only at equal allocation; P/D wins with modest extra resources" (H3 supported in its second branch) from "no allocation within 1.25× fixes P/D" (H3 supported in its first branch) from "P/D reaches parity below 1.25×" (allocation leg killed).

# System
**vLLM** — identical to E4, no new code. P/D cells at increasing allocation; colocated stays at the E4 configuration. The only variable added is the GPU-allocation dependent variable.

# Workload
The same fixed 5,000-request trace at the joint condition (CV = 3, cache-affine, 25 Gbps, mixed prompts), same SLO tuple (TTFT P95 ≤ 4.0 s, TPOT P95 ≤ 0.15 s, attainment ≥ 95% — revised), same mean rate, both cache scenarios, same warmup/discard and per-cell KS/hit-rate/bandwidth guards as E4.

# Hardware
The E4 testbed (3× A100: 2+1) for the equal-allocation (2-GPU) and 1.5× (3-GPU) cells. If the sweep needs 4 GPUs (2P+2D), add one more A100 on node B (testbed → 4× A100, 2+2). Not less: the 1.25× boundary for a 2-GPU colocated baseline falls at 2.5 GPUs, so the informative allocations are 2 (1×), 3 (1.5×), and optionally 4 (2×) GPUs.

# Instrumentation Required
Same per-request/per-cell instrumentation as E4, plus per allocation level: P/D SLO-compliant goodput, colocated goodput (unchanged reference), P99 TTFT ratio P/D-vs-colocated at each allocation, GPU count per configuration, HBM utilization per GPU.

# Baseline
Colocated chunked prefill at 2 GPUs (E4's D cell, replayed under the same conditions) — the fixed resource reference; P/D goodput is measured at 2 / 3 / (4) GPUs against this single reference.

# Experimental Conditions
A single-variable sweep: P/D allocation ∈ {1P+1D (2 GPUs, 1×), 2P+1D (3 GPUs, 1.5×), 2P+2D (4 GPUs, 2×) — the last only if parity is not yet reached}, all at the joint condition (25 Gbps, CV = 3, cache-affine), everything else identical to E4's C cell. Allocation order fixed; per-cell verification that the workload statistics are unchanged.

# Success Criterion
Exactly the H3 thresholds: at the joint condition, colocated attains ≥ 100% of P/D's SLO-compliant goodput at equal GPU count (already established in E4's C vs D cell), OR P/D requires ≥ 1.25× the colocated GPU allocation (i.e., goodput stays < 100% of colocated at 2 GPUs and only reaches ≥ 100% at ≥ 2.5 GPUs → demonstrated at 3 GPUs, 1.5×) to reach parity. Benign-cell leg (A cell, E4): P/D ≥ 1.3× colocated goodput at NVLink/CV = 1 — required for the full H3 statement; if the benign reversal fails while the joint condition holds, report H3 as supported in its failure-condition branch only.

# Falsification Criterion
Exactly the H3 kill condition at each allocation level: if P/D sustains ≥ 100% of colocated SLO goodput at equal GPU allocation (2 GPUs) with P99 TTFT ≤ 1.1× colocated — already the E4 kill — H3 is dead. For the allocation leg specifically: if P/D reaches ≥ 100% of colocated goodput (P99 ≤ 1.1×) at an allocation below 1.25× (e.g., demonstrably at 2 GPUs = 1×, or interpolated parity below 2.5), the "needs ≥ 1.25×" branch is killed and H3 is dead as stated even if colocated also reaches 100% at equal allocation, because then no resource-equivalence deficit exists.

# Estimated Complexity
Medium

# Expected Runtime
multi-day (2–3 additional serving runs plus the E4 C/D replay reference; bounded by the E4 harness bring-up already paid)