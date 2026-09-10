# experiment_plan.md — RQ-4 Stage 3A Minimum-Cost Kill-Sequence (revised post-review)

> Revision 2026-08-27: restructured after `design_review.md` — the original hypothesis was mathematically unsatisfiable under its own model. Changes: E0 gate-0 algebra step added; per-policy load anchoring; harmed class restated as the hot class; skew re-parameterized as p_hot (Zipf decoration dropped); L_prefix 1792→1024; hit-gain threshold replaced by the E0-derived value G; common absolute SLO; "least-loaded" = token-weighted backlog; suffix-sensitivity control; E4 floor scoping (H3 requires ≥3 nodes, "void" verdict on the floor).

## 0. Scope, Cost Ordering, and Kill-Gate Chain

**Scope (locked by `hypothesis.md`).** Stage 3A establishes only whether the affinity-versus-tail/fairness problem exists as a *measurable phenomenon* under a matched harness. No scheduler, admission policy, or fairness controller is designed. The only experimental conditions are three **existing** routing behaviors, held identical in every other component:

- **A — Affinity-only:** greedy longest-prefix routing to the node holding the longest matching prefix, tiebreak by token-weighted backlog (the behavior SGLang's RadixAttention implements and whose starvation it documents; `SGLang.md` §3/§8, Theorem 3.1).
- **B — Load-only:** route to the node with minimum token-weighted backlog, no prefix preference (Mooncake's `load-balancing` scheduling baseline; `Mooncake.md` §6/Fig 8).
- **C — Simple threshold replication:** when the hot-prefix request rate exceeds a fixed popularity threshold, replicate the hot prefix to a fixed number of nodes (k = 2); otherwise affinity-only (Mooncake's `kvcache_balancing_threshold` hot-spot replication behavior; `Mooncake.md` §3/§6).

The deliverable is a binary, pre-registered verdict on H1–H3; any "partial" outcome is reported as non-discriminating (`hypothesis.md`, Ambiguous Outcome).

**Cost ordering (strictly preferred top-down).** The sequence is built on the insight that RQ-4's phenomenon is **mostly queueing**: GPU prefill/decode costs enter only as *service-time distributions*, so a closed-form analysis + calibrated event-driven model kills H1 and H2 with no GPU serving. Only H3's controllability claim (real caching hit-rate/eviction trade-offs) and the cold-class scheduling-order question require a real system.

| Tier | Cost item | Used by |
|---|---|---|
| 0 | Closed-form queueing algebra (M/G/1 concentration analysis, CPU only) | E0 |
| 1 | Offline trace analysis (replay through queueing model, no serving) | E1 |
| 2 | Simulation (event-driven queueing + prefill-cost + cache model) | E2, E3 |
| 3 | Instrumentation of an existing system | E4 (warm-up step) |
| 4 | Small microbenchmark (single GPU) | E4 (step 0, length–service curve fit) |
| 5 | Existing benchmark harness | E4 (trace replay) |
| 6 | Minimal system modification (routing-decision wrapper) | E4 |
| 7 | Complex system modification | **avoided** |

**Kill gates (stop as soon as any condition is met):**

```
E0 (gate-0 algebra) ── produces every cell's predicted utilization, ratios, SLO crossings, hit-gain G
   │ (if G < 10pp is derived, the hit leg is killed pre-registrationally — H1 rests on the queue leg)
   ▼
E1 (H1 point test, p_hot ∈ {0.8, 0.9}) ──kill──▶ reclassify G5 as engineering composition; STOP
   │ survive
   ▼
E2 (H2 sweep, completes H1 grid falsification) ──kill──▶ STOP (H1 and/or H2 falsified)
   │ survive
   ▼
E3 (H3 in calibrated simulator) ──kill──▶ STOP (H3 falsified; no GPU spent)
   │ survive
   ▼
E4 (real-system verification: H1/H2 anchors on 2 nodes; H3 requires ≥3 nodes — "void" verdict on the floor)
```

E0–E3 run on CPU only. E4 runs only if E1, E2, and E3 all survive.

## 1. Shared Experimental Infrastructure

### 1.1 Declared metric definitions (copied exactly from `hypothesis.md`, revisions applied)

- **SLO** (revised R5): common absolute — per-class P99 TTFT ≤ **2.0 s**, identical for both classes (the per-class 5×-isolated definition was abandoned: its denominators were 8× apart and not comparable).
- **Fairness index FI** (revised R5): `FI(p) = min_k A_k(p) / max_k A_k(p) ∈ [0,1]`, where `A_k(p)` = per-class SLO attainment fraction on the COMMON SLO. A policy is **fair** iff `FI(p) ≥ 0.95`, **unfair** iff `FI(p) < 0.8`. **FI is a report-only metric** — the per-class verdicts ride on the direct P99 ratios in H1–H3.
- **Slowdown metric**: `slowdown_k(p) = P99_TTFT(k, p) / P99_TTFT(k, load-only at equal offered load)`; H3's ratio construction compares the hot-class ratio against the cold-class ratio.
- **Load anchoring (revised R4):** all loads are expressed as fractions of the AFFINITY configuration's own saturation (computed in E0). The overload region (hot-node utilization ≥ 1.0) is marked in every report and excluded from all verdicts.

### 1.2 Calibrated queueing simulator specification (E1–E3; E4 only validates it)

**What is modeled.**
- **Nodes:** N = 4 homogeneous nodes (matches the hypothesis's "4 GPUs" control; single-node placement ⇒ transfer cost ≈ 0). Each node = one FIFO prefill server plus per-node queue. A cluster-wide admission cap c (maximum in-flight = in-service + queued) is enforced identically across conditions. **The global admission FIFO is removed** (design_review issue 1): admission = dispatch to the chosen node immediately; the shared-admission construct was the artifact that manufactured overload-ratios identically across policies. Its absence makes the simulator's verdicts honest; its real-system variant (if any) is an E4 observation.
- **Prefill service time:** `T_prefill(tokens_not_cached) = γ · tokens_not_cached` (linear; justified by Sarathi's finding that linear ops dominate >80% of prefill time, `Sarathi-Serve.md` §3, Fig 4). Cache hit ⇒ cached prefix tokens skipped ⇒ service uses only the non-prefix remainder. γ default calibrated so a full 2048-token prefill ≈ 1.0 s; robustness band γ×{0.5, 2.0}; **shape check deferred to E4 step 0** (the linear-vs-superlinear assumption is load-bearing for the hot/cold service ratio — design_review issue 10; if the E4 curve fit rejects linearity, the simulator constants are re-fit before any verdict is accepted).
- **Cache model:** per-node LRU radix-style prefix cache, memory-accounted: total KV budget (bytes per node × 4) held EQUAL across policies A/B/C (design_review issue 4). A prefix is resident on a node only after a request with that prefix was served there; LRU eviction leaf-first. Cold-class requests carry unique prefixes ⇒ effectively always miss (Mooncake: >50% of blocks never reused, `Mooncake.md` §2/Fig 6). **Load-only accumulates the hot prefix on every node it serves hot traffic from; the hit-rate differential is a derived quantity, not an asserted anchor** (E0 derivation; the pre-revision 15pp anchor was contradicted by the plan's own cache model).
- **Routing decision function:** exactly the three policies of §0. A: route to the node whose radix cache holds the longest matching prefix; tie/empty → minimum token-weighted backlog (revised R6). B: route to minimum token-weighted backlog (no prefix preference). C: when hot-prefix request rate exceeds fixed threshold θ, replicate hot prefix to k = 2 nodes, then affinity-route within the replica set; θ held constant across the sweep.
- **Replication accounting (policy C):** replication bytes transferred and additional KV memory consumed are modeled and reported; the total-budget constraint (memory-accounted) applies to C as well.

**What it is calibrated against (corpus numbers).**
- Cache/hit envelope: SGLang benchmark hit rates 50–99% (96% of optimal) and production 52.4% (LLaVA-NeXT-34B) / 74.1% (Vicuna-33B) with 1.7× first-token reduction (`SGLang.md` §6, §13); Mooncake hit-ratio-vs-capacity curve 30% @1K → 50% @50K → 51% @Inf (LRU best) (`Mooncake.md` §2, Table 1, Fig 6). The budget levels are calibrated so the E0-derived gain G is realized inside the corpus-supported envelope.
- Queue/prefill split: Mooncake's `TTFT = T_queue + T_prefill + T_transfer` routing estimate (`Mooncake.md` §3/§6); FastServe's queueing at up to 90% of end-to-end latency under long-tail load (`FastServe.md` §1/§3).
- Service-time scale and SLO convention: Sarathi's 5× decode-iteration SLO convention and median prompts 1730 (openchat) / 7059 (arxiv) (`Sarathi-Serve.md` §6, Table 2); Mooncake mean input 7590, output 182 (`Mooncake.md` §2, §5); KVFlow's 1024-token fixed prompts and the 0.57× concurrency collapse at 64 concurrent workflows (`KVFlow.md` §10).
- Arrival model and trace scale: Llumnix's Poisson + Gamma (varying CV) 10,000-request traces (`Llumnix.md` §8); Mooncake's 23,608-entry trace (`Mooncake.md` §5); the hypothesis's fixed 10,000-request runs.

**What is NOT modeled and why it is acceptable for a kill test.**
- **Decode phase / TBT:** not a metric in H1–H3 (all TTFT-based); the tail that matters forms in the prefill queue (FastServe 90% queueing).
- **Within-node scheduling order (SGLang's longest-prefix-first starvation):** NOT representable by FIFO queues; it is the cold-class mechanism and is deferred to E4 as a reported observation (revision R1, critic #D). **Consequence (pre-registered):** E1–E3 pass verdicts are labeled "queue-concentration passes; the tail claim is untested until E4."
- **Continuous batching / Sarathi tau / batch composition:** batching changes mean service times but not the routing-induced concentration; batch composition is recorded in E4 as a confounder check.
- **GPU kernels, memory fragmentation, PagedAttention blocks:** constant across policies; only ratios and SLO-crossing (scale-invariant under the linear model) are used.
- **Network/KV transfer:** single-node design ⇒ T_transfer ≈ 0; replication traffic modeled and accounted for policy C.
- **Preemption/migration (Llumnix):** not part of the three existing routing policies; excluded by construction.

### 1.3 Synthetic workload generator (shared, revised R3)

- **Request count:** 10,000 per run (hypothesis control; Llumnix 10,000, Mooncake 23,608 scale).
- **Classes:** hot class (shared prefix L_prefix = 1024 tokens + unique suffix) and cold class (unique, disjoint prefixes). Skew = hot-prefix request fraction **p_hot ∈ {0.5, 0.7, 0.8, 0.9}** (the Zipf decoration is dropped — revision R3; the "1:1 by count" case is p_hot = 0.5).
- **Context length:** hot and cold prompts share the same per-request TOTAL length distribution, mean 2048 tokens, Uniform(1920, 2176); the shared prefix is 1024 tokens. **Suffix-sensitivity variants (revision R6):** hot suffix ∈ {256, 1024} (i.e., total 1280 / 2048) and a length-mismatched variant (hot mean 3072, cold unchanged).
- **Output length:** fixed max_tokens = 256 (hypothesis control; Mooncake output mean 182).
- **Arrival pattern:** Poisson at rate λ with load levels {0.5, 0.65, 0.8} × the AFFINITY configuration's own saturation (per-policy anchoring, revision R4; the overload region ≥ 1.0 hot-node utilization is excluded from verdicts); Gamma-burst arrivals with CV ∈ {1, 2, 4} at matched mean (identical inter-arrival sequences replayed under every policy).
- **Concurrency:** c ∈ {8, 16, 32, 64} plus c = 128 for H2's falsification range.
- **Reuse characteristics:** hot prefix shared by fraction p_hot of requests; cold prefixes never reused.
- **Warm-up / steady state:** fixed warm-up schedule (hot prefix filled by a predeclared number of hot-class requests); metrics on the steady-state window only.

### 1.4 Seed, replication, and statistics

- Fixed seeds; deterministic (greedy) generation semantics; 10 seeded replications per condition for P99 stability. P99 on the pooled per-request TTFT within the steady-state window. γ-robustness: E1/E2 verdicts re-checked at γ×{0.5, 2.0}; under the linear service model the *ratios* and the SLO crossing are γ-scale-invariant.

### 1.5 Confounder detection (recorded in all experiments where available)

Per-node queue-length traces and utilization (Confounder 1); batch-composition statistics (Confounder 2); per-node CPU utilization in E4 only (Confounder 3); both budget levels always reported (Confounder 4); identical inter-arrival sequences across policies (Confounder 5); suffix-sensitivity and length-mismatch variants (Confounder 6, revised R6); replication bytes / KV memory for policy C with total budget constant (Confounders 7–8); warm-up schedule fixed (Confounder 9).

---

## E0 — Gate-0 algebra: closed-form concentration analysis

# Experiment ID
E0

# Target Hypothesis
All three — this is a derivation step, not a verdict. Its outputs are the pre-registered cell parameters: per-cell hot-node utilization, predicted P99 ratios (M/G/1), predicted SLO-crossing points, and the hit-rate gain G.

# Purpose
Derive, in closed form, what the queueing model predicts before ANY simulation or serving: (i) hot-node utilization = (p_hot · suffix_hot)/(per-node total work) × per-node load-only utilization; (ii) hot-class P99 ratio via M/G/1 waiting-time ratios W(ρ_hot)/W(ρ_load-only); (iii) cold-class P99 ratio via mixed-traffic P-K formulas; (iv) the SLO-crossing points (p_hot*, c*) implied by the derived utilization schedule; (v) the memory-accounted hit-gain G from the cache model (load-only must replicate the hot prefix across the nodes it serves hot traffic from; equal total budget bounds that replication). This replaces the pre-revision "most favorable point" premise with a derived grid (design_review issues 1, 4, 7): E1's points are chosen where the derivation says the effect is present but the system is stable.

# System
Closed-form analysis (paper + script; no GPU, no simulator). Cheapest sufficient: the queue component is M/G/1 algebra; the cache component is a deterministic LRU budget argument.

# Workload
n/a — parameterized over p_hot ∈ {0.5, 0.7, 0.8, 0.9}, c ∈ {8, 16, 32, 64}, loads {0.5, 0.65, 0.8} × affinity saturation, γ ∈ {0.5, 1, 2}×.

# Hardware
None.

# Instrumentation Required
Output tables: per (p_hot, c, load, γ): hot-node utilization, predicted hot/cold P99 ratios, SLO crossing flags, predicted G per budget level; the stable-region boundary; the pre-registered E1/E2 grid points and the expected values attached to each.

# Baseline
M/G/1 reference formulas (standard closed-form results; the analysis is checked against the E2 simulator at 3 anchor points before E2 verdicts are accepted).

# Experimental Conditions
n/a (derivation over the parameter grid).

# Success Criterion
E0 succeeds iff it produces: (i) a stable-region map; (ii) predicted ratios per cell (used as E1/E2's expected values); (iii) the value G with its derivation; (iv) pre-registered E1/E2 grid points. **Pre-registered kill (revision R2): if the derivation yields G < 10pp at both budget levels, the hit-rate leg of H1 is killed pre-registrationally and H1 rests on the queue leg alone** (recorded as an E0 outcome, not a hypothesis change).

# Falsification Criterion
n/a (no hypothesis is decided at E0). If E0's derived predictions contradict the corpus anchors by >3× on any ratio, the calibration constants are re-checked before E1 runs.

# Estimated Complexity
Very Low

# Expected Runtime
hours (desk analysis + script)

---

## E1 — H1 point kill test: offline trace replay through a queueing model

# Experiment ID
E1

# Target Hypothesis
H1 (existence claim on the hot-node hotspot). Primary verdict gate for H1's points.

# Purpose
Cheapest possible existence check: at the E0-derived stable operating points (p_hot = 0.8 and p_hot = 0.9, c = 32, equal total offered load, equal memory-accounted cache budget, stable region), does affinity-only routing produce (a) a ≥G memory-accounted hit-rate gain with aggregate mean TTFT ≤ 0.8× and (b) a hot-class P99 TTFT ≥ 2× its load-only value while cold-class P99 stays ≤ 1.2×? The two points cover the predicted effect region; a null at both points kills the RQ at zero GPU cost (the "most favorable point" premise is removed — revision R7; the effect monotonically increases with p_hot per E0's concentration algebra, so two points bracketing the threshold are sufficient).

# System
**Trace analysis + queueing-model replay (cost tier 1; CPU only, no GPU, no serving engine).** Cheapest sufficient because H1 is a statement about *ratios of TTFT distributions under equal load*, and the differential is queueing (FastServe: queueing up to 90% of E2E). GPU costs enter only as the calibrated service-time distributions of §1.2. The evaluator replays the fixed 10,000-request trace through per-node FIFO queues (admission = dispatch, no global FIFO — revision R4): for each request, the routing decision function (A vs B) picks the node, hit/miss by the per-node cache model, service = γ × uncached tokens, completion by deterministic per-node recursion. Expected values per cell come from E0.

# Workload
Per §1.3: 10,000 requests; p_hot ∈ {0.8, 0.9}; matched total-length contexts (mean 2048, hot suffix 1024); fixed max_tokens = 256; Poisson arrivals at the E0-derived stable loads (0.65 and 0.8 of affinity saturation; the 0.5 level as a mild check); concurrency c = 32; hot prefix L_prefix = 1024; both budget levels.

# Hardware
No GPU. CPU-only deterministic replay; minutes on a single modern core.

# Instrumentation Required
Per request: arrival time, routing decision, node, queue wait, prefill cost (tokens), hit/miss, TTFT. Per node: queue-length trace, utilization. Aggregate: memory-accounted hit rate, aggregate mean TTFT, per-class P99/mean TTFT, per-class queue-wait P99, slowdown ratios, FI (report-only).

# Baseline
**B — load-only** at equal total offered load, equal memory-accounted cache budget, equal request set (trace-replay). Fairness: identical scheduler, batching, model, admission, and eviction code; only the routing decision differs.

# Experimental Conditions
- A: affinity-only (greedy longest-prefix, token-weighted-backlog tiebreak).
- B: load-only (minimum token-weighted backlog).
- Routing-metric sensitivity cell: B' = request-count least-loaded (revision R6) at p_hot = 0.8 — verifies the verdict does not flip on the load metric.
- Both budget levels; loads {0.5, 0.65, 0.8} × affinity saturation (overload region excluded from verdicts, reported separately); γ-robustness band.

# Success Criterion
(H1, `hypothesis.md`, at p_hot ∈ {0.8, 0.9}, c = 32, equal total offered load, equal memory-accounted budget, stable region):
- aggregate cache hit rate under A is **≥ G** percentage points above B (G from E0; expected ≥10pp); and
- aggregate mean TTFT under A is **≤ 0.8×** the B value; and
- hot-class P99 TTFT under A is **≥ 2×** the hot-class P99 TTFT under B; and
- cold-class P99 TTFT under A is **≤ 1.2×** its B value.
- FI reported (expected FI(A) < 0.8, FI(B) ≥ 0.95 on the common SLO).

# Falsification Criterion
(H1 kill at this gate; formal grid falsification completed in E2.) If at p_hot = 0.8 and p_hot = 0.9, c = 32 on Poisson arrivals at equal offered load in the stable region **either** (i) the memory-accounted hit-rate gain of A over B is **≤ 5 percentage points** (no reuse advantage to trade off), **or** (ii) hot-class P99 under A stays **within 20%** of its B value — then H1 is dead at its predicted-strongest points and the RQ is terminated, G5 reclassified as an engineering composition (a valid Stage 3A outcome). If the hit leg was pre-registrationally killed by E0 (G < 10pp), criterion (i) does not apply.

# Estimated Complexity
Very Low.

# Expected Runtime
Minutes (2 policies × 3 loads × 2 cache levels × 2 γ settings × 2 points × 10 reps of a 10,000-request deterministic replay).

---

## E2 — H2 threshold sweep in the event-driven simulator (completes H1 grid falsification)

# Experiment ID
E2

# Target Hypothesis
H2 (boundary/scaling claim on the hot class) — and the grid-wide falsification check of H1.

# Purpose
Convert E1's point observation into a boundary claim: find p_hot* ∈ [0.7, 0.9] and c* ∈ [16, 64] such that for all p_hot ≥ p_hot*, c ≥ c* the hot class's P99 TTFT under affinity-only crosses the common SLO (P99 ≤ 2.0 s) while the cold class stays ≥99% attained; verify non-increasing thresholds; and formally test H1 across the full p_hot × c grid in the stable region. Also tests the burstiness robustness axis (Gamma CV) and both budget levels.

# System
**Event-driven discrete-event simulation (cost tier 2; CPU only, no GPU).** Reuses the calibrated model of §1.2 (admission = dispatch; no global FIFO) with exact event dynamics (arrival, admission-cap c, per-node FIFO service, completion, eviction); full skew × concurrency grid, both budget levels, SLO-crossing evaluation. Expected values from E0; the simulator is checked against E0 at 3 anchor points before verdicts are accepted.

# Workload
Per §1.3: 10,000 requests; p_hot ∈ {0.5, 0.7, 0.8, 0.9}; c ∈ {8, 16, 32, 64, 128}; Poisson CV = 1 primary at stable loads; Gamma-burst CV ∈ {2, 4} at matched mean on the reduced grid (p_hot ∈ {0.7, 0.8, 0.9}, c ∈ {16, 32, 64}) as robustness; both budget levels; matched total lengths (hot suffix 1024); fixed max_tokens = 256.

# Hardware
No GPU. CPU-only discrete-event simulation; ~10³ runs × 10⁴ events, hours on a multi-core CPU.

# Instrumentation Required
Per run: per-class P99/mean TTFT, per-class SLO attainment A_hot, A_cold, FI, per-class queue-wait P99, memory-accounted hit rate, aggregate mean TTFT, slowdown ratios; per-node queue-length and utilization traces at the identified threshold points; identification of p_hot*, c* and their monotonicity pattern.

# Baseline
**B — load-only** at equal total offered load, equal memory-accounted budget, equal request set (trace-replay), for every grid cell.

# Experimental Conditions
- A: affinity-only; B: load-only — full grid.
- C: threshold replication (k = 2) at the surviving threshold points as a first look at whether replication moves hot P99 (feeds E3); θ fixed.
- Both budget levels; Poisson primary + Gamma robustness; γ-robustness band.

# Success Criterion
(H2, `hypothesis.md`.)
- There exists a skew threshold **p_hot\* ∈ [0.7, 0.9]** and a concurrency threshold **c\* ∈ [16, 64]** such that for all p_hot ≥ p_hot* and c ≥ c* (each dimension tested with the other fixed at its threshold) the hot class's P99 TTFT under A **exceeds the common SLO (P99 TTFT ≤ 2.0 s)**, while the cold class's P99 TTFT under A **remains within the same SLO at ≥99% attainment**; and
- **p_hot\* and c\* are non-increasing in the fixed dimension**; and
- (expected point values from E0) hot P99 crosses the SLO at **p_hot\* ≤ 0.8** with c fixed at 32, and at **c\* ≤ 32** with p_hot fixed at 0.8; at p_hot = 0.9, c = 64 the hot class's attainment under A falls **below 95%** while the cold class's stays **≥99%**.

# Falsification Criterion
(H2, `hypothesis.md`.) H2 is falsified if **either** (i) the hot class's P99 TTFT under A stays **within the common SLO at ≥95% attainment or better across the entire tested range p_hot ∈ [0.5, 0.9], c ∈ [8, 128]**; **or** (ii) the identified thresholds **fail the monotonicity condition** (e.g., a regression at p_hot = 0.8 disappears at p_hot = 0.9), indicating non-monotone noise. Additionally, this grid completes H1's formal falsification: H1 is falsified if at **every** p_hot ∈ {0.5, 0.7, 0.8, 0.9} and every c ∈ {8, 16, 32, 64} on Poisson at equal load in the stable region the hot P99 under A stays **within 20%** of its B value, or if the memory-accounted hit-rate gain of A over B is **≤ 5pp at all skew levels**.

# Estimated Complexity
Low.

# Expected Runtime
Hours (CPU-only: full Poisson grid ≈ 4×5×2 budget levels × 10 reps ≈ 400 runs + reduced Gamma grid ≈ 288 runs + γ band re-checks).

---

## E3 — H3 controllability test in the calibrated simulator

# Experiment ID
E3

# Target Hypothesis
H3 (routing-attributability and controllability of the hot-class cost).

# Purpose
Before spending any GPU, decide whether the hot-class regression is (a) attributable to the routing decision (H3) rather than to load asymmetry or prompt-length asymmetry, and (b) controllable through the existing threshold-replication routing behavior. If H3 is falsified here, the RQ terminates with no real system run.

# System
**Calibrated event-driven simulator (cost tier 2; CPU only), extended from E2** with (i) the suffix-sensitivity and length-mismatch control variants (revision R6 — the matched-cost control of the pre-revision version is restated: the shared prefix is constitutive of the treatment, so the control varies the SUFFIX, testing whether the effect scales with concentration (p_hot) or with suffix length), (ii) policy C (fixed threshold replication, k = 2) with replication-traffic and additional-KV-memory accounting, (iii) memory-accounted budget equality across A/B/C. Simulator pass verdicts are labeled "queue-concentration passes; the tail claim is untested until E4" (revision R1).

# Workload
Per §1.3, focused on the points that survived E1/E2: the H1 points (p_hot = 0.8, 0.9 at c = 32) and the identified H2 thresholds, plus one above-threshold point (p_hot = 0.9, c = 64); Poisson CV = 1 at stable loads; **suffix-sensitivity pair (hot suffix {256, 1024}) and length-mismatch variant (hot mean 3072)**; both budget levels; 10,000 requests; fixed max_tokens = 256.

# Hardware
No GPU. CPU-only; tens of runs, hours.

# Instrumentation Required
Same per-request and per-node records as E1/E2; plus for policy C: replication bytes, additional KV memory, per-node KV utilization, eviction counts; batch-composition statistics; queue-length and utilization traces at the threshold points.

# Baseline
**B — load-only** at equal total offered load, equal memory-accounted budget, equal request set (trace-replay); the H3 ratio construction uses B as its denominator for both classes.

# Experimental Conditions
- A: affinity-only; B: load-only; C: simple threshold replication (2 replicas of the hot prefix, fixed θ).
- Suffix-sensitivity pair and length-mismatch variant for all three policies.
- Both budget levels; surviving skew/concurrency points.

# Success Criterion
(H3, `hypothesis.md`.)
- At equal total offered load, equal memory-accounted budget, and equal request mix, the ratio **P99_TTFT(hot, A) / P99_TTFT(hot, B) is ≥ 2× the corresponding ratio for the cold class**; and
- simple threshold replication of the hot prefix to **two nodes** (policy C) **restores the hot class's P99 TTFT to within 30% of its load-only value** while retaining aggregate hit rate **within 5 percentage points of affinity-only**; and
- (expected) replication traffic remains **< 10% of total KV transfer**.

# Falsification Criterion
(H3, `hypothesis.md`.) H3 is falsified if **any** of: (i) the hot-class regression **appears identically under load-only routing at the same load** (load-driven, not routing-driven); (ii) the hot-class ratio P99(A)/P99(B) **equals the cold-class ratio within measurement error (±20%) at every tested condition**; (iii) the effect **scales with hot suffix length rather than with p_hot** (suffix-sensitivity control — a prefill-cost artifact, revision R6); (iv) simple threshold replication **fails to move the hot-class P99 by more than 20% in either direction while holding hit rate within 5 points**.

# Estimated Complexity
Low.

# Expected Runtime
Hours (CPU-only; focused point set × 3 policies × 3 length variants × 2 budget levels × 10 reps).

---

## E4 — Real-system verification of surviving claims (H3 on hardware + anchors + length–service curve fit)

# Experiment ID
E4

# Target Hypothesis
H1, H2, H3 — real-system verification of whatever survived E1–E3. Runs **only if E1, E2, and E3 all survive**. **Floor scoping (revision R7):** the 2-node floor validates ONLY the H1/H2 anchors; H3's threshold-replication verdict requires ≥3 nodes and is **"void" on the floor** (every node would be a hot replica on 2 nodes — restoration is undefined); H3 verdicts are drawn only at N ≥ 3.

# Purpose
Confirm on a real engine that (a) the queue-concentration trade-off H1/H2 predicts actually materializes with real prefill cost asymmetry and real batching dynamics; (b) H3's controllability holds with real caching (hit-rate vs eviction trade-offs) and real replication; (c) the result is not framework-dependent (T11) or confounded by hot-node CPU saturation or batch-composition differences; (d) **the cold-class scheduling-order question is observed** (SGLang's longest-prefix-first starvation — the mechanism the simulator cannot represent; recorded as a separate observation, not merged into H1–H3 verdicts, revision R1).

# System
**Existing engine + minimal routing wrapper (cost tiers 3–6; one GPU microbenchmark, then SGLang-style engine with a routing-decision wrapper; no complex system modification).** A thin wrapper intercepts only the routing decision and overrides the destination node per policy A/B/C while all scheduler, batching, admission, and eviction code is shared and unchanged. Engine: SGLang (RadixAttention, longest-prefix scheduling) or a vLLM-with-prefix-cache engine; recorded; H1/H2 anchors repeated on the second engine as the T11 guard if budget allows. **Step 0 of E4 (revision R7, design_review issue 10):** single-GPU length–service CURVE fit across [256, 3072] tokens (linear vs superlinear), not a point calibration of γ — the simulator's linear-service assumption is load-bearing for the hot/cold service ratio; if the fit rejects linearity, the simulator constants are re-fit and E1–E3 verdicts are re-checked before acceptance.

# Workload
Per §1.3, replayed as a real trace: 10,000 requests; surviving skew/concurrency points (minimum: p_hot = 0.8, c = 32; the H2 threshold points; one above-threshold point p_hot = 0.9, c = 64); Poisson CV = 1 primary with Gamma CV = 2 as a robustness replica; suffix-sensitivity pair; both budget levels (real KV budgets); fixed max_tokens = 256; deterministic greedy decoding; fixed seed; warm-up identical to the simulator.

# Hardware
**Primary: 4 GPUs on a single node** (e.g., 4× A10G 24 GB; single-node placement keeps transfer cost constant across policies). **Floor: 2 nodes × 1 GPU** — validates H1/H2 anchors only; **H3 verdict = void on the floor** (needs ≥3 nodes so a node can remain cold). A 7B decoder-only model (e.g., Llama-2-7B), fixed FP16 weights.

# Instrumentation Required
All of E1–E3's per-request records (arrival time, routing decision, queue wait, prefill cost in tokens, hit/miss, TTFT) **plus**: per-node CPU utilization (flag/discard runs >80%); batch-composition statistics; per-node KV utilization and eviction counts; replication bytes and additional KV memory for policy C; per-node queue-length traces; per-class P99/mean TTFT, per-class SLO attainment A_hot, A_cold, FI, slowdown ratios, memory-accounted hit rate, aggregate mean TTFT; the length–service curve fit log (step 0); cold-class scheduling-order observation log (queue-order policy per node, class mix in the wait queue).

# Baseline
**B — load-only** at equal total offered load, equal memory-accounted budget, equal request set (trace-replay), same engine, same scheduler, same batching; only the routing decision differs. H1/H2 anchors referenced to B exactly as in E1/E2; H3's ratio construction uses B as denominator for both classes.

# Experimental Conditions
- A: affinity-only; B: load-only; C: simple threshold replication (2 replicas of the hot prefix, fixed θ).
- Suffix-sensitivity pair (the H3 control on real hardware).
- Surviving skew/concurrency points; both budget levels; (optional) second engine for the T11 check.

# Success Criterion
The surviving claims re-confirmed on hardware, criteria copied exactly as in E1–E3:
- **H1 (at p_hot = 0.8, c = 32, equal load, equal budget, stable region):** memory-accounted hit rate under A **≥G** above B; aggregate mean TTFT under A **≤0.8×** B; hot P99 under A **≥2×** hot P99 under B; cold P99 under A **≤1.2×** its B value.
- **H2 (at the simulator-identified p_hot\*, c\*):** hot P99 under A exceeds the common SLO (2.0 s) for p_hot ≥ p_hot\*, c ≥ c\* while cold attainment **≥99%**; SLO-crossing point reproduces within ±20% of the simulated crossing.
- **H3 (N ≥ 3 nodes):** ratio P99(hot, A)/P99(hot, B) **≥2×** the cold-class ratio; policy C restores hot P99 **within 30%** of load-only while hit rate stays **within 5pp** of affinity-only; replication traffic **<10%** of total KV transfer.
- **FI:** report-only, per §1.1.

# Falsification Criterion
The same thresholds as the simulator gates, now on hardware:
- **H1:** hot P99 under A stays **within 20%** of its B value at every tested point on Poisson at equal load, or memory-accounted hit-rate gain **≤5pp** at all tested skew levels.
- **H2:** hot P99 under A stays within the SLO at **≥95% attainment** across the tested range, or the thresholds **fail monotonicity**.
- **H3 (N ≥ 3):** hot regression appears **identically under load-only**; or hot ratio **equals cold ratio within ±20%**; or the effect scales with suffix length; or replication **fails to move hot P99 by >20%** while holding hit within 5pp.
- **Guards:** runs with any node >80% CPU are flagged/discarded; results reversing on the second engine bound the claim to the tested code path (T11); cold-class observations are reported separately (revision R1).

# Estimated Complexity
Medium.

# Expected Runtime
Multi-day (model download + step-0 length–service curve fit + warm-up, then ≈ (2–3 points) × 3 policies × 2 length variants × 2 budget levels × (2–3) repetitions of 10,000-request replays, plus optional second-engine repeats).

---

## 2. Execution and stopping rules (summary)

| Gate | Runs if | GPU | Kills RQ-4 if | Proceeds if |
|---|---|---|---|---|
| E0 | always | no | (derivation; kills the hit leg pre-registrationally if G < 10pp) | grid + G produced |
| E1 | E0 done | no | H1 dead at p_hot ∈ {0.8, 0.9}, c=32 (hit gain ≤5pp or hot ratio within 20%) | effect present at the points |
| E2 | E1 survived | no | H1 grid-falsified, or H2 falsified (no SLO crossing / non-monotone) | p_hot\*, c\* identified and monotone |
| E3 | E1–E2 survived | no | H3 falsified (load-driven, equal ratios, suffix-scaling, or replication immovable) | H3 survives in simulator |
| E4 | E1–E3 survived | yes (4 GPUs; floor 2 nodes; H3 needs ≥3) | any surviving claim fails its exact threshold on hardware | all surviving claims verified |

**Pre-registered verdict:** H1–H3 each receive a binary PASS/FAIL from the last gate that runs (H3 = "void" on the 2-node floor). Any "partial" outcome (overload-equal-violation, suffix-only explanation, small hit gap, both-classes-moving-together, burst-only effect, framework-dependent result) is reported as **non-discriminating** per `hypothesis.md` Ambiguous Outcome, not as support. Simulator passes are labeled "queue-concentration passes; tail claim untested until E4". No optimization algorithm or new system design is proposed anywhere in this plan; the three policies are existing routing behaviors used as experimental conditions.