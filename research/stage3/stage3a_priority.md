# Stage 3A Priority — Which Experiments to Run First

> Date: 2026-08-27. All four Tier 1 RQs were reworked pre-freeze after adversarial review (`design_review.md` per RQ). This file ranks them for Stage 3B execution by Evidence Strength, Expected Impact, Cost to Validate, Falsifiability, and Risk of Confounding.

## 1. Score table

Scores 1–5 (5 = best for that dimension; Confounding 5 = lowest risk).

| RQ | Evidence | Impact | Cost (5 = cheap) | Falsifiability | Low Confounding | Total | Priority |
|---|---:|---:|---:|---:|---:|---:|---|
| RQ-2 Static budget fragility | 5 | 5 | 4 | 4 | 3 | **21** | **1** |
| RQ-1 P/D vs colocated crossover | 5 | 5 | 3 | 4 | 3 | **20** | 2 |
| RQ-4 Affinity vs per-class tail/fairness | 4 | 4 | 4 | 4 | 3 | **19** | 3 |
| RQ-8 Token × bit-width × tier non-additivity | 4 | 3 | 4 | 4 | 3 | **18** | 4 |

(Total is a rough composite, not a weighted score; the order below is justified qualitatively.)

## 2. Ranking rationale

### #1 — RQ-2 (run first)
- **Evidence:** strongest in the corpus — multiple in-paper within-model signals of budget fragility (SnapKV −29% GovReport, KIVI Falcon 4-bit requirement, SCOPE phase-split, GEAR CoT collapse); the rework removed the "we already know this" objection by moving the claims to out-of-suite conditions.
- **Impact:** foundational — a positive result gates any future adaptation work; a null kills the overgeneralized claim cheaply.
- **Cost:** E1 is desk analysis; E2–E4 run on ONE A100 with official repos/harnesses (no system engineering). Total ≈ 3–5 GPU-days.
- **Falsifiability:** binary per gate; kill lines reachable; phase cell unconditional; bootstrap CIs pre-registered.
- **Residual risk:** dataset noise floor (mitigated by the ±2-point band) and repo-fidelity (mitigated by using the methods' own implementations).

### #2 — RQ-1 (run second; cheap gates can go first)
- **Evidence:** highest — the P/D-vs-transfer tension is documented on both sides with measured numbers.
- **Impact:** high — an operating-boundary result constrains any future P/D or placement work.
- **Cost:** E1 (model) and E2 (simulator) are CPU-only and should run immediately — they now have reachable kill lines. Only E4 needs the 3× A100 multi-node testbed (the most expensive hardware of all four RQs), and E1/E2/E3 must survive before it.
- **Residual risk:** implementation maturity of vanilla vLLM-Disagg and node-identity effects (mitigated by pinned commits, balanced node layout, and the serialized-transfer fallback cell).

### #3 — RQ-4 (run third; cheapest CPU path of all)
- **Evidence:** Medium — the trade-off direction is documented in prose everywhere but never measured per-class.
- **Impact:** Medium-High — the deliverable is the field's first per-class quantification with a declared fairness index; useful as a scheduler-design constraint, confirmatory rather than agenda-setting.
- **Cost:** E0 (algebra, hours) + E1–E3 (CPU-only) can kill the RQ without any GPU; E4 needs 4 GPUs (single node) and is only justified after E0–E3 survive.
- **Residual risk:** this RQ received the deepest restructure (the original hypothesis was unsatisfiable under its own model); residual modeling risk is mitigated by the E0 algebra step, the per-policy load anchoring, and the E4 length–service-curve fit. Positive claims depend on E4 — the simulator alone cannot confirm the tail claim (within-node scheduling order is unmodeled).

### #4 — RQ-8 (run fourth in priority; its cheap decisive path should still be scheduled early)
- **Evidence:** Medium-High (T12 tension + hybrid papers), but the positive is a confidence-upgrade in a narrow envelope; **the null is the genuinely novel outcome** — and the design now makes the null cleanly reachable.
- **Impact:** Medium — it is a gate for Stage 4 (whether a compound mechanism is warranted), not a standalone mechanism claim.
- **Cost:** E1 (minutes), E2a (hours), E2b (hours) can answer much of the question at microbenchmark cost; only E3/E4 need the vLLM-class port, which is rated High with a pre-registered 10-person-day budget and a fallback.
- **Residual risk:** the glue semantics are now pre-registered (the decoupled arm separates mechanism from glue), but the E3/E4 port is the largest single engineering risk in the program.

## 3. Recommended execution schedule (Stage 3B, parallelizable)

| Wave | Workload | RQ experiments |
|---|---|---|
| W1 (no GPU, parallel) | hours–days CPU | RQ-1 E1+E2; RQ-2 E1; RQ-8 E1; RQ-4 E0+E1 |
| W2 (single A100) | ~2 weeks | RQ-2 E2–E4; RQ-8 E2a+E2b (+E3 if survived); RQ-4 E2+E3 (CPU, same wave) |
| W3 (cluster: 3× A100 2+1; then 4 GPUs) | 1–3 weeks | RQ-1 E3/E4 (if E1+E2 survived); RQ-4 E4 (if E0–E3 survived) |

Wave 1 can kill RQ-4 (E0+E1), RQ-1 (E1/E2), RQ-8 (E1) and shrink RQ-2's matrix before any GPU is spent. Kill gates stop the pipeline: a Wave-1 null saves the corresponding Wave-2/Wave-3 budget.

## 4. Kill-early summary (expected savings)

| RQ | Cheapest decisive gate | Cost to reach it | What it can decide |
|---|---|---|---|
| RQ-1 | E1 (analytical) + E2 (simulator) | days, CPU | H1 + H2 (kills the RQ if both dead) |
| RQ-2 | E1 (triage) + E2 wave 1 | ~2 GPU-days | H1 (kills it if the out-of-suite cells cross) |
| RQ-8 | E1 + E2a + E2b | ~1–2 GPU-days | H1 + H2 + the primary's interaction surface |
| RQ-4 | E0 + E1 | hours, CPU | H1 (kills the RQ if dead at both test points) |

## 5. Traceability

Scores synthesize `stage2_final_review.md` §8 (Tier-1 rationale), `research_tensions.md` (T1/T3/T4/T7/T12), the four `hypothesis.md`/`experiment_plan.md`/`design_review.md`/`LOCKED_PLAN.md` pairs in `research/stage3/`, and the post-review fix records in each `design_review.md` §3.