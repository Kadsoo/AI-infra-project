# Design Review — RQ-8 (Token × Bit-width × Tier Non-Additivity)

> Stage 3A adversarial review. Critic verdict: **require rework before freezing**. All fixes below were applied to `hypothesis.md` and `experiment_plan.md` BEFORE freezing; the frozen state is `LOCKED_PLAN.md`.

## 1. Critic verdict summary

| # | Question | Verdict | Severity |
|---|---|---|---|
| 1 | Does the experiment really test the hypothesis? | FIX-REQUIRED | High |
| 2 | Same result from a different cause? | FIX-REQUIRED | High |
| 3 | Is the baseline fair? | WARN | Medium |
| 4 | Is the workload cherry-picked? | WARN | Medium |
| 5 | Are success criteria post-hoc? | FIX-REQUIRED | High |
| 6 | Are there hidden variables? | FIX-REQUIRED | Medium |
| 7 | Can a cheaper experiment kill the hypothesis? | PASS | — |
| 8 | Re-statement of an existing paper conclusion? | WARN | Medium |
| 9 | Enough impact if true? | WARN | Medium |
| A | Statistical soundness | FIX-REQUIRED | High |
| B | Implementation risk | FIX-REQUIRED | High |
| C | Faithful stacking or novel system? | FIX-REQUIRED | High |
| D | D3 blind spot → false null? | WARN | Medium |
| E | Null suppressed or explained away? | WARN | Medium |

## 2. Key findings (abridged)

1. **Unattributable interaction (mechanism vs glue).** The H2O-eviction × KIVI-2-bit combination has four unspecified semantics that determine the numerics: eviction/quantization ordering, H2O recent-window size vs KIVI's R=128 residual, the attention-score source (FP16 vs 2-bit-reconstructed), and group-statistics semantics under eviction compaction. Any of these can produce a measured "interaction" that is an artifact of the glue, not of the axes composing non-additively — undermining the study's core claim ("stacking EXISTING mechanisms").
2. **Threshold calculus contradiction.** Support uses R²_int absolute (SS_int/SS_total on per-request data); falsification uses R²_int_rel (SS_int/SS_main). Retention's main effects dominate variance by construction, so the falsification bar is near-automatic while the support bar can be unreachable for a real modest interaction — a genuine interaction can be simultaneously unsupported and unfalsified. The 15% constant does triple duty; the gray zone is un-pre-registered.
3. **"Overall null" shortcut contradicts the hypothesis.** Plan declares the overall null from E1+E2a alone, but the hypothesis's overall null requires all three sub-hypotheses falsified — H3 (Pareto) is never tested by E1/E2a and can hold via main effects alone.
4. **E3-null overclaims the primary's falsification.** The 2×2×2 endpoint design (R ∈ {20%, 100%}) cannot see tier non-additivity at R=60% (fetch-count lumpiness), and the D3 "closed by E4 if E3 survives" blind spot is never closed on the kill path.
5. **Implementation risk misrated.** The KIVI-in-vLLM port (per-block quantized metadata, residual flush across 16-token pages, fused dequant over non-contiguous layouts, block-manager eviction wrapper, offload path) is weeks of engineering, not "minimal modification" — and the kill-early economics rest on this being cheap.
6. **Regression coding + H1 asymmetry.** "R linear in budget %" cannot separate a nonlinear R main effect from an R×B interaction at 3 levels; E4's coding is unspecified; H1's support needs ≥1 workload but its kill needs both — a single false-positive workload keeps the RQ alive.

## 3. Resolution table (issue → resolution applied)

| # | Fix | Where applied | Status |
|---|---|---|---|
| 1 | **Four glue semantics pre-registered** (critic #2/#C): (i) H2O eviction is always scored on FP16 attention (decoupled arm: eviction decides the retained set on FP16; quantization is then applied to the retained set — isolates "error lands on heavy hitters" from "2-bit changed the heavy hitters"); (ii) H2O recent-window protection = 128 tokens = KIVI R (exact overlap, pre-registered); (iii) quantization groups formed over retained tokens in position order, scales/zero-points recomputed per group after eviction; (iv) residual queue = the last 128 retained positions. Fake-quant cell added to ≥1 GPU+host E3 cell so dequant-on-fetch-path cost is attributed at the tier cells | hypothesis.md (H1 mechanism note), experiment_plan.md (E2a/E3) | Applied |
| 2 | **Unified statistics** (critic #5/#A): one denominator for support AND falsification: R²_int_rel (SS_interactions/SS_main); ANOVA on cell means (not per-request data); support = significant (α=0.05, BH across metric×workload families) AND R²_int_rel > 0.50 in ≥1 metric & ≥1 workload; falsification = non-significant AND R²_int_rel ≤ 0.10 for all; gray zone [0.10, 0.50] significant → inconclusive → E4 required; CI on R² estimates reported; N from pilot cell-means variance | hypothesis.md (Primary Hypothesis, Falsification), experiment_plan.md (§2 statistics) | Applied |
| 3 | **Overall-null shortcut reworded** (critic #1): E1+E2a null path declares primary + H1 + H2 null; H3 explicitly NOT assessed at that stage (needs measured coordinates from E3/E4) | experiment_plan.md (§1 pipeline) | Applied |
| 4 | **E3-null relabeled + blind-spot cell added** (critic #1/#D): E3-null = "additive at R ∈ {20%, 100%} endpoints"; the corner cell (60%, 2-bit, GPU+host) added to E3's W2 repeat (9th cell); full primary falsification reserved for E4 | experiment_plan.md (E3) | Applied |
| 5 | **Implementation risk re-rated** (critic #B): E3 rated High; pre-registered port budget ≤10 person-days with trigger; pre-registered fallback (HF batch-1 harness + real offload path + emulator at the ≤15% anchor-calibration standard) so a stalled port cannot silently cancel the tier question | experiment_plan.md (E3/E4 complexity) | Applied |
| 6 | **Regression coding unified** (critic #6): R coded as categorical contrasts in BOTH E2a and E4 (robustness check against linear coding reported); BH correction extended across term×workload families; H1's asymmetric burden (≥1 workload support vs both-workload kill) explicitly pre-registered as a design choice with a per-workload verdict reported | hypothesis.md (H1), experiment_plan.md (§2, E2a, E4) | Applied |
| 7 | Q8/Q9 accepted as-is: the null is the genuinely novel outcome; the positive is a confidence-upgrade measurement in a narrow envelope. Framed accordingly in LOCKED_PLAN (the E4 budget is explicitly gated behind E1/E2/E3) | LOCKED_PLAN.md | Applied |

## 4. Accepted as-is (no change)

- E1 analytical screening (honestly scoped, minutes/no-GPU) and the kill-early structure.
- E2a at 3×2×2 = 12 cells (near-minimal for H1's ratio test; the 100% cells anchor the main effects).
- E2b microbenchmark as H2's empirical gate (correct use of microbench+model).
- Pre-registered exits ("not testable in this regime", emulator-fidelity downgrade) — disclosed, retained.

## 5. Critic bottom line (verbatim)

> "The plan is close — genuinely pre-registered, confounder-aware, correctly kill-ordered — but it cannot be frozen as-is. Four defects change what the study would be able to conclude: (a) the measured 'interaction' is unattributable... (b) the statistical calculus can make the primary simultaneously unsupported and unfalsified... (c) the pipeline's cheap-null paths claim the overall null and the primary's falsification without testing H3 or the 60% retention level... (d) the harness port cost is understated by an order of magnitude with no fallback... Once issues 1–5 are resolved and the E2a/E4 coding is unified (issue 6), the plan is defensible."

Resolved by fixes 1–6. Freeze proceeds on the corrected versions.