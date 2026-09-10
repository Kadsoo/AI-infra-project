# Review Log — Measurement Method Audit (independent Reviewer)

## R-2026-08-30-01 — Environment inventory review
- **Verdict**: PASS (with BLOCKED note). Inventory did NOT guess; every UNKNOWN marked. Probe `nvidia-smi` byte-faithful (RTX 4060 8GB 596.21), WSL2 kernel confirmed but userland+CUDA absent, V100 doc found only in `reproduction/v100` training configs (32 GiB ×2) and no SSH alias. Marking V as BLOCKED (not NOT USABLE) is correct per §3.4/§14f: missing credential is operator-owned resource, not engineering fixable. Accept `environments.md` as §7 deliverable.
- **Risk**: GPU 4060 vs V100 32G difference confounds absolute comparison; reviewer requires stationarity-only comparison (§8) — satisfied (file explicitly bans absolute ranking).
- **Cherry-picking check**: frozen failures (67%, 20%, 87%) all retained; worst case reported, not best.

## R-2026-08-30-02 — Methodology review (definition audit)
- **Measurement definition**: PASS. Distinguishes user-visible `client_send→first_content` vs `handler→first_content_yield` vs `first_forward→token_sampled` with clock source `perf_counter_ns`. No conflation; answers different questions (SLO vs runtime boundary vs compute). Q4 satisfied.
- **Timestamp minimality**: PASS. Chooses L0 OFF baseline + L1 6-int (handler, submit, start, forward_begin, yield, done) with observer gate 10% before claiming reduced variance. Bans Full L2 as default — correct per §10. Lowest-overhead principle respected.
- **Instrumentation vs knee preservation**: PASS. Gate throughput delta ≤10% matches locked plan; no post-hoc relaxation.
- **Missing**: byte-faithful L1 vs OFF gate not yet measured (deferred to E-04); methodology correctly marks it as blocked, not claimed.

## R-2026-08-30-03 — Hypotheses & decomposition
- **Verdict**: PASS. Competing hypotheses H1-H7 cover client, WDDM, dispatch, prefill/allocator/GC, serialization, workload, general/framework without pre-selecting winner. Each has falsifier (Var ratio, R², single-V100 control). H2 correctly notes WSL2 FAIL 87% is not decisive because kernel≠native. H6 notes c1 tail 19% implies not pure queue. No hypothesis promoted to Skill without phase timing — compliant with durable learning rule.

## R-2026-08-30-04 — Stationarity claim
- **Verdict**: FAIL correctly reported. Raw table 10 runs PID 44756 stable threads 67-68 RSS ±6 MB, no thermal throttling, yet c4 median rel 87% (77 ms abs) >>5% and p95 32% >>10%. Central `CV` 36.67% dwarfs L1 mock 3.9% overhead per `migration_summary §9` → L1 gate correctly BLOCKED. Comparison across 3M-B/D/E (67%→20%→87%) shows GC+sleep helped 3.3x but not to pass → next branch must NOT be another warmup loop per §15; must be client vs server audit. Reviewer agrees to stop warmup loop.
- **Order effects**: interleaved `c1,c4,c4,c1,c1,c4,c1,c4,c4,c1` with rho 0.43 (drift rule |rho|>0.6) → PASS (not drift); intermittent outliers not monotonic.
- **Fairness**: workload token variance <0.08% → not confounding.

## R-2026-08-30-05 — Environment comparison fairness
- **Windows vs V100**: PASS as documented limitation, not stall reason. Single-V100-first policy (§9) isolates TP/NCCL. No absolute ms ranking. Reviewer will keep stage open until at least one L0 on Native Linux (single V100) OR Windows L1 server-side CV measured is produced — per host gate.

## R-2026-08-30-06 — Gate handling & A_01 freeze
- **Threshold handling**: PASS. Central 5% tail 10% unchanged after seeing data; no redefinition. A_01 stays FROZEN; no optimization, scheduler, KV policy, novelty claim in this audit — compliant with §14.

## Open items before STATE A
- [ ] E-03: single-V100 L0 OFF 5 reps ×{c1,c4} byte-faithful raw/processed + stationarity_summary.json
- [ ] OR E-04: Windows L1 sidecar CV_server vs CV_client with observer gate proof
- [ ] After either, re-evaluate H1/H5 vs H2/H3/H4 with F-test and R²

## Recommendation
Approve audit as **informative but not yet VALIDATED**; classification is **STATE F** (blocked by V100 credential) with Windows flagged NOT RESEARCH-GRADE for TTFT. Allow reviewer hand-off; do not invoke Manager stage writer.
