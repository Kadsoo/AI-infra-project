# Final Audit — Measurement Method Audit & Environment Recovery (2026-08-30)

**Final State: STATE F — BLOCKED BY OPERATOR-OWNED RESOURCE** (dual-V100 Linux not reachable from this workstation; Windows NOT RESEARCH-GRADE for TTFT)

> A_01 remains **SUSPENDED — ENVIRONMENT / MEASUREMENT BLOCKED**. Do NOT unfreeze. Even if later gate passes, mark READY FOR OPERATOR/GPT REVIEW only.

## 1. Environments used
- **W**: Windows 11 + RTX 4060 Laptop 8GB WDDM 596.21 (LAPTOP-1PB54QSI) — AVAILABLE, probed live, frozen `environment_linux.md` + `measurement/environment.md` + `environment_migration/l0_stationarity.md` (PID 44756).
- **V**: 2xV100 32GB conteb-v100-20260826 (`/home/yaoyunchao`) — documented in `F:\papers\reproduction\v100/*` yml/sh (32 GiB ×2, sdpa, fp16 autocast, 2-process DDP), but **no SSH/VPN/host alias** found in exhaustive repo grep; **BLOCKED**. Native Linux GPU server marked BLOCKED in `research/environment_linux.md` §1 (no IP/SSH). No new Linux provisioned.
- **WSL2 fallback**: kernel 6.6.87.2 only (docker-desktop), no Ubuntu userland/python3/nvidia-smi; install hangs; L0 FAIL 87% — same pathology, not research-grade.

Unused/missing is documented limitation, not stall excuse; re-probe before use.

## 2. Did we find and use local 2xV100?
- **Found**: YES — in `F:\papers\reproduction\v100` (training domain, not serving). Configs prove `n_gpus 2`, 32 GiB per GPU, `torch 2.2.2 cu121`, Python 3.10, accelerate `multi_gpu 2`, path `/home/yaoyunchao/conteb-v100-20260826`. Full inventory in `environments.md` §V with UNKNOWN marks for PCIe/SXM/NVLink/CPU/RAM/kernel/driver version/distro.
- **Used**: NO for serving measurement — blocked per house rule: do not output passwords, do not commit secrets, do not modify unrelated jobs, require authorized SSH/tunnel. No credential on this Windows box; probing `/home/yaoyunchao` unreachable.

## 3. V100 actual config
Documented per §3.1 table; key: 2×V100 32GB, driver CUDA 12.2 support, CUDA 12.1 wheel, torch 2.2.2+cu121, transformers 4.48.0, sdpa, attn sdpa (flash not buildable), per_device batch 2 global 4, gradient_checkpointing reentrant=false, DDP 2×. Everything else UNKNOWN / NOT DOCUMENTED (PCIe/SXM/NVLink/CPU/RAM/distro/kernel/topology not in files). See `environments/v100.md`.

## 4. Major experiments (what was actually done)
- **E-2026-08-30-01** inventory probe (read-only): exhaustive V100/SSH/NVLink search + live `nvidia-smi`/`wsl`/`torch` probes — evidence packet stored, no GPU load.
- **E-2026-08-30-02** re-validation of frozen L0 (read-only): 10-run simple-warmup gate (c1/c4 ×5) PID 44756 interleaved — c1 median 3.28% PASS but p95 19.78% FAIL; c4 median 87.29% CV 36.67% FAIL, p95 32.46% FAIL; thr c4 6.21% FAIL vs throughput stable prior (2.90%). Compared across 3M-B 67% →3M-D 20% (GC+sleep helped 3.3×) →3E 87% worst; proves simple uniform warmup insufficient and WSL2 kernel did NOT rescue. Reviewer stopped warmup loop per §15.
- No new serving runs executed this audit that required GPU model load (to avoid violating STATE F / no credential). L0 on single V100 and Windows L1 sidecar remain planned but blocked/gated (see next gate).

## 5. Hypotheses negated
- **Warmup/GC alone fixes TTFT**: negated — 3E simple 87% worse than 3M-D 20% with GC; WSL2 fallback same failure.
- **Throughput stable ⇒ system stable**: negated — thr CV 1-3% PASS masks TTFT 36% FAIL; first-token path specific.
- **Throughput/latency variance explains TTFT**: negated — p95_lat PASS (4.76% c4) while TTFT catastrophic.
- **WDDM proven not cause**: NOT negated nor proven — WSL2 kernel failed but not Native Linux; remains OPEN (H2).
- Remaining H1-H7 stay OPEN with falsifiers defined.

## 6. Strongest verified observations
- [EXPERIMENTAL OBSERVATION] Windows RTX4060 WDDM + `hf_server.py` naive (fixed executor 32, torch 16/16, pooled httpx, tiny-gpt2 512/64) shows **TTFT non-stationarity specific to first token** across 3 independent PIDs (19372, 34892, 44756) and 3 warmup variants: c4 median rel 67% →20% →87%, all >>5%. Absolute 77 ms on 88 ms median.
- [SOURCE-CODE FACT] Naive `asyncio.to_thread` per-token loop (`_stream_real` 64× to_thread + past_key_values growth) + L1 6-int minimal tracing available but unused; observer effect not measured against 87% baseline (gate would be meaningless).
- [DOCUMENTED FACT] `research/environment_linux.md` explicitly marks Native Linux GPU server BLOCKED; exhaustive grep found zero serving SSH doc.
- [HYPOTHESIS] WS-1/4/5 candidates (server-anchored TTFT, CUDA-event operator, WDDM limit) remain live but untested at measurement gate.

## 7. Client vs Server TTFT conclusion
- **Client TTFT** (`send→first_content`) variance dominated by 77 ms run-to-run jitter plus 6 ms pooled overhead and p95 23 ms tail; unstable at both c=1 tail and c=4.
- **Server TTFT** (`handler_enter→first_content_yield` or tighter `→first_token_sampled`) hypothesized more stable if H1/H5 holds (`CV_server << CV_client`), but **not yet measured** — L1 sidecar blocked until observer gate can be validated against ≤10% thr delta. Q1/Q2 remain OPEN. Methodology for direct sidecar comparison defined (`F-test Var ratio`, `CV_server` vs `CV_client`, slice `executor_wait` vs `model_to_yield`).

## 8. Measurement method current status
- Definitions frozen: user-visible vs server vs execution (Q4).
- Lowest-overhead instrumentation path defined: L0 OFF baseline → L1 6-int (handler/submit/start/forward_begin/yield/done) → L1b sampled → L1c counters; Full L2 banned. Owner gate ≤10%.
- L0 stationarity gate locked (central ≤5% CV ≤5% tail ≤10% outlier ≤20% drift rho>0.6&slope>2%/step, 5 reps interleaved, raw preserved).
- **NOT validated**: no environment currently passes L0 for TTFT; method ready but not research-grade until at least one env passes.

## 9. Windows vs V100 stationarity contrast
- Windows (actual): c1 median PASS 3.28% but tail FAIL; c4 FAIL 87%/32% → **NOT research-grade for TTFT**.
- V100 (candidate): **no L0 data** — cannot contrast absolute ms; hypothesis is platform-specific if single-V100 Native Linux passes, framework-general if it also fails. WSL2 kernel (half-Linux) failing 87% suggests not pure OS string → need Native.
- Comparison is *stationarity-only*, not speed.

## 10. Remaining uncertainties
- Is TTFT variance client/emission vs runtime vs WDDM vs past_key_values/GC? Requires E-03/E-04.
- Does server TTFT fix CV to ≤5% on Windows alone (STATE B) or need Linux migration (STATE D/E)?
- V100 32GB path for serving (not training) may need different quantization / tiny-gpt2 vs ModernBERT; recorded as limitation.

## 11. Most important artifacts
- `research/measurement_method_audit/environments.md` (§7 compliance, §3.1 full table)
- `environments/v100.md` (V100 config + access status)
- `methodology.md` (Q1-Q5 definitions + L0 gate + L1 6-int)
- `hypotheses.md` (H1-H7 competing, falsifiers)
- `experiment_log.md` + `review_log.md` + `state.md`
- Frozen upstream: `F:\AIinfraResearch\research\environment_migration\raw/quick-3e_*` (10 runs), `processed/stationarity_summary.json` (NON-STATIONARY), `research/environment_linux.md` (WSL2-kernel-only)

## 12. Exact next gate
- **If operator grants V100**: Run E-03 single-V100 L0 OFF (c=1,4 ×5, 512/64, n=40, pooled, 32/16, interleaved, L0 OFF) — preserves `raw/` (json/csv/system.csv/manifest/trace_count/health) + `processed/` + `stationarity_gate.json` with rel_range/CV/rho/outlier. PASS (central ≤5% tail ≤10% + no drift) → promote V100 to VALIDATED RESEARCH ENVIRONMENT, then proceed to E-04 server-side comparison on V100.
- **If V100 stays blocked**: Run E-04 Windows L1 sidecar (c=4 ×5 OFF vs 5 L1, same workload) — prove observer gate <10% thr, then compare `CV_server` vs `CV_client` (F-test). If `CV_server ≤5%` while `CV_client` FAIL → STATE B (measurement artifact) → fix metric to `TTFT_server` and re-gate; else if both FAIL → STATE C/E (runtime/general).
- Both retain A_01 SUSPENDED and require external review before reopening.

## 13. A_01 still suspended?
YES — SUSPENDED — ENVIRONMENT / MEASUREMENT BLOCKED. No causal knee claim, no scheduler/KV optimization, no Stage 4. Even if next env passes, mark READY FOR OPERATOR/GPT REVIEW only.

## 14. Decisions needing external GPT/operator
- Confirm V100 serving reachability (SSH/VPN/tunnel, port, single-V100 reservation, authorized `hf_server.py` run) or explicitly deny it so audit closes as STATE E/D with documented limits.
- Approve next L1 sidecar on Windows if V100 denied (low-overhead check).
- Review methodology locks (5%/10%/10%) — do NOT relax without principled justification.

*Probe receipts retained; no simulation counted as serving evidence.*
