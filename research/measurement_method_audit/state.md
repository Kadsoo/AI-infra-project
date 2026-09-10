# Measurement Method Audit — State (2026-08-30)

## Current State: STATE F — BLOCKED BY OPERATOR-OWNED RESOURCE (V100) + WINDOWS NOT VIABLE FOR TTFT

- **A_01**: SUSPENDED — ENVIRONMENT / MEASUREMENT BLOCKED (since Stage 3M-B/D/E). Do NOT lift.
- **L0 stationarity**: FAIL on Windows/WDDM (c4 median TTFT 87.29% rel, CV 36.67%, 77 ms abs) and WSL2-kernel fallback (same 87%); throughput stable (CV 1-3%) proves failure is first-token path, not full harness. L1 tracing remains BLOCKED because instrumentation effect < baseline variance.
- **V100**: 2xV100 32GB documented in `reproduction/v100` (training context), but no serving stack and no reachable SSH/tunnel on this workstation → BLOCKED per §3.4. Cannot run L0 on V100 without operator action. Do NOT claim V100 PASS.
- **Next gate**: operator provides V100 serving access OR confirms alternative Linux GPU. Until then audit completes with definitions + decomposition plan, not a reopened A_01.

## What was checked 2026-08-30
- Inventory exhaustive grep for V100/SSH/NVLink across F:\papers and F:\AIinfraResearch: found only `reproduction/v100` training configs + paper-notes GEAR etc., zero serving SSH alias.
- Frozen artifacts re-checked: `research/environment_linux.md`, `measurement/environment.md`, `environment_migration/l0_stationarity.md`, `migration_summary.md`, `measurement/harness/hf_server.py`, `minimal_trace.py`, `harness.py`.
- Live probes: `nvidia-smi` (RTX 4060 8GB 596.21), `wsl --list --version`, `torch.cuda.is_available()==False` (2.13.0+cpu), single-GPU WDDM confirmed.
- Environments file written: `research/measurement_method_audit/environments.md` with W=AVAILABLE/NOT-RESEARCH-GRADE, V=BLOCKED, WSL2=PARTIALLY AVAILABLE.
- Single-V100-first policy (§9) recorded; dual-V100 deferred until single passes.

## Terminal classification
- **STATE F** (blocked by operator-owned resource) for V100 access.
- Local Windows branch is **STATE D-like**: WINDOWS PLATFORM NOT RESEARCH-GRADE FOR THIS METRIC (TTFT), but broader claim stays STATE F because general conclusion requires control env that is still blocked.
- Measurement method audit proceeds to definition + decomposition; no Stage 4.

## Required operator decision (one question)
> Provide reachability for the 2xV100 Linux host that backs `/home/yaoyunchao/conteb-v100-20260826` (SSH alias / VPN / tunnel + authorized `hf_server.py` port) and confirm can run single-V100 `sshleifer/tiny-gpt2` serving on port 8036, OR confirm no V100 serving access for this project.

## Blocking details per §20 STATE F
1. Missing: SSH host alias / VPN / campus-net credential for the conteb-v100 host, or explicit statement that no serving allocation exists.
2. Why needed: L0 requires Native Linux GPU to separate platform (WDDM jitter) from measurement-vs-runtime variance; Windows alone cannot disambiguate (WSL2 kernel already failed to fix 87% jitter).
3. Why blocked without it: cannot run independent 5× c=1/4 L0 OFF on single V100; any claim about platform-specific vs general would be assumption, not evidence.
4. After credential: run minimal L0 (c=1,4 ×5 reps, 512/64, 40 req, pooled client, fixed 32/16 threads, L0 OFF) on single V100, interleaved, raw preserved, gate central ≤5% tail ≤10%.

## Artifacts produced
- `environments.md`, `state.md`, `hypotheses.md`, `methodology.md`, `experiment_log.md`, `review_log.md`, `final_audit.md`, `environments/v100.md` (doc-only), `raw/` `processed/` `controls/` (empty until L0).

## Handoff
- next_owner: reviewer (independent review REQUIRED; stage_transition DISABLED; do not invoke Manager stage writer)
