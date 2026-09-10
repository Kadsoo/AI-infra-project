# Experiment Log — Measurement Method Audit (2026-08-30)

> Raw artifact contract §11: every formal run preserves raw+processed+config+command+PID/port+commit diff+workload token counts+hardware+balanced order. Exploratory vs confirmatory separated.

## E-2026-08-30-01 — Environment inventory (confirmatory, no GPU load)
- **Goal**: answer §3.1 V100 checklist without guessing.
- **Probe**: `Get-ChildItem` + `Select-String`/`rg` fallback over `F:\papers` and `F:\AIinfraResearch` for `V100|GPU server|SSH|NVLink|2 GPU`; `nvidia-smi --query-gpu=...`, `wsl --list --verbose --version`, `wsl -d docker-desktop -- uname -a`, `python -c "import torch"` (probed via .venv), `pip show torch/transformers`.
- **Result**: W verified single RTX 4060 Laptop 8GB WDDM 596.21 CUDA 13.2 driver, no toolchain; WSL2 kernel 6.6.87.2 present but no Ubuntu userland/python3/nvidia-smi; `F:\AIinfraResearch` reports Native Linux BLOCKED; `F:\papers\reproduction\v100` confirms 2xV100 32GB training domain (`n_gpus 2`, `torch 2.2.2 cu121`, Python 3.10, path `/home/yaoyunchao/conteb-v100-20260826`) but zero serving SSH/host alias. Full table in `environments.md` (§3.1) with UNKNOWN marks.
- **Raw**: `raw/probe-2026-08-30-01-*.log` (to be byte-faithful when next L0 runs); this log is summary.

## E-2026-08-30-02 — Re-validate Windows L0 (read-only, uses frozen)
- **Goal**: confirm latest stationarity without new run (avoid repeating failed warmup loop per §15).
- **Source**: `F:\AIinfraResearch\research\environment_migration\l0_stationarity.md` (PID 44756, 2026-08-29 17:45, 10 runs interleaved) + `migration_summary.md` + `environment_linux.md`.
- **Observations**: c1 median 3.28% PASS but p95 19.78% FAIL; c4 median 87.29% (77 ms abs) CV 36.67% FAIL, p95 32.46% FAIL outlier 20.8%; thr c4 6.21% FAIL (vs prior 2.02% PASS); rho 0.43 no drift but intermittent 2/5 outliers ~150 ms vs 3/5 ~74 ms. Comparison: 3M-B 67%, 3M-D 20% (GC+sleep helped 3.3x but not enough), 3E simple 87% worst. Throughput stable proves TTFT-specific. Decision: do NOT repeat identical warmup; next must distinguish H1/H5 vs H2/H3/H4 (§15 branch rule).
- **Artifacts**: cited frozen JSONs/CVS in `F:\AIinfraResearch\research\environment_migration\raw/quick-3e_*`, `processed/stationarity_summary.json` (NON-STATIONARY).

## Prior branches reviewed (§15: every 2-3 branches reviewer judges information gain)
- 3M-B (2×c8 n20): severe 67% — learned warmup insufficient.
- 3M-D (Protocol D 1×c4+GC+sleep): 20% — learned GC helps but not to 5% → allocator/GC contributes but not sole.
- 3E simple (1×c4 n40): 87% — learned simple uniform warmup insufficient and worse; platform not rescued by WSL2-kernel.

Reviewer verdict 2026-08-30: repeating 1-variable warmup has diminishing gain; higher-gain branch is server-vs-client decomposition (Q1/Q2) then Native Linux control (Q5).

## Planned next (gated, not yet run — require V100 access or reviewer approve Windows-only L1)
- **E-2026-08-30-03 (next if operator grants V100)**: Single-V100 L0 OFF, c=1,4 ×5 reps, 512/64 n=40, pooled, 32/16, L0 OFF, interleaved. Gate central ≤5% tail ≤10%. Success → promote V100 to VALIDATED RESEARCH ENVIRONMENT; fail → evidence platform not sole cause.
- **E-2026-08-30-04 (alternate if V100 stays blocked)**: Windows L1 minimal 6-int sidecar on same workload, c=4 ×5 reps OFF vs 5 reps L1, observer gate <10% thr delta, compare CV_server vs CV_client (Q2). If CV_server ≤5% while CV_client >12% → support H1/H5 (measurement artifact) → fix metric to TTFT_server.

## Ordering
- Balanced Latin-square order locked: `c1,c4,c4,c1,c1,c4,c1,c4,c4,c1`; per-run seeds fixed; workload token stats logged via `describe_workload` (input mean 510.8±11.2, <0.08% variance).

## Retention
- No silent deletions; failures kept in `raw/`; exclusion logged with reason; processed derived via `compute_processed` (median/p95 thr/lat/TTFT, system CPU/RSS/threads).

*Next update after E-03 or E-04 probe byte-faithful logs.*
