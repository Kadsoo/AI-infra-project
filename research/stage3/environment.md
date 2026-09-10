# Stage 3B Environment Record

> Recorded at the start of Stage 3B execution. Every experiment result in this stage must be traced back to this environment. Any GPU-cell result produced on different hardware than the frozen control (A100-80GB) is invalid by pre-registration and will be rejected.

## Machine

| Item | Value |
|---|---|
| Hostname | LAPTOP-1PB54QSI |
| OS | Microsoft Windows 11 Home (Chinese), 10.0.26200 |
| CPU | Intel Core i7-14650HX (24 threads) |
| RAM | 31.8 GB |
| Disk | NTFS (F:\, project root) |

## GPU

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB (8188 MiB) |
| Driver | 596.21 |
| CUDA visible devices | 1 |

**Constraint (frozen control mismatch):** all GPU cells in the four LOCKED_PLANs mandate a single **A100-80GB** (RQ-2/RQ-8) or 3×/4× A100 (RQ-1, RQ-4). This machine has a single 8 GB laptop GPU. An 8 GB card cannot host even the FP16 weights of a 7B model (~14 GB), let alone the LongBench full-KV paired baselines at 13K avg context required by RQ-2 E2, nor the batch-8 decode cells of RQ-2 E3. **Therefore all GPU-gated experiments (RQ-1 E3/E4/E5; RQ-2 E2–E5; RQ-4 E4; RQ-8 E2a/E2b/E3/E4) are BLOCKED on this machine.** They are recorded as not-runnable rather than run on non-conforming hardware. Wave-1 CPU-only gates are fully executable and are the entire scope of this session.

## Software

| Item | Value |
|---|---|
| Python | 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, MSC v.1943) |
| pip | 26.0.1 |
| numpy | 2.3.5 |
| scipy | 1.17.1 |
| pandas | 3.0.1 |
| matplotlib | 3.10.7 |
| scikit-learn | 1.8.0 |
| PyTorch / CUDA runtime | **NOT INSTALLED** |
| vLLM / SGLang / HF Transformers | **NOT INSTALLED** (no serving framework on this machine) |

## Reproducibility notes

- The project directory `F:\AIinfraResearch` is **NOT a git repository** — no git commit can be recorded. For our own experiment code we record SHA-256 file hashes in each RQ's experiment log.
- All random generators seeded with fixed seeds; the fixed seeds are logged per experiment.
- No network access was used to fetch external data; all constants come from the corpus paper notes (`research/paper_notes/`) and the frozen plans.

## Experiment date

2026-08-27 (all Wave-1 experiments in this session).

## Hardware topology (relevant)

- Single laptop; no second node; no 25 Gbps cross-node link; no NVLink pair; no A100. RQ-1 E3/E4's two-node testbed does not exist on this machine.
- Host DRAM tier (PCIe path) exists (32 GB RAM) but RQ-8 E2b/E3's host-tier measurements are gated behind the A100 control and are therefore not run.
