# Research Opportunity Matrix — Stage 2B

> This matrix marks only cells with material local-corpus evidence. A blank (`—`) means **not assessed from the supplied corpus**, not “no work exists.”  
> It is an opportunity map, not a novelty map: `Potentially interesting` requires Stage 3 validation and later novelty search.

## Legend

| Mark | Meaning |
|---|---|
| **W** | Well studied in the corpus; further work needs a sharply different condition or evaluation claim. |
| **A** | Active; multiple approaches exist but the operating boundary remains unsettled. |
| **S** | Sparse local evidence or narrow/partial evaluation. |
| **C** | Conflicting or strongly conditioned evidence; a matched study is especially valuable. |
| **I** | Potentially interesting Stage 3 cell, conditional on a testable hypothesis. |
| **—** | No material marking made from this corpus. |

## Evidence-marked matrix

| Solution / workload family | TTFT | TPOT | Throughput | Memory | P95/P99 | Bandwidth | Utilization | Scalability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Memory allocation / layout** | A | — | A | **W** | S | A | — | S | — |
| **KV reuse** | A | — | A | A | S | A | A | A | **C** |
| **Eviction / retention** | A | A | A | **W** | S | — | A | — | **C** |
| **Structural / numerical compression** | S | A | A | **W** | — | — | A | — | **C** |
| **Offloading / tier placement** | **C** | A | **C** | **W** | S | **C** | A | S | — |
| **KV transfer / communication** | A | A | A | — | S | **W** | A | S | — |
| **Scheduling / routing** | A | A | **W** | A | **A** | — | **W** | A | — |
| **Batching / preemption** | **C** | A | **W** | A | A | — | **W** | — | — |
| **P/D disaggregation** | A | A | A | A | **C** | **C** | A | A | — |
| **Distributed serving / elasticity** | A | A | A | A | S | A | A | **S** | — |
| **Emerging RAG / agent / long-context workloads** | S | S | S | S | S | S | S | S | A |

## How to read the important regions

| Region | Why it is marked | Evidence boundary | Stage 3 implication |
|---|---|---|---|
| **KV reuse × quality (C)** | Exact-prefix reuse is mature, but non-prefix reuse has different repair/training trade-offs. | CacheBlend, Cache-Craft, and KVLink do not use one shared workload/hardware/quality harness. | Compare reuse validity, quality, TTFT, and storage at the same operating point. |
| **Eviction/compression × quality (C)** | Strong memory reductions coexist with task/model/length sensitivity. | KIVI, GEAR, H2O, PyramidKV, SnapKV, and SCOPE use different quality suites and budgets. | Test whether a policy fails under a declared task switch rather than seek a generic compression claim. |
| **Offload × TTFT/bandwidth (C)** | Capacity gains may turn into fetch/coherence/control-path delay. | FlexGen, InfiniGen, ShadowKV, LMCache, and Beluga target different tiers and machines. | Establish a topology/granularity boundary for fetch versus recompute. |
| **P/D × bandwidth (C)** | P/D resolves interference only when transfer and placement overheads remain bounded. | DistServe/Splitwise, FlowKV, Mooncake, and Beluga represent distinct paths/topologies. | Trace a phase split under controlled network/load variations. |
| **Scheduling × P95/P99 (A)** | SLO-aware systems exist, but cache affinity/fairness and burst behavior are unevenly measured. | Sarathi, FastServe, DistServe, Mooncake, and SGLang use non-identical SLO definitions. | Measure a joint hit-rate/throughput/tail/fairness frontier. |
| **Distributed serving × scalability (S)** | Multi-node studies exist but controller, incast, failure, and large-scale evidence are bounded. | Mooncake/Beluga/LMCache/FlowKV vary in scale and topology; DéjàVu covers a different pipeline failure model. | Treat it first as a scale/reliability evaluation question. |
| **Emerging workloads × all service metrics (S/A)** | RAG, agents, and long contexts motivate papers, but their traces and workflow semantics remain heterogeneous. | KVFlow/Continuum offer direct agent evidence; Cache-Craft offers RAG evidence; no unified workload matrix is present. | Build small, representative trace slices before generalizing a result. |

## High-signal opportunity cells to validate first

1. **P/D disaggregation × P95/P99 × bandwidth** — `C/C/C`: test whether traffic burst and topology reverse a P/D placement decision.
2. **KV reuse × quality × TTFT** — `C/A/A`: test the quality-cost boundary of exact, repaired, and trainable non-prefix reuse.
3. **Eviction/compression × quality × memory** — `C/W/W`: test whether a static retention/precision setting crosses a task/phase failure boundary.
4. **Scheduling × cache affinity × P95/P99** — `A/A/A`: test whether saved prefill is outweighed by hotspot queueing or unfairness.
5. **Offload × TTFT × bandwidth** — `C/C/C`: test fetch/recompute decisions at realistic transfer sizes and concurrency.
6. **Workflow lifecycle × memory/bandwidth/P95** — `S/S/S`: test graph/lifetime prediction under branching or tool-delay variation; this is high uncertainty rather than established novelty.

## Cells intentionally not promoted

- **Memory allocation × memory:** well studied; any Stage 3 study needs a new concrete boundary such as heterogeneous transfer granularity, not another generic paging comparison.
- **Raw throughput-only comparisons:** insufficient because Stage 2A papers use incompatible traces, SLOs, and systems; pair throughput with TTFT/TPOT/P95/P99 and quality when approximation is involved.
- **Cost and energy:** the requested axes do not contain cost/power, but `evaluation_map.md` and `candidate_gaps.md` show it is underreported. This is an evaluation extension, not evidence for a new KV algorithm.

## Audit cautions

1. `W` is not “solved.” It says only that the corpus contains repeated direct work. A new claim still needs a distinct workload, topology, constraint, or mechanism.
2. `C` often reflects incompatible experimental settings rather than a logical contradiction. The first Stage 3 value may be a matched experiment, not an invention.
3. The local corpus spans different GPUs, memory tiers, node counts, request distributions, and SLO definitions. Cross-paper speedup multiplication or ranking is not valid.

## Traceability

The markings synthesize `evaluation_map.md`, `literature_matrix.md`, `contradictions.md`, `limitation_map.md`, `candidate_gaps.md`, and `research_tensions.md`, with representative local notes for SGLang, Sarathi-Serve, DistServe, Mooncake, FlowKV, Beluga, CacheBlend, Cache-Craft, KVLink, KIVI, GEAR, PyramidKV, SCOPE, KVFlow, and Continuum.
