# Paper Metadata

- **Title:** Online Scheduling for LLM Inference with KV Cache Constraints [PAPER FACT]
- **Authors:** Patrick Jaillet, Jiashuo Jiang, Konstantina Mellou, Marco Molinaro, Chara Podimata, Zijie Zhou [PAPER FACT]
- **Venue:** arXiv:2502.07115 [cs.LG] v1 10 Feb 2025, v5 15 Jan 2026 [PAPER FACT]; 36 pages, CC BY 4.0 [PAPER FACT]
- **Code:** [NOT REPORTED] uses Vidur simulator [PAPER FACT]
- **Reading Source:** webfetch https://arxiv.org/abs/2502.07115 + https://arxiv.org/html/2502.07115v5 [PAPER FACT]

> Authenticity rule: [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED] numeric traced else [NOT REPORTED]

## 1 Problem [PAPER FACT]

Online batching and scheduling for LLM inference with KV cache memory constraints; minimize total end-to-end latency under non-preemptive execution [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLM inference token sequential generation, KV cache linear growth, memory overflow risk [PAPER FACT]
- Existing engineering batching lacks theory bounds, pathological cases [PAPER FACT]
- Operational cost 700k per day, energy water sustainability [PAPER FACT]

## 3 Bottleneck [PAPER FACT]

1. Per-token memory growth s_i+j [PAPER FACT]
2. Non-preemptive sequential dependency [PAPER FACT]
3. Millisecond decisions vs IP intractability [PAPER FACT]

## 4 Core Idea [PAPER FACT]

Hindsight optimal IP benchmark + MC-SF Memory Constrained Shortest First with overestimated predictions and future memory feasibility check [PAPER FACT]

## 5 System Changes [PAPER FACT]

- Model: discrete time, single worker M, request s_i prompt size o_i output length, batch time 1, memory sum(s_i+o(t)_i) <=M [PAPER FACT]
- IP: x_i,t start variable, objective sum t*x_{i,t}+o_i-a_i, constraint sum_{active} s_i+t-k <=M [PAPER FACT]
- MC-SF: prioritize S(t) in-progress, sort R(t) by predicted length, max prefix feasible via Eq5 check at p_j+tilde_o_j [PAPER FACT]
- Complexity O(M^2) per round [PAPER FACT]

## 6 Target Metrics [PAPER FACT]

- TEL total end-to-end latency sum(c_i-a_i), per-request latency, average latency, throughput, competitive ratio [PAPER FACT]

## 7 Baselines [PAPER FACT]

- alpha-protection greedy (alpha 0.1/0.2), alpha-protection phi-clearing, MC-Benchmark (FCFS+memory check) [PAPER FACT]
- vLLM FCFS threshold based [PAPER FACT]

## 8 Workloads [PAPER FACT]

- Synthetic: M 30-50, s 1-5, n 40-60, T 40-60, Poisson rate 0.5-1.5, 200 trials each model [PAPER FACT]
- Real: LMSYS 10k subset from 210k IPs, prompt mean 40.62 median 11, output mean 85.32 median 45, Poisson lambda 50 high 10 low [PAPER FACT]

## 9 Hardware [PAPER FACT]

- Synthetic: Gurobi solver [PAPER FACT]
- Real: Llama2-70B on 2x A100 M=16492, Vidur simulator [PAPER FACT]
- M >=2 max(s+tilde_o) required for theory [PAPER FACT]

## 10 Main Results [PAPER FACT]

- Hardness: no deterministic algorithm constant competitive, Omega(sqrt(n)) lower bound [PAPER FACT]
- Theory: MC-SF O(1)-competitive when s uniform, all arrive 0, M>=2 max(s+tilde_o), prediction within alpha [PAPER FACT]
- Synthetic Model1 avg ratio 1.005 best 1.000 worst 1.074 114/200 optimal [PAPER FACT]
- Synthetic Model2 avg 1.047 best 1.000 worst 1.227 [PAPER FACT]
- Real high demand slope 1/6 vs 1/2 benchmark, low demand 1/800 vs 1/100 [PAPER FACT]
- Prediction error epsilon 0.2/0.5/0.8 with protection 0.1 still beats FCFS [PAPER FACT]

## 11 Assumptions [PAPER FACT]

- Discrete time batch 1, single worker, non-preemptive, known s_i and predicted tilde_o_i >= o_i [PAPER FACT]
- M >=2 max(s+tilde_o), prediction within constant factor alpha [PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- Adversarial arrivals no constant ratio; need multi-worker heterogeneity, heavy-tail outliers, joint prediction design [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- Synthetic small scale (M 30-50) vs real 16492, uniform s assumption optimistic [AGENT INFERENCE]
- Single worker, homogeneous, no PD disaggregation [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. Adaptive alpha prediction? 2. Multi-GPU matching? 3. Heavy-tail hybrid model? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- Sarathi, Vidur, vLLM, Ao et al 2504.11320, Bari et al 2025, Li et al 2025a [PAPER FACT]

---
### Details Supplement [PAPER FACT]

- IP objective: min sum_i sum_t t*x_{i,t}+o_i-a_i s.t. sum_t x_{i,t}=1, sum_{active} s_i+t-k <=M [PAPER FACT]
- MC-SF sorts by tilde_o_i, checks Eq5 at p_j+tilde_o_j peaks, O(M^2) per round [PAPER FACT]
- Volume volo = s*o + o(o+1)/2, Lemma 4.4 UB 1536/M sum no vol +24 sum no*o, Lemma 4.7 LB 1/6M sum [PAPER FACT]
- Synthetic: Gurobi optimal, 200 trials, M 30-50 random, s 1-5, n 40-60 [PAPER FACT]
- Real: Vidur estimates batch time, alpha 0.21 high 0.24 low smallest feasible, 6 configs 0.3/0.25/0.2/0.1 with phi 0.2/0.1 [PAPER FACT]
- Throughput per-second tokens vs arrival tokens, MC-SF higher most intervals [PAPER FACT]
- Model: s_i prompt tokens, o_i output tokens, memory s_i+j for j-th token, peak s_i+o_i, batch 1 time unit, non-preemptive [PAPER FACT]
- Arrival: online a_i revealed at arrival, prediction tilde_o_i >= o_i, 80 percent accuracy example Zheng et al 2024 [PAPER FACT]
- Evaluation: TEL sum c_i-a_i, competitive ratio alpha vs OPT, Gurobi for hindsight [PAPER FACT]
- Real: alpha-protection 0.21 high 0.24 low, 6 configs, MC-Benchmark sorts by arrival time with future check [PAPER FACT]
