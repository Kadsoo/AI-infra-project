"""RQ-1 E1: transfer-materiality cost model + trace analysis (CPU-only).

Spec: research/stage3/RQ_01/experiment_plan.md # E1. Thresholds from LOCKED_PLAN.md.
Verdict evaluation only; verdict interpretation is done by the analyst (result.md).

H1 (frozen): at >=8k prompts, <=25 Gbps cross-node, no NVLink co-location:
  P95 transfer fraction of E2E TTFT >= 30% (P50 >= 15%); NVLink keeps P95 <= 5%.
Kill: P95 fraction < 30% at 25 Gbps across ALL plausible (f, rho, prefill-scale) points,
  OR the 25 Gbps-vs-NVLink P95 TTFT gap < 25% of the NVLink value.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BYTES_PER_TOKEN = 131072  # Llama-3.1-8B-Instruct FP16 GQA: 2*32*8*128*2 B
G25 = 25e9 / 8.0          # 25 Gbps nominal in B/s
NV_BW = 600e9             # NVLink 600 GB/s nominal
NV_CAP = 8e-3             # Splitwise non-overlapped layer-wise constant, 8 ms
A100_FLOPS = 312e12       # A100 TFLOPS
N_PARAMS = 8e9            # Llama-3.1-8B
MFU = 0.5
P95Q_FACTOR = 2.75        # M/D/1 tail: P95 queue ~ 2.5-3x mean wait; 2.75 recorded choice


def t_prefill(tokens, scale=1.0):
    return 2.0 * N_PARAMS * tokens / (A100_FLOPS * MFU) * scale


def t_transfer(bytes_, f=1.0, link=G25):
    return bytes_ / (link * f)


def t_transfer_nv(bytes_):
    return min(bytes_ / NV_BW, NV_CAP)


def md1_queue(mean_service, rho):
    if rho >= 1.0:
        return np.inf
    return rho * mean_service / (2.0 * (1.0 - rho))


def main():
    meta = json.load(open(os.path.join(BASE, "raw", "trace_meta.json")))
    rows = []
    for f in [0.2, 0.5, 1.0]:
        for rho in [0.3, 0.5, 0.7]:
            for pscale in [0.5, 1.0, 2.0]:
                for pclass, lens in [
                    ("long_only", {"p50": meta["long_len_p50"], "p95": meta["long_len_p95"]}),
                    ("mixed", {"p50": meta["all_len_p50"], "p95": meta["all_len_p95"]}),
                ]:
                    for q in ["p50", "p95"]:
                        L = lens[q]
                        kv = BYTES_PER_TOKEN * L
                        t_pre = t_prefill(L, pscale)
                        t_tr = t_transfer(kv, f)
                        t_tr_nv = t_transfer_nv(kv)
                        wq = md1_queue(t_pre, rho)
                        ttft = t_pre + P95Q_FACTOR * wq + t_tr
                        ttft_nv = t_pre + P95Q_FACTOR * wq + t_tr_nv
                        rows.append({
                            "f": f, "rho": rho, "prefill_scale": pscale, "class": pclass,
                            "quantile": q, "prompt_len": L, "kv_bytes": kv,
                            "t_prefill": t_pre, "t_transfer_25": t_tr, "t_transfer_nv": t_tr_nv,
                            "queue_mean_wait": wq, "queue_p95": P95Q_FACTOR * wq,
                            "ttft_25": ttft, "ttft_nv": ttft_nv,
                            "transfer_frac_25": t_tr / ttft,
                            "transfer_frac_nv": t_tr_nv / ttft_nv,
                            "gap_pct": (ttft - ttft_nv) / ttft_nv * 100.0,
                        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(BASE, "raw"), exist_ok=True)
    df.to_csv(os.path.join(BASE, "raw", "e1_cells.csv"), index=False)

    # corpus-anchored point: f=0.24 (FlowKV 1.0 GiB / 1.346 s on 25 Gbps nominal),
    # rho=0.5, prefill 1x, evaluated at the trace's long-class P50/P95 lengths
    F_ANCHOR = 0.24
    kv8k = BYTES_PER_TOKEN * meta["long_len_p95"]
    f_anchor = F_ANCHOR
    anchor = df[(df.f == 1.0) & (df.rho == 0.5) & (df.prefill_scale == 1.0) &
                (df["class"] == "long_only") & (df["quantile"] == "p95")].iloc[0].copy()
    t_pre = t_prefill(meta["long_len_p95"], 1.0)
    t_tr = kv8k / (G25 * f_anchor)
    t_tr_nv = t_transfer_nv(kv8k)
    wq = md1_queue(t_pre, 0.5)
    ttft = t_pre + P95Q_FACTOR * wq + t_tr
    ttft_nv = t_pre + P95Q_FACTOR * wq + t_tr_nv
    anchor_row = {
        "label": "corpus_anchor", "f_realized": f_anchor,
        "transfer_frac_25_p95": t_tr / ttft,
        "transfer_frac_25_p50": (BYTES_PER_TOKEN * meta["long_len_p50"]) / (G25 * f_anchor) /
                                (t_prefill(meta["long_len_p50"], 1.0) +
                                 P95Q_FACTOR * md1_queue(t_prefill(meta["long_len_p50"], 1.0), 0.5) +
                                 BYTES_PER_TOKEN * meta["long_len_p50"] / (G25 * f_anchor)),
        "transfer_frac_nv_p95": t_tr_nv / ttft_nv,
        "gap_pct": (ttft - ttft_nv) / ttft_nv * 100.0,
        "t_transfer_25_s": t_tr, "t_prefill_s": t_pre, "ttft_25_s": ttft,
    }
    with open(os.path.join(BASE, "raw", "e1_anchor.json"), "w") as fh:
        json.dump(anchor_row, fh, indent=2)

    # verdict evaluation (long-only 25 Gbps cells, P95):
    ll = df[(df["class"] == "long_only") & (df["quantile"] == "p95")]
    p95_fracs = ll["transfer_frac_25"]
    gaps = ll["gap_pct"]
    nv = df[(df["class"] == "long_only") & (df["quantile"] == "p95")]
    nv_frac_max = nv["transfer_frac_nv"].max()
    verdict = {
        "all_cells_p95_frac_25_ge_30": bool((p95_fracs >= 0.30).all()),
        "p95_frac_25_min": float(p95_fracs.min()),
        "p95_frac_25_max": float(p95_fracs.max()),
        "gap_pct_min": float(gaps.min()),
        "gap_pct_max": float(gaps.max()),
        "gap_ge_25_all": bool((gaps >= 25.0).all()),
        "nv_p95_frac_max": float(nv_frac_max),
        "anchored_frac_p95": float(anchor_row["transfer_frac_25_p95"]),
        "anchored_frac_p50": float(anchor_row["transfer_frac_25_p50"]),
        "anchored_nv_frac_p95": float(anchor_row["transfer_frac_nv_p95"]),
        "anchored_gap_pct": float(anchor_row["gap_pct"]),
    }
    with open(os.path.join(BASE, "raw", "e1_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    print("anchor:", json.dumps(anchor_row, indent=2))


if __name__ == "__main__":
    main()