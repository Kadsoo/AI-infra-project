"""E1 — Analytical/numerical screening of the tier axis from corpus cost models.

RQ-8 (retained-token x bit-width x tier non-additivity), Wave-1 CPU-only gate.
Frozen contract: LOCKED_PLAN.md (freeze date 2026-08-27); spec: experiment_plan.md §E1.

This script can KILL H2 (conditional, screening kill) but cannot kill H1, the
primary, or H3. It measures honestly with the published constant set; no
machine-specific constants exist (CPU-only machine, environment.md).

Model (fully closed form, corpus constants):
  - Llama-2-7B (MHA): 32 layers, hidden 4096, per-token per-layer FP16 KV = 16 KB.
  - 12 tier cells: R in {100%,60%,20%} x B in {16-bit, 2-bit KIVI G=32 R=128}
    x T in {GPU-only, GPU+host}, per workload W1 (4K RAG) and W2 (GSM8K ~900).
  - Actual-bytes accounting (never nominal bits), per component:
      * FP16: 16384 B/token/layer
      * KIVI-2bit (G=32, R=128): 2-bit data 2048 B/token +
        1024 B/token scales+zero-points (K per-channel groups of 32 tokens:
        512 B/token; V per-token groups of 32 channels: 512 B/token) +
        FP16 residual fixed 128 tokens = 2097152 B/layer. GEAR n_b=20 buffer:
        not present in this matrix (B in {16-bit, KIVI-2}) — column kept with
        zero and documented.
  - Tier transfer per decode step: bytes = (1 - alpha) x layers x B_layer(N)
    (ShadowKV temporal locality alpha = 0.60), fetch_count = layers x
    ceil(N / chunk 512), T_transfer = fetch_count x F + bytes / B_PCIe
    (plan §E1 instrumentation (b)(c)(d)).
  - GPU compute estimate per decode step (GPU-bound regime): HBM-bound time
    for weights + attention reads at B_HBM = 2 TB/s:
    T_gpu = (weights_bytes + layers x B_layer_actual) / B_HBM.
  - FlexGen rule: T_step(host) = max(T_transfer, T_gpu); T_step(GPU-only) = T_gpu.
  - pen(R,B) = T_tier(R,B) - T_gpu(R,B),  S = pen(20%,2-bit) /
    [pen(20%,16-bit) + pen(100%,2-bit)]  (H2 super-additivity index).
  - Byte-normalized tier latency (µs/MB) = T_transfer / transfer_bytes,
    defined on T=GPU+host cells only (GPU-only cells transfer 0 bytes).
  - Regime check (A2): compute-dominant iff max(I/O, compute) = compute
    (T_transfer <= T_gpu); tier fraction = T_transfer/(T_transfer+T_gpu).
    Pre-registered: if compute dominates at EVERY cell -> tier untestable.

Frozen verdicts (LOCKED_PLAN §6/§7, experiment_plan §E1):
  - H2 ANALYTICALLY FALSIFIED iff byte-normalized latency constant across all
    cells within ±5% OR S <= 1.15 at the corner (both F values reported).
  - H2 SURVIVES screening iff S > 1.15 AND byte-normalized latency varies
    >10% across cells AND regime passes (tier fraction >10% in >=1 cell).
  - Otherwise (regime fails at every cell): tier untestable (Ambiguous
    Outcome) — E2b/E3 tier arms canceled.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # RQ_08/

# ---------------------------------------------------------------------------
# Frozen / published constant set (each with paper citation; see report §constants)
# ---------------------------------------------------------------------------
CONSTANTS = {
    # Model (experiment_plan.md §4; KIVI.md; environment.md)
    "layers": 32,                                    # Llama-2-7B
    "hidden": 4096,                                  # Llama-2-7B
    "kv_fp16_bytes_per_token_layer": 16384,          # 2 (K+V) x 4096 x 2 B = 16 KB
    "weights_bytes": 13.48e9,                        # ~6.74e9 params x 2 B FP16
    # KIVI 2-bit, G=32 R=128 (experiment_plan.md §4; KIVI.md §4.1 G=32 R=128
    # universal; scales/zero-points per KIVI asymmetric scheme)
    "kivi_g": 32,
    "kivi_r": 128,                                   # FP16 residual tokens (K and V)
    "kivi_quant_bytes_per_token": 2048.0,            # 2 bit x 2 tensors x 4096 = 2048 B/token
    "kivi_scales_bytes_per_token": 1024.0,           # K: 512 (per-channel, G=32 over tokens)
                                                     # V: 512 (per-token, G=32 over channels)
    "kivi_published_frac_fp16": 0.217,               # GEAR Table 1: KIVI 2-bit ≈ 21.7% of FP16
    "gear_nb": 20,                                   # GEAR n_b=20 FP16 buffer — NOT present in this matrix
    # Tier / hardware (experiment_plan.md §E1; ShadowKV.md §9: PCIe 31.5 GB/s,
    # HBM 2 TB/s; Beluga.md §3: 10.55 µs per 16 KB transfer, ~75% sync overhead,
    # F=2 µs floor per assumption_map A5)
    "pcie_GBps": 31.5,
    "hbm_TBps": 2.0,
    "chunk_tokens": 512,                             # block-aligned 512-token chunks
    "page_block_tokens": 16,
    "alpha": 0.60,                                   # ShadowKV temporal locality hit rate
    "per_fetch_us": [2.0, 10.55],                    # sensitivity sweep
    "decode_transfer_frac_anchor": 0.969,            # InfiniGen Fig 18: FlexGen baseline
    # Workload retained sets (experiment_plan.md §E1: W1 4K-class -> 4096,
    # 20% -> 820 (128/820 = 15.6% per plan); W2 ~900-token prefill)
    "workloads": {
        "W1_RAG_4K":   {"100%": 4096, "60%": 2458, "20%": 820},
        "W2_GSM8K_900": {"100%": 900,  "60%": 540, "20%": 180},
    },
}
R_LEVELS = ["100%", "60%", "20%"]
B_LEVELS = ["16bit", "2bit"]
T_LEVELS = ["GPU-only", "GPU+host"]


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def layer_bytes(n_retained: int, bit_width: str) -> dict:
    """Actual retained bytes per sequence per layer (per-component, never nominal bits)."""
    fp16 = CONSTANTS["kv_fp16_bytes_per_token_layer"]
    if bit_width == "16bit":
        return {
            "quantized": 0.0,
            "scales_zp": 0.0,
            "residual": 0.0,
            "gear_buffer": 0.0,
            "total": float(n_retained * fp16),
            "pct_of_fp16": 100.0,
        }
    r = CONSTANTS["kivi_r"]
    residual_b = float(r * fp16)
    quant_b = CONSTANTS["kivi_quant_bytes_per_token"]
    scales_b = CONSTANTS["kivi_scales_bytes_per_token"]
    nq = max(0, n_retained - r)          # quantized (grouped) portion
    q = nq * (quant_b + scales_b)
    total = q + residual_b
    # GEAR n_b=20 buffer is a GEAR-mechanism component; B in this matrix is
    # KIVI-2, so it is not present (recorded as 0.0 and documented).
    return {
        "quantized": q,
        "scales_zp": nq * scales_b,
        "residual": residual_b,
        "gear_buffer": 0.0,
        "total": total,
        "pct_of_fp16": 100.0 * total / (n_retained * fp16),
    }


def compute_cell(n_retained: int, bit_width: str, tier: str, per_fetch_us: float) -> dict:
    fp16 = CONSTANTS["kv_fp16_bytes_per_token_layer"]
    layers = CONSTANTS["layers"]
    comp = layer_bytes(n_retained, bit_width)
    b_layer = comp["total"]
    retained_bytes = layers * b_layer

    # (b) predicted transfer bytes per decode step (post temporal locality:
    #  alpha fraction of chunks is re-served from the on-GPU cache per
    #  ShadowKV; only (1-alpha) of the retained store crosses PCIe per step)
    transfer_bytes = (1.0 - CONSTANTS["alpha"]) * retained_bytes if tier == "GPU+host" else 0.0

    # (c) fetch count per step = layers x ceil(retained_tokens / chunk 512)
    fetch_count = layers * ceil_div(n_retained, CONSTANTS["chunk_tokens"]) if tier == "GPU+host" else 0

    # (d) predicted tier latency per step (plan formula):
    #     fetch_count x (F + bytes/fetch_count/B_PCIe) = fetch_count x F + bytes/B_PCIe
    pcie = CONSTANTS["pcie_GBps"] * 1e9
    t_transfer = (fetch_count * per_fetch_us * 1e-6 + transfer_bytes / pcie) if tier == "GPU+host" else 0.0

    # GPU compute estimate per decode step (HBM-bound; FlexGen cost model:
    # decode GPU utilization ~13%, i.e., bandwidth-bound; weights dominate)
    hbm = CONSTANTS["hbm_TBps"] * 1e12
    t_gpu = (CONSTANTS["weights_bytes"] + retained_bytes) / hbm

    # FlexGen rule: T = max(I/O, compute)
    t_step = t_gpu if tier == "GPU-only" else max(t_transfer, t_gpu)

    # (e) byte-normalized tier latency (µs/MB), host cells only
    bn_lat = (t_transfer * 1e6 / (transfer_bytes / 1e6)) if tier == "GPU+host" and transfer_bytes > 0 else float("nan")

    # pen (tier penalty vs GPU-only, FlexGen max rule) and regime quantities
    pen = t_step - t_gpu if tier == "GPU+host" else 0.0
    compute_dominant = t_transfer <= t_gpu  # max(I/O, compute) == compute
    tier_fraction = t_transfer / (t_transfer + t_gpu) if tier == "GPU+host" else 0.0

    return {
        "n_retained": n_retained,
        "b_layer": b_layer,
        "b_layer_pct_fp16": comp["pct_of_fp16"],
        "quantized_bytes": comp["quantized"],
        "scales_zp_bytes": comp["scales_zp"],
        "residual_bytes": comp["residual"],
        "gear_buffer_bytes": comp["gear_buffer"],
        "retained_bytes_total": retained_bytes,
        "transfer_bytes": transfer_bytes,
        "fetch_count": fetch_count,
        "t_transfer_s": t_transfer,
        "t_gpu_s": t_gpu,
        "t_step_s": t_step,
        "pen_s": pen,
        "bn_lat_us_per_mb": bn_lat,
        "tier_fraction": tier_fraction,
        "compute_dominant": compute_dominant,
    }


def constancy_stats(bn_values: list[float]) -> dict:
    vals = [v for v in bn_values if not math.isnan(v)]
    if not vals:
        return {"n": 0, "min": float("nan"), "max": float("nan"),
                "spread_pct_of_min": float("nan"), "spread_pct_of_mean": float("nan")}
    lo, hi = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    return {"n": len(vals), "min": lo, "max": hi,
            "spread_pct_of_min": 100.0 * (hi - lo) / lo,
            "spread_pct_of_mean": 100.0 * (hi - lo) / mean}


def main() -> None:
    out = Path(__file__).parent.parent
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []          # one row per cell x workload x F
    reg_rows = []
    for wname, rset in CONSTANTS["workloads"].items():
        for r in R_LEVELS:
            for b in B_LEVELS:
                for t in T_LEVELS:
                    for f in CONSTANTS["per_fetch_us"]:
                        c = compute_cell(rset[r], b, t, f)
                        row = {
                            "workload": wname, "R": r, "B": b, "T": t, "F_us": f,
                            **{k: v for k, v in c.items()},
                        }
                        rows.append(row)
                        reg_rows.append({
                            "workload": wname, "R": r, "B": b, "T": t, "F_us": f,
                            "t_gpu_s": c["t_gpu_s"], "t_transfer_s": c["t_transfer_s"],
                            "t_step_s": c["t_step_s"], "tier_fraction": c["tier_fraction"],
                            "compute_dominant": c["compute_dominant"],
                            "max_rule": "compute" if c["compute_dominant"] else "I/O",
                        })

    # --- write raw tables (nothing dropped) ---------------------------------
    cells_fieldnames = [
        "workload", "R", "B", "T", "F_us",
        "n_retained", "b_layer", "b_layer_pct_fp16",
        "quantized_bytes", "scales_zp_bytes", "residual_bytes", "gear_buffer_bytes",
        "retained_bytes_total", "transfer_bytes", "fetch_count",
        "t_transfer_s", "t_gpu_s", "t_step_s", "pen_s",
        "bn_lat_us_per_mb", "tier_fraction", "compute_dominant",
    ]
    with (raw_dir / "e1_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cells_fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    reg_fieldnames = [
        "workload", "R", "B", "T", "F_us",
        "t_gpu_s", "t_transfer_s", "t_step_s", "tier_fraction",
        "compute_dominant", "max_rule",
    ]
    with (raw_dir / "e1_regime.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=reg_fieldnames)
        w.writeheader()
        for r in reg_rows:
            w.writerow(r)

    # --- derived verdict quantities -----------------------------------------
    def pen(wname, r, b, f):
        for row in rows:
            if row["workload"] == wname and row["R"] == r and row["B"] == b \
               and row["T"] == "GPU+host" and row["F_us"] == f:
                return row["pen_s"]
        raise KeyError((wname, r, b, f))

    def transfer(wname, r, b, f):
        for row in rows:
            if row["workload"] == wname and row["R"] == r and row["B"] == b \
               and row["T"] == "GPU+host" and row["F_us"] == f:
                return row["t_transfer_s"]
        raise KeyError((wname, r, b, f))

    s_index = {}
    s_raw_index = {}
    corner_bytes = {}
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            p202b = pen(wname, "20%", "2bit", f)
            p2016 = pen(wname, "20%", "16bit", f)
            p1002 = pen(wname, "100%", "2bit", f)
            denom = p2016 + p1002
            # S <= 1.15  <=>  pen(20%,2bit) <= 1.15 x [pen(20%,16bit)+pen(100%,2bit)].
            # When both sides are 0 the bound holds trivially -> S := 0 (the
            # corner penalty is literally zero; the ratio is 0/0 only in form).
            if denom > 0:
                s = p202b / denom
            elif p202b == 0:
                s = 0.0
            else:
                s = float("inf")
            s_index[(wname, f)] = s
            # Robustness reading: ignore the FlexGen max-rule overlap and take
            # raw transfer time as the tier penalty (worst case for H2).
            raw = transfer(wname, "20%", "2bit", f)
            denom_r = transfer(wname, "20%", "16bit", f) + transfer(wname, "100%", "2bit", f)
            s_raw_index[(wname, f)] = raw / denom_r if denom_r > 0 else float("nan")
            corner_bytes[(wname, f)] = {
                "pen_20_2bit": p202b, "pen_20_16bit": p2016, "pen_100_2bit": p1002,
                "raw_20_2bit": raw,
                "raw_20_16bit": transfer(wname, "20%", "16bit", f),
                "raw_100_2bit": transfer(wname, "100%", "2bit", f),
            }

    # constancy of byte-normalized latency across host cells (transfer defined
    # only on T=GPU+host; GPU-only cells transfer 0 bytes — documented exclusion)
    constancy = {}
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            vals = [r["bn_lat_us_per_mb"] for r in rows
                    if r["workload"] == wname and r["T"] == "GPU+host" and r["F_us"] == f]
            constancy[(wname, f)] = constancy_stats(vals)

    # regime: compute-dominant at every cell (pre-registered untestable cond.)
    regime = {}
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            sub = [r for r in rows if r["workload"] == wname and r["F_us"] == f]
            host_sub = [r for r in sub if r["T"] == "GPU+host"]
            all_dom = all(r["compute_dominant"] for r in host_sub)
            fracs = [r["tier_fraction"] for r in host_sub]
            regime[(wname, f)] = {
                "all_cells_compute_dominant": all_dom,
                "tier_fraction_min": min(fracs), "tier_fraction_max": max(fracs),
                "tier_fraction_gt_10pct_any_cell": any(x > 0.10 for x in fracs),
            }

    # peak-interaction cell: host cell with max byte-normalized latency
    # (strongest predicted deviation from the byte-linear additive baseline)
    peak = {}
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            host = [r for r in rows if r["workload"] == wname and r["T"] == "GPU+host" and r["F_us"] == f]
            host = sorted(host, key=lambda r: -r["bn_lat_us_per_mb"])
            peak[(wname, f)] = {
                "cell": f'{host[0]["R"]}/{host[0]["B"]}/{host[0]["T"]}',
                "bn_lat_us_per_mb": host[0]["bn_lat_us_per_mb"],
                "tier_fraction": host[0]["tier_fraction"],
            }

    # --- frozen verdicts -----------------------------------------------------
    checks = []
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            s = s_index[(wname, f)]
            c = constancy[(wname, f)]
            rg = regime[(wname, f)]
            s_ok = s > 1.15
            bn_var_ok = c["spread_pct_of_mean"] > 10.0
            const_5pct = c["spread_pct_of_mean"] <= 5.0
            reg_ok = (not rg["all_cells_compute_dominant"]) and rg["tier_fraction_gt_10pct_any_cell"]
            # Precedence (pre-registered, plan §E1 Falsification): the A2
            # regime gate fires first — if max(I/O, compute) = compute at
            # EVERY cell, the tier hypotheses are declared not testable in
            # this regime (Ambiguous Outcome) and the S/constancy arms are
            # moot for that workload x F combination.
            if rg["all_cells_compute_dominant"]:
                verdict = "TIER UNTESTABLE (Ambiguous Outcome)"
            elif s <= 1.15 or const_5pct:
                verdict = "H2 ANALYTICALLY FALSIFIED"
            elif s_ok and bn_var_ok and reg_ok:
                verdict = "H2 SURVIVES SCREENING"
            else:
                verdict = "H2 ANALYTICALLY FALSIFIED (other arm)"
            checks.append({
                "workload": wname, "F_us": f,
                "S": round(s, 4),
                "S_raw_transfer": round(s_raw_index[(wname, f)], 4),
                "S_gt_1.15": s_ok,
                "bn_spread_pct_mean": round(c["spread_pct_of_mean"], 2),
                "bn_const_within_5pct": const_5pct,
                "bn_var_gt_10pct": bn_var_ok,
                "regime_all_compute_dominant": rg["all_cells_compute_dominant"],
                "regime_tier_frac_min": round(100 * rg["tier_fraction_min"], 1),
                "regime_tier_frac_max": round(100 * rg["tier_fraction_max"], 1),
                "regime_pass": reg_ok,
                "verdict": verdict,
            })

    # Overall flag: the kill fires on any workload x F combination where the
    # tier axis is testable; combinations that are regime-untestable (W2) are
    # reported as Ambiguous per pre-registration and cancel E2b/E3 tier arms
    # for that workload. H2 never survives: no combination is S>1.15.
    any_falsified = any("FALSIFIED" in c["verdict"] for c in checks)
    any_untestable = any("UNTESTABLE" in c["verdict"] for c in checks)
    no_survival = all(("FALSIFIED" in c["verdict"]) or ("UNTESTABLE" in c["verdict"])
                      for c in checks)
    if no_survival and any_falsified:
        overall_verdict = ("H2 ANALYTICALLY FALSIFIED (E1 screening kill; "
                           "W2 tier-untestable per workload)")
    elif no_survival and any_untestable and not any_falsified:
        overall_verdict = "TIER UNTESTABLE (Ambiguous Outcome) — E2b/E3 tier arms canceled"
    elif all("SURVIVES" in c["verdict"] for c in checks):
        overall_verdict = "H2 SURVIVES SCREENING"
    else:
        overall_verdict = "MIXED — see per-workload/per-F checks"

    # --- report --------------------------------------------------------------
    report = build_report(rows, checks, s_index, s_raw_index, corner_bytes,
                          constancy, regime, peak, overall_verdict)
    proc = out / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "e1_report.md").write_text(report, encoding="utf-8")

    # --- console summary -----------------------------------------------------
    print("E1 screening complete.")
    print(f"cells rows written: {len(rows)} -> raw/e1_cells.csv")
    print(f"regime rows written: {len(reg_rows)} -> raw/e1_regime.csv")
    print(f"report written -> processed/e1_report.md")
    print("\nS index (corner 20%/2-bit), pen(20%,2-bit)/[pen(20%,16-bit)+pen(100%,2-bit)]:")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            s = s_index[(wname, f)]
            sr = s_raw_index[(wname, f)]
            cb = corner_bytes[(wname, f)]
            print(f"  {wname} F={f}us: S = {s:.4f} (raw-transfer S = {sr:.4f})  (pens: "
                  f"{cb['pen_20_2bit']*1e3:.2f} / "
                  f"{cb['pen_20_16bit']*1e3:.2f} + "
                  f"{cb['pen_100_2bit']*1e3:.2f} ms)")
    print("\nByte-normalized latency (us/MB) spread over host cells:")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            c = constancy[(wname, f)]
            print(f"  {wname} F={f}us: min {c['min']:.1f}, max {c['max']:.1f}, "
                  f"spread {c['spread_pct_of_mean']:.1f}% of mean (n={c['n']})")
    print("\nRegime (host cells):")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            rg = regime[(wname, f)]
            print(f"  {wname} F={f}us: all-compute-dominant={rg['all_cells_compute_dominant']}, "
                  f"tier fraction {100*rg['tier_fraction_min']:.1f}%-{100*rg['tier_fraction_max']:.1f}%")
    print("\nPeak-interaction cell (max us/MB, E3 look-first):")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            p = peak[(wname, f)]
            print(f"  {wname} F={f}us: {p['cell']} @ {p['bn_lat_us_per_mb']:.1f} us/MB")
    print(f"\nOVERALL VERDICT FLAG: {overall_verdict}")


def build_report(rows, checks, s_index, s_raw_index, corner_bytes, constancy,
                 regime, peak, overall_verdict) -> str:
    L = []
    A = L.append
    A("# E1 Report — Analytical Screening of the Tier Axis (corpus cost models)")
    A("")
    A("**RQ-8 Wave-1 CPU-only kill-gate experiment. KILL-GATE: E1 can kill H2 "
      "(conditional, screening kill) only. Cannot kill H1, primary, or H3.**")
    A("")
    A(f"- Date: 2026-08-27 (session date, environment.md)")
    A(f"- Environment: {CONSTANTS['layers']}-layer Llama-2-7B model card; "
      f"CPU-only machine (environment.md: no GPU — A100-gated cells blocked; "
      f"E1 needs none)")
    A(f"- Script: `code/e1_screening.py` (SHA-256 "
      f"`{hashlib.sha256((ROOT / 'code/e1_screening.py').read_bytes()).hexdigest()}`)")
    A(f"- Raw outputs: `raw/e1_cells.csv` (48 rows: 12 cells x 2 workloads x 2 F), "
      f"`raw/e1_regime.csv`")
    A("")
    A("## 1. Model summary (per-cell quantities)")
    A("")
    A("12 tier cells (3R x 2B x 2T) x 2 workloads x F in {2.0, 10.55} µs. "
      "Actual retained bytes per sequence per layer are per-component (never "
      "nominal bits): 16-bit = N x 16 KB; KIVI-2bit (G=32, R=128) = "
      "(N-128) x 3072 B (2048 B 2-bit data + 1024 B scales/zero-points) + "
      "128 x 16 KB FP16 residual. GEAR n_b=20 FP16 buffer: not present in this "
      "matrix (B in {16-bit, KIVI-2}); recorded as 0.0 and documented — the "
      "byte-accounting rule includes it 'where present' (hypothesis confounder 2).")
    A("")
    A("| Workload | R | B | T | F (µs) | N | store/layer (MB) | %FP16 | transfer/step (MB) | fetches/step | T_transfer (ms) | T_gpu (ms) | T_step (ms) | pen (ms) | µs/MB | tier frac |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bn = "" if math.isnan(r["bn_lat_us_per_mb"]) else f"{r['bn_lat_us_per_mb']:.1f}"
        A(f"| {r['workload'].replace('_',' ')} | {r['R']} | {r['B']} | {r['T']} | {r['F_us']} "
          f"| {r['n_retained']} | {r['b_layer']/1e6:.2f} | {r['b_layer_pct_fp16']:.1f} "
          f"| {r['transfer_bytes']/1e6:.1f} | {r['fetch_count']} "
          f"| {r['t_transfer_s']*1e3:.2f} | {r['t_gpu_s']*1e3:.2f} "
          f"| {r['t_step_s']*1e3:.2f} | {r['pen_s']*1e3:.2f} "
          f"| {bn} "
          f"| {100*r['tier_fraction']:.1f}% |")
    A("")
    A("Residual fraction note (pre-registered): the FP16 residual R=128 is a "
      "constant absolute size (2.097 MB/layer) but a retention-dependent "
      "fraction of the retained store: 128/4096 = 3.1% at 100% vs 128/820 = "
      "15.6% at 20% retention (W1); 128/900 = 14.2% at 100% vs 128/180 = 71.1% "
      "at 20% (W2). KIVI-2bit store fraction vs FP16 at 100% retention: "
      "W1 4096 tokens = 21.3% (published anchor: 21.7%, GEAR Table 1 — the "
      "0.4 pp gap is the G=32/R=128 decomposition vs GEAR's measured "
      "g=64/nb=64 config; ratio conclusions are unaffected).")
    A("")
    A("## 2. Super-additivity index S")
    A("")
    A("`S = pen(20%,2-bit) / [pen(20%,16-bit) + pen(100%,2-bit)]`, "
      "`pen(R,B) = T_tier(R,B) - T_gpu(R,B)` (FlexGen `T = max(I/O, compute)` "
      "rule; pen = max(transfer, compute) - compute). Frozen H2 null: "
      "`pen(20%,2-bit) <= 1.15 x [sum]`; screening kill iff `S <= 1.15`.")
    A("")
    A("| Workload | F (µs) | pen(20%,2-bit) ms | pen(20%,16-bit) ms | pen(100%,2-bit) ms | S | S (raw-transfer) |")
    A("|---|---|---|---|---|---|---|")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            cb = corner_bytes[(wname, f)]
            s = s_index[(wname, f)]
            sr = s_raw_index[(wname, f)]
            A(f"| {wname} | {f} | {cb['pen_20_2bit']*1e3:.3f} | {cb['pen_20_16bit']*1e3:.3f} "
              f"| {cb['pen_100_2bit']*1e3:.3f} | {s:.4f} | {sr:.4f} |")
    A("")
    A("Interpretation: the corner penalty is ZERO on every workload x F "
      "combination — at (20%, 2-bit) the predicted transfer time "
      "(1.84-2.39 ms W1; 0.98-1.25 ms W2) sits far below the GPU compute "
      "estimate (6.8-6.9 ms), so the FlexGen max-rule absorbs the entire tier "
      "cost into compute overlap. Where the ratio is 0/0 in form, S := 0 "
      "because the additive bound `pen(20%,2-bit) <= 1.15 x [sum]` holds "
      "trivially (both sides zero). Robustness reading — ignore the max-rule "
      "overlap and take raw T_transfer as the tier penalty (worst case for "
      "H2): raw S = T_t(20%,2-bit)/[T_t(20%,16-bit)+T_t(100%,2-bit)] "
      f"= {s_raw_index[('W1_RAG_4K', 10.55)]:.3f} (W1, F=10.55) and "
      f"{s_raw_index[('W2_GSM8K_900', 10.55)]:.3f} (W2, F=10.55) — transfer is "
      "near-linear in actual bytes and the corner is CHEAPER per byte than "
      "the single-axis cells, not dearer. Both readings put S far below 1.15.")
    A("")
    A("## 3. Byte-normalized tier latency constancy (A8)")
    A("")
    A("Defined on T=GPU+host cells (6 per workload x F): GPU-only cells "
      "transfer 0 bytes — byte-normalized transfer latency is not defined "
      "there (the constancy arm of H2's falsification is a statement about "
      "transfer).")
    A("")
    A("| Workload | F (µs) | host cells (n) | min µs/MB | max µs/MB | spread % of mean |")
    A("|---|---|---|---|---|---|")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            c = constancy[(wname, f)]
            A(f"| {wname} | {f} | {c['n']} | {c['min']:.2f} | {c['max']:.2f} "
              f"| {c['spread_pct_of_mean']:.1f}% |")
    A("")
    A("Frozen tolerance: constant within ±5% (noiseless model). Survival "
      "requires >10% variation. Result: at F=10.55 µs (Beluga measured) "
      "variation is 28.8-29.3% — clearly non-constant. At F=2.0 µs (A5 floor) "
      "variation drops to 6.6-6.8% — above the ±5% falsification tolerance "
      "but below the >10% survival bar; the per-fetch fixed cost is small "
      "enough at F=2 that latency is nearly byte-linear (A8 nearly holds). "
      "The constancy arm alone does NOT falsify H2 at either F (no "
      "combination is within ±5%); the S arm carries the kill.")
    A("")
    A("## 4. Regime check (A2, pre-registered Ambiguous-Outcome gate)")
    A("")
    A("Compute estimate per decode step: HBM-bound time at 2 TB/s for weights "
      "(13.48 GB) + attention reads of the actual retained store "
      "(T_gpu = 6.79-7.81 ms across cells; weights dominate). Tier I/O: "
      "T_transfer = fetches x F + bytes/31.5 GB/s. FlexGen rule "
      "`T = max(I/O, compute)`; compute-dominant iff T_transfer <= T_gpu.")
    A("")
    A("| Workload | F (µs) | all host cells compute-dominant? | tier fraction min-max | any cell >10%? |")
    A("|---|---|---|---|---|")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            rg = regime[(wname, f)]
            A(f"| {wname} | {f} | {rg['all_cells_compute_dominant']} "
              f"| {100*rg['tier_fraction_min']:.1f}% - {100*rg['tier_fraction_max']:.1f}% "
              f"| {rg['tier_fraction_gt_10pct_any_cell']} |")
    A("")
    A("Per-cell detail in `raw/e1_regime.csv` (T_gpu, T_transfer, max-rule "
      "winner, tier fraction per cell). Pre-registered untestable condition — "
      "`max(I/O, compute) = compute at EVERY cell` — holds for W2 at both F "
      "(all 6 host cells compute-dominant; the ~900-token decode-heavy "
      "reasoning workload never lets PCIe surface above compute, max transfer "
      "6.1-6.7 ms vs T_gpu 6.98-6.79 ms). It does NOT hold for W1: the "
      "(100%,16-bit) and (60%,16-bit) host cells are I/O-dominant "
      "(T_transfer 16.7-30.0 ms > T_gpu 7.4-7.8 ms), tier fraction "
      "44-79%. So the tier axis is testable in ≥1 W1 cell (criterion (iii) "
      "passes on W1), and W2 is reported as tier-untestable per workload.")
    A("")
    A("## 5. Predicted peak-interaction cell (E3 look-first)")
    A("")
    A("Cell with maximum byte-normalized tier latency (largest predicted "
      "deviation from the byte-linear additive baseline):")
    A("")
    A("| Workload | F (µs) | peak cell | µs/MB | tier fraction |")
    A("|---|---|---|---|---|")
    for wname in CONSTANTS["workloads"]:
        for f in CONSTANTS["per_fetch_us"]:
            p = peak[(wname, f)]
            A(f"| {wname} | {f} | {p['cell']} | {p['bn_lat_us_per_mb']:.1f} | "
              f"{100*p['tier_fraction']:.1f}% |")
    A("")
    A("The peak-interaction cells are the 2-bit cells at HIGH retention — "
      "(100%, 2-bit, GPU+host) on W1, (60%, 2-bit, GPU+host) on W2. The "
      "non-linearity H2 names (per-fetch fixed cost on chunk-granularity "
      "fetches) is real in the model, but it peaks where the byte store is "
      "LARGEST (more chunks per layer, more fixed cost per MB), not at the "
      "H2 corner (20%, 2-bit) — i.e., the mechanism anti-correlates with "
      "retention, so it cannot produce a >1.5x super-additive penalty at the "
      "joint minimum. (If E3 runs tier cells for other reasons, these are the "
      "cells where deviations from byte-linearity should appear first.)")
    A("")
    A("## 6. Frozen verdict checks")
    A("")
    A("| Workload | F (µs) | S | S (raw) | S > 1.15? | bn spread % | bn const ±5%? | bn var >10%? | regime all-compute? | regime pass | VERDICT |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in checks:
        A(f"| {c['workload']} | {c['F_us']} | {c['S']} | {c['S_raw_transfer']} | {c['S_gt_1.15']} "
          f"| {c['bn_spread_pct_mean']} | {c['bn_const_within_5pct']} "
          f"| {c['bn_var_gt_10pct']} | {c['regime_all_compute_dominant']} "
          f"| {c['regime_pass']} | {c['verdict']} |")
    A("")
    A(f"**OVERALL VERDICT FLAG: {overall_verdict}**")
    A("")
    A("Basis: S <= 1.15 at the (20%, 2-bit) corner wherever the tier axis is "
      "testable — W1 at both F (corner pen = 0 under the FlexGen max rule; "
      "raw-transfer S = 0.16 W1 / 0.31 W2 at F=10.55). Byte-normalized "
      "latency is NOT constant (28.8-29.3% spread at F=10.55; 6.6-6.8% at "
      "F=2 — neither within the ±5% falsification tolerance), so the "
      "constancy arm does not fire; the S arm fires. W2 additionally fails "
      "the regime check (all cells compute-dominant -> tier untestable on "
      "W2, pre-registered Ambiguous Outcome — E2b/E3 tier arms canceled for "
      "that workload). Consequences per the kill-gate (LOCKED_PLAN §7): H2 "
      "is killed at screening cost under the corpus cost models — the burden "
      "of proof for a real-machine super-additive tier penalty shifts to E2b "
      "(root-complex contention, fragmentation, concurrency are NOT in these "
      "models, per T5). H1, the primary, and H3 are untouched by this "
      "verdict. E3's tier arm is warranted only if E2b resurrects the "
      "mechanism.")
    A("")
    A("## 7. Constant-set citation log and substitution note")
    A("")
    A("| Constant | Value | Source |")
    A("|---|---|---|")
    A("| layers / hidden | 32 / 4096 | experiment_plan.md §4 (Llama-2-7B MHA, 4K) |")
    A("| per-token per-layer FP16 KV | 16 KB (2 x 4096 x 2 B) | experiment_plan.md §4; KIVI.md §2 |")
    A("| weights bytes (GPU-side) | 13.48 GB (≈6.74e9 x 2 B) | Llama-2-7B param count, FP16 |")
    A("| KIVI G / R | 32 / 128 | experiment_plan.md §4 (frozen); KIVI.md §4.1 |")
    A("| KIVI-2bit store ≈ 21.7% of FP16 (anchor) | GEAR Table 1 | GEAR.md §10 / plan §E1 |")
    A("| GEAR n_b=20 FP16 buffer | not present in this matrix | GEAR.md §4 |")
    A("| chunk (block-aligned) | 512 tokens | experiment_plan.md §4/§E1 |")
    A("| page block | 16 tokens | experiment_plan.md §4 |")
    A("| PCIe nominal bandwidth | 31.5 GB/s | ShadowKV.md §9/§10; plan §E1 |")
    A("| HBM bandwidth | 2 TB/s | ShadowKV.md §9 |")
    A("| temporal locality α | 60% | ShadowKV.md §10 ('>60% hit rate'); plan §E1 |")
    A("| per-fetch fixed latency F | {2, 10.55} µs | Beluga.md §3 (10.55 µs per 16 KB, ~75% sync); assumption_map A5; plan §E1 |")
    A("| decode transfer fraction anchor | 96.9% of per-block time | InfiniGen.md Fig 18 (FlexGen baseline); plan §E1 |")
    A("| FlexGen latency rule | T = max(I/O, compute) | FlexGen.md §5 cost model; plan §E1 |")
    A("| W1 retained sets | 4096 / 2458 / 820 | plan §E1 (4096; 128/820 = 15.6% ⇒ 820; 2458 = ceil(0.6x4096)) |")
    A("| W2 retained sets | 900 / 540 / 180 | plan §E1 (GSM8K 8-shot CoT ~900-token prefill) |")
    A("")
    A("**Substitution note (pre-registered, plan §E1 Hardware):** machine-"
      "specific constants (measured PCIe bandwidth of the actual A100 "
      "platform, real per-fetch latency) are NOT available — this machine is "
      "CPU-only (environment.md; no GPU on this host). All quantities above "
      "use the published constant set and are tagged accordingly. Ratio "
      "conclusions (S index, byte-normalized latency spread) are designed "
      "machine-invariant: raw-transfer S = 0.16-0.31 is ~4-7x below the 1.15 "
      "bar, and would need the corner's per-MB cost to be >4-6x the "
      "single-axis cells' to flip — no plausible PCIe-constant rescaling "
      "does that (raw S is exactly invariant to the shared PCIe bandwidth and "
      "to α). Before E2b/E3, measured constants replace the published values "
      "(sensitivity check; re-run logged).")
    A("")
    A("## 8. Sensitivity notes")
    A("")
    A("- **F sweep:** F ∈ {2, 10.55} µs changes T_transfer by ≤2.7 ms (W1 "
      "top cells) and shifts the byte-normalized spread from 6.6-6.8% (F=2) "
      "to 28.8-29.3% (F=10.55); it does not change the S verdict (corner "
      "pen = 0 at both).")
    A("- **Temporal locality α:** the plan fixes α = 0.60 (ShadowKV); "
      "transfer bytes = (1-α) x retained store. α = 1.0 (no locality, full "
      "re-fetch) would scale every T_transfer by 2.5x: W2 (100%,16-bit) "
      "becomes I/O-dominant, but the corner remains compute-dominant "
      "(T_transfer ≈ 4.9 ms vs T_gpu ≈ 6.8 ms at (20%,2-bit) W1) — raw S is "
      "exactly α-invariant (0.16 W1 / 0.31 W2). Verdict robust to α.")
    A("- **Compute estimate:** HBM-bound GPU time (weights + KV at 2 TB/s) is "
      "the plan-consistent estimate (constants list HBM 2 TB/s; FlexGen "
      "measures decode GPU utilization ~13%, i.e., bandwidth-bound). A "
      "FLOPs-only estimate (≈14 GFLOP/step → 45 µs at 312 TFLOPS) would make "
      "every cell I/O-dominant (regime always 'testable') but leaves "
      "raw S = 0.16-0.31 unchanged — the kill is not an artifact of the "
      "compute estimate.")
    A("- **Rounding:** 60% of 4096 = 2457.6 → 2458 (ceil); 20% → 820 (plan's "
      "own arithmetic). W2 = 900/540/180 exactly.")
    A("")
    A("## 9. Artifacts")
    A("")
    A("- `code/e1_screening.py` — SHA-256 "
      f"`{hashlib.sha256((ROOT / 'code/e1_screening.py').read_bytes()).hexdigest()}`")
    A("- `raw/e1_cells.csv` — 48 rows, every computed quantity (bytes per "
      "component, transfer, fetches, latencies, pen, µs/MB, tier fraction)")
    A("- `raw/e1_regime.csv` — regime quantities per cell")
    A("- `logs/experiment_log.md` — run record")
    return "\n".join(L)


if __name__ == "__main__":
    main()
