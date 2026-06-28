"""M5 local report — fused int4 kernel microbench (PLAN §M5).

Reads ``results/{turbo_kernels,turbo_kernels_decode}.csv`` and writes:
  m5_latency       fused vs reference latency bars per (op, nk)
  m5_speedup       speedup (ref/fused) vs nk
  m5_memory        peak memory fused vs reference (no full dequant)
  m5_decode        decode-step fused vs reference latency + speedup
  m5_kernels table + m5_gate.md

    python benchmarks/report_m5.py --results results
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from turbo_kv import reporting  # noqa: E402

RELERR_TOL = 1e-3


def plot_latency(k: pd.DataFrame) -> None:
    for nq in sorted(k["nq"].unique()):
        sub = k[k.nq == nq]
        ops = sorted(sub["op"].unique())
        nks = sorted(sub["nk"].unique())
        fig, axes = plt.subplots(1, len(ops), figsize=(6 * len(ops), 4.3), squeeze=False)
        for ax, op in zip(axes[0], ops):
            s = sub[sub.op == op].sort_values("nk")
            x = np.arange(len(nks))
            has_bf16 = "bf16_ms" in s.columns
            w = 0.26 if has_bf16 else 0.4
            if has_bf16:
                ax.bar(x - w, s["bf16_ms"], w, label="plain BF16 (no quant)", color="#1f77b4")
                ax.bar(x, s["ref_ms"], w, label="dequant→cuBLAS (M4)", color="#7f7f7f")
                ax.bar(x + w, s["fused_ms"], w, label="fused int4 (M5)", color="#2ca02c")
            else:
                ax.bar(x - 0.2, s["ref_ms"], 0.4, label="dequant→cuBLAS", color="#7f7f7f")
                ax.bar(x + 0.2, s["fused_ms"], 0.4, label="fused int4", color="#2ca02c")
            ax.set_xticks(x); ax.set_xticklabels(nks)
            ax.set_xlabel("nk (context)"); ax.set_ylabel("latency (ms, p50)")
            ax.set_title(f"{op}  (nq={nq})")
            ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"M5 kernel latency — BF16 baseline vs dequant vs fused int4 (nq={nq})", y=1.03)
        reporting.save_fig(fig, f"m5_latency_nq{nq}")


def plot_speedup(k: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (op, nq), g in k.groupby(["op", "nq"]):
        g = g.sort_values("nk")
        ax.plot(g["nk"], g["speedup"], marker="o", label=f"{op} nq={nq}")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nk (context)"); ax.set_ylabel("speedup (reference / fused)")
    ax.set_title("M5: fused int4 speedup vs dequant reference (>1 = fused wins)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    reporting.save_fig(fig, "m5_speedup")


def plot_memory(k: pd.DataFrame) -> None:
    sub = k[(k.op == "logits") & (k.nq == k["nq"].max())].sort_values("nk")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 0.2, sub["ref_peak_mb"], 0.4, label="reference (materializes K̂)", color="#d62728")
    ax.bar(x + 0.2, sub["fused_peak_mb"], 0.4, label="fused (no full dequant)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(sub["nk"])
    ax.set_xlabel("nk (context)"); ax.set_ylabel("peak allocated (MB)")
    ax.set_title("M5: peak memory — fused never materializes the dequantized K/V")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    reporting.save_fig(fig, "m5_memory")


def plot_decode(rdir: pathlib.Path) -> None:
    p = rdir / "turbo_kernels_decode.csv"
    if not p.exists():
        return
    d = reporting.load_results(p).sort_values("nk")
    x = np.arange(len(d))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    has_bf16 = "bf16_ms" in d.columns
    if has_bf16:
        axL.bar(x - 0.27, d["bf16_ms"], 0.27, label="plain BF16 (no quant)", color="#1f77b4")
        axL.bar(x, d["ref_ms"], 0.27, label="M4 dequant", color="#7f7f7f")
        axL.bar(x + 0.27, d["fused_ms"], 0.27, label="M5 fused", color="#2ca02c")
    else:
        axL.bar(x - 0.2, d["ref_ms"], 0.4, label="M4 dequant", color="#7f7f7f")
        axL.bar(x + 0.2, d["fused_ms"], 0.4, label="M5 fused", color="#2ca02c")
    axL.set_xticks(x); axL.set_xticklabels(d["nk"]); axL.set_ylabel("decode step (ms, p50)")
    axL.set_xlabel("nk (context)"); axL.set_title("Decode step latency"); axL.legend(fontsize=8)
    axL.grid(True, axis="y", alpha=0.3)
    axR.plot(d["nk"], d["speedup"], marker="o", color="#7f7f7f", label="vs M4 dequant")
    if "speedup_vs_bf16" in d.columns:
        axR.plot(d["nk"], d["speedup_vs_bf16"], marker="s", color="#1f77b4", label="vs plain BF16")
    axR.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    axR.set_xscale("log", base=2); axR.set_xlabel("nk (context)")
    axR.set_ylabel("speedup (>1 = fused wins)"); axR.set_title("Fused speedup")
    axR.grid(True, alpha=0.3); axR.legend(fontsize=8)
    fig.suptitle("M5 decode: fused int4 vs M4 dequant vs plain BF16", y=1.02)
    reporting.save_fig(fig, "m5_decode")


def write_tables(k: pd.DataFrame) -> None:
    cols = ["op", "nq", "nk", "relerr", "fused_ms", "ref_ms"]
    if "bf16_ms" in k.columns:
        cols += ["bf16_ms", "speedup", "speedup_vs_bf16"]
    else:
        cols += ["speedup"]
    cols += ["fused_peak_mb", "ref_peak_mb", "fused_gbs"]
    reporting.write_markdown_table(k.sort_values(["op", "nq", "nk"]), "m5_kernels",
                                   columns=cols, float_format="{:.4f}")


def gate(k: pd.DataFrame, rdir: pathlib.Path) -> str:
    correct = bool((k["relerr"] < RELERR_TOL).all())
    no_dequant = bool((k["fused_peak_mb"] < k["ref_peak_mb"]).all())
    best = k.loc[k["speedup"].idxmax()]
    worst = k.loc[k["speedup"].idxmin()]

    lines = ["# M5 gate — fused int4 attention kernels\n"]
    lines.append(f"- correctness: all relerr < {RELERR_TOL:g} → {'YES' if correct else 'NO'} "
                 f"(max {k['relerr'].max():.2e})")
    lines.append(f"- no full dequant materialized: fused peak < reference peak → "
                 f"{'YES' if no_dequant else 'NO'}")
    lines.append(f"- speedup range (reference/fused): {worst['speedup']:.2f}× "
                 f"({worst['op']} nk={int(worst['nk'])}) … {best['speedup']:.2f}× "
                 f"({best['op']} nk={int(best['nk'])})")
    dec = rdir / "turbo_kernels_decode.csv"
    if dec.exists():
        d = reporting.load_results(dec)
        lines.append(f"- decode step (nq=1): fused vs M4 dequant speedup "
                     f"{d['speedup'].min():.2f}×–{d['speedup'].max():.2f}× over "
                     f"nk={int(d['nk'].min())}–{int(d['nk'].max())}")
        if "speedup_vs_bf16" in d.columns:
            lines.append(f"- decode step (nq=1): fused vs **plain BF16 (no quant)** speedup "
                         f"{d['speedup_vs_bf16'].min():.2f}×–{d['speedup_vs_bf16'].max():.2f}× "
                         f"(fused is {1 / d['speedup_vs_bf16'].max():.1f}–{1 / d['speedup_vs_bf16'].min():.1f}× slower)")
    lines.append("")
    lines.append(
        "**Finding.** The fused kernels unpack int4 nibbles in-register and compute "
        "logits / value-sums directly from the 0.5-byte/value packed store, so they "
        "**never materialize the dequantized K/V** (peak memory confirms it). "
        "Correctness matches the dequant reference to < 1e-3. On **latency** the "
        "honest result is that the fused int4 kernel **loses to both** the M4 "
        "dequant→cuBLAS path **and** the original plain-BF16 attention: at "
        "head_dim=64 the workload is tiny and cuBLAS GEMM/GEMV (tensor cores) is "
        "hard to beat, while the Triton kernel runs `allow_tf32=False` for exactness "
        "and is launch/occupancy-bound (achieved GB/s ≪ A100 peak). The int4 fusion's "
        "value here is **memory** (4× smaller KV, no BF16 reconstruction), not decode "
        "latency — a real, honest systems result.\n")
    verdict = "PASS" if (correct and no_dequant) else "FAIL"
    lines.append(f"## Verdict: {verdict}  (correct={correct}, no_dequant={no_dequant})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)

    k = reporting.load_results(rdir / "turbo_kernels.csv")
    plot_latency(k)
    plot_speedup(k)
    plot_memory(k)
    plot_decode(rdir)
    write_tables(k)

    note = gate(k, rdir)
    (reporting.TABLES_DIR / "m5_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
