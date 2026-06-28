"""M13 local report — group-wise VALUE quantization (KIVI/AWQ).

Reads ``results/turbo_e2e_valuegroup.csv`` (WikiText eval, rotation=none,
per_channel keys, post-RoPE) and shows the quality gain of quantizing VALUES
per-token in GROUPS of ``value_group_size`` coordinates (one int4 scale per group)
instead of one scale per whole token. Finer value scales help when intra-head
coordinate magnitude varies; the cost is a small amount of extra scale storage
(``ng`` scales/zeros per token instead of 1). The ``residual_length`` BF16 window
IS the KIVI residual buffer — recent tokens kept full precision and evicted in
chunks — already implemented in the cache.

Expected CSV columns (the AML job writes ``turbo_e2e.csv``; rename on download to
``turbo_e2e_valuegroup.csv``):
    ctx, key_quant, value_group_size, ppl_bf16, ppl_turbo, ppl_ratio,
    tf_kl, tf_argmax_match, peak_mb_turbo

    python benchmarks/report_m13.py --results results
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from turbo_kv import reporting  # noqa: E402


def _largest_ctx(e2e):
    return e2e[e2e["ctx"] == max(e2e["ctx"].unique())] if "ctx" in e2e else e2e


def plot_value_groups(e2e) -> None:
    """tf_kl (log y) vs value_group_size at the largest ctx (expect drop 0->32)."""
    if not {"value_group_size", "tf_kl"}.issubset(e2e.columns):
        return
    sub = _largest_ctx(e2e).sort_values("value_group_size")
    ctx = int(max(e2e["ctx"].unique())) if "ctx" in e2e else 0
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(sub["value_group_size"], sub["tf_kl"], marker="o", color="#1f77b4",
            label="grouped int4 values")
    ax.set_yscale("log")
    ax.set_xlabel("value_group_size (0 = whole-head per-token)")
    ax.set_ylabel("teacher-forced KL(ref \u2016 int4) (log)")
    ax.set_title(f"M13: group-wise value quantization (WikiText, ctx={ctx})")
    ax.set_xticks(sorted(sub["value_group_size"].unique()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m13_value_groups")


def write_table(e2e) -> None:
    cols = [c for c in ["ctx", "key_quant", "value_group_size", "ppl_bf16",
                        "ppl_turbo", "ppl_ratio", "tf_kl", "tf_argmax_match",
                        "peak_mb_turbo"]
            if c in e2e.columns]
    sort_cols = [c for c in ["ctx", "key_quant", "value_group_size"] if c in e2e.columns]
    reporting.write_markdown_table(
        e2e.sort_values(sort_cols) if sort_cols else e2e,
        "m13_value_groups", columns=cols, float_format="{:.4f}")


def gate(e2e) -> str:
    lines = ["# M13 gate \u2014 group-wise VALUE quantization (KIVI/AWQ)\n"]
    lines.append("- gate: grouped values (value_group_size=32) tf_kl is \u2264 "
                 "whole-head (value_group_size=0) tf_kl at the largest context\n")
    ok = False
    if {"value_group_size", "tf_kl", "ctx"}.issubset(e2e.columns):
        sub = _largest_ctx(e2e).sort_values("value_group_size")
        ctx = int(max(e2e["ctx"].unique()))
        xs = [int(x) for x in sub["value_group_size"]]
        kls = [float(x) for x in sub["tf_kl"]]
        if 0 in xs and any(x > 0 for x in xs):
            kl0 = kls[xs.index(0)]
            grouped = [(x, k) for x, k in zip(xs, kls) if x > 0]
            best_g = min(grouped, key=lambda t: t[1])
            ok = best_g[1] <= kl0
            lines.append(f"- ctx={ctx}: "
                         + "  ->  ".join(f"vgs={x}: {k:.3e}" for x, k in zip(xs, kls)))
            if best_g[1] > 0:
                lines.append(f"- tf_kl whole-head(0)={kl0:.3e} -> grouped({best_g[0]})="
                             f"{best_g[1]:.3e} ({kl0 / max(best_g[1], 1e-12):.2f}x)")
            if "peak_mb_turbo" in sub.columns:
                mb = {int(r): float(m) for r, m in
                      zip(sub["value_group_size"], sub["peak_mb_turbo"])}
                if 0 in mb and best_g[0] in mb:
                    lines.append(f"- memory cost: peak {mb[0]:.0f}MB -> "
                                 f"{mb[best_g[0]]:.0f}MB (+{mb[best_g[0]] - mb[0]:.0f}MB) "
                                 f"for the extra per-group value scales")
        else:
            lines.append("- (need both value_group_size=0 and a grouped row to gate)")
    else:
        lines.append("- (insufficient columns in CSV to evaluate the gate)")
    lines.append("")
    lines.append(
        "**Finding.** Values were quantized per-token with ONE int4 scale over all "
        "64 head coordinates. When a value head's coordinate magnitude varies "
        "across the head, that single scale is set by the largest sub-block and "
        "wastes the int4 grid on the rest. KIVI/AWQ quantize values per-token but "
        "in GROUPS (e.g. 32 coords), one affine scale/zero per group, so each "
        "sub-block gets its own range \u2014 finer value resolution exactly where "
        "intra-head magnitude varies. The cost is small and honest: ``ng = D/G`` "
        "scales+zeros per token instead of 1 (counted in the table's "
        "peak_mb / scale_zero bytes). This is a VALUES-only lever \u2014 keys keep "
        "their per-channel / per-token-outlier schemes, untouched. Separately, our "
        "``residual_length`` BF16 window IS the KIVI **residual buffer**: the most "
        "recent tokens are kept full-precision and only evicted (and quantized) in "
        "chunks once they fall out of the window \u2014 already implemented.\n")
    lines.append(f"## Verdict: {'PASS' if ok else 'FAIL'}  "
                 f"(grouped values tf_kl \u2264 whole-head at largest ctx={ok})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)
    e2e = reporting.load_results(rdir / "turbo_e2e_valuegroup.csv")
    plot_value_groups(e2e)
    write_table(e2e)
    note = gate(e2e)
    (reporting.TABLES_DIR / "m13_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
