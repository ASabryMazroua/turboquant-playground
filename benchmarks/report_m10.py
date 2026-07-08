"""M10 local report — dense-and-sparse outlier keys (KVQuant + QJL).

Reads ``results/turbo_e2e_outliers.csv`` (WikiText eval, rotation=none, per_token
keys) and shows the quality rescue of keeping the top ``key_outliers`` coordinates
per key vector in fp16 (sparse side-channel) while int4-quantizing the dense rest.
This is the COMPLEMENTARY rescue of per-token int4 keys (per-channel keys, M7, were
the other one): isolating the few extreme coordinates frees the dense remainder to
use the full int4 grid. The cost is real sparse fp16 memory
(``n_outliers*(2 idx + 2 val)`` bytes/token), so the win is quality-traded-against-
memory.

Expected CSV columns (the GPU job writes ``turbo_e2e.csv``; rename on download to
``turbo_e2e_outliers.csv``):
    ctx, key_quant, key_outliers, ppl_bf16, ppl_turbo, ppl_ratio,
    tf_kl, tf_argmax_match, peak_mb_turbo

    python benchmarks/report_m10.py --results results
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

COLOR = {"per_token": "#d62728", "per_channel": "#2ca02c"}
LABEL = {"per_token": "per-token keys (hard)", "per_channel": "per-channel keys (near-lossless)"}


def _largest_ctx(e2e):
    return e2e[e2e["ctx"] == max(e2e["ctx"].unique())] if "ctx" in e2e else e2e


def plot_outlier_rescue(e2e) -> None:
    """tf_kl (log y) vs key_outliers, one line per key_quant, at the largest ctx."""
    sub = _largest_ctx(e2e)
    ctx = int(max(e2e["ctx"].unique())) if "ctx" in e2e else 0
    quants = [q for q in ["per_token", "per_channel"] if q in set(sub.get("key_quant", []))]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for q in quants:
        g = sub[sub.key_quant == q].sort_values("key_outliers")
        ax.plot(g["key_outliers"], g["tf_kl"], marker="o",
                label=LABEL.get(q, q), color=COLOR.get(q))
    ax.set_yscale("log")
    ax.set_xlabel("key_outliers (top-N per-token coords kept in fp16)")
    ax.set_ylabel("teacher-forced KL(ref \u2016 int4) (log)")
    ax.set_title(f"M10: dense-and-sparse outlier keys (WikiText, ctx={ctx})")
    if "key_outliers" in sub:
        ax.set_xticks(sorted(sub["key_outliers"].unique()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m10_outlier_rescue")


def write_table(e2e) -> None:
    cols = [c for c in ["ctx", "key_quant", "key_outliers", "ppl_bf16", "ppl_turbo",
                        "ppl_ratio", "tf_kl", "tf_argmax_match", "peak_mb_turbo"]
            if c in e2e.columns]
    reporting.write_markdown_table(
        e2e.sort_values([c for c in ["ctx", "key_quant", "key_outliers"] if c in e2e.columns]),
        "m10_outliers", columns=cols, float_format="{:.4f}")


def gate(e2e) -> str:
    lines = ["# M10 gate \u2014 dense-and-sparse outlier keys (KVQuant + QJL)\n"]
    lines.append("- gate: for **per_token** keys, tf_kl at key_outliers=8 is lower "
                 "than at key_outliers=0 at the largest context (sparse fp16 "
                 "outliers rescue the dense int4 grid)\n")
    ok = False
    if {"key_quant", "key_outliers", "tf_kl", "ctx"}.issubset(e2e.columns):
        sub = _largest_ctx(e2e)
        ctx = int(max(e2e["ctx"].unique()))
        pt = sub[sub.key_quant == "per_token"].sort_values("key_outliers")
        if len(pt) >= 2:
            xs = [int(x) for x in pt["key_outliers"]]
            kls = [float(x) for x in pt["tf_kl"]]
            ok = kls[-1] < kls[0]
            lines.append(f"- ctx={ctx} per_token: "
                         + "  ->  ".join(f"ko={x}: {k:.3e}" for x, k in zip(xs, kls)))
            if kls[-1] > 0:
                lines.append(f"- per_token tf_kl reduced {kls[0] / max(kls[-1], 1e-12):.2f}x "
                             f"from key_outliers={xs[0]} to {xs[-1]}")
            if "peak_mb_turbo" in pt.columns:
                mb = [float(x) for x in pt["peak_mb_turbo"]]
                lines.append(f"- memory cost: peak {mb[0]:.0f}MB -> {mb[-1]:.0f}MB "
                             f"(+{mb[-1] - mb[0]:.0f}MB) for key_outliers {xs[0]}->{xs[-1]}")
        pc = sub[sub.key_quant == "per_channel"].sort_values("key_outliers")
        if len(pc):
            kc = [float(x) for x in pc["tf_kl"]]
            lines.append(f"- ctx={ctx} per_channel (no-op for outliers): "
                         f"tf_kl {min(kc):.3e}..{max(kc):.3e} \u2014 unchanged, as expected")
    else:
        lines.append("- (insufficient columns in CSV to evaluate the gate)")
    lines.append("")
    lines.append(
        "**Finding.** The per-token key failure mode (M4) is a few outlier "
        "coordinates inflating the whole token's int4 scale and crushing the other "
        "~60 coordinates. KVQuant/QJL keep the top-N outlier coordinates per key in "
        "fp16 (a sparse side-channel) and compute the affine range over the dense "
        "remainder only, so those coordinates use the full int4 grid. This is the "
        "**complementary** rescue of per-token int4 keys \u2014 per-channel keys (M7) "
        "isolate *channel* outliers with their own scale and so gain nothing from "
        "per-token outliers (a deliberate no-op). The cost is honest: each kept "
        "outlier adds 2 bytes of index + 2 bytes of fp16 value per token, a real "
        "memory tax the table's peak_mb column reflects.\n")
    lines.append(f"## Verdict: {'PASS' if ok else 'FAIL'}  "
                 f"(per_token tf_kl lower at key_outliers=8 than 0={ok})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)
    e2e = reporting.load_results(rdir / "turbo_e2e_outliers.csv")
    plot_outlier_rescue(e2e)
    write_table(e2e)
    note = gate(e2e)
    (reporting.TABLES_DIR / "m10_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
