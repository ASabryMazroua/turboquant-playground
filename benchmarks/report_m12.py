"""M12 local report — NUQ vs uniform from the non-uniform-quant sweep CSV.

Reads ``results/turbo_nuq.csv`` (downloaded from the AML job) and writes plots,
an error matrix, and a gate. matplotlib is only needed here (local), never in
the metric library.

The headline regime is **per-token, no rotation**: heavy-tailed coordinates
otherwise stretch a uniform grid, so fitted (NUQ) levels help most. Per-channel
is shown as a contrast (already adapts per coordinate, so NUQ helps less).

    python benchmarks/report_m12.py --results results
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from turbo_kv import reporting  # noqa: E402

METHOD_COLORS = {"uniform": "#d62728", "nuq-quantile": "#1f77b4", "nuq-kmeans": "#2ca02c"}
METHOD_LABEL = {
    "uniform": "uniform affine",
    "nuq-quantile": "NUQ (quantile)",
    "nuq-kmeans": "NUQ (k-means)",
}
HEADLINE_AXIS = "token"
HEADLINE_ROT = "none"
LOW_BITS = 3.0


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over layers for each (rotation, axis, bits, method)."""
    keys = [k for k in ["rotation", "axis", "bits", "effective_bits", "method"]
            if k in df.columns]
    return df.groupby(keys, as_index=False).mean(numeric_only=True)


def plot_nuq_vs_uniform(df: pd.DataFrame, metric: str, ylabel: str) -> None:
    """``metric`` vs bits, lines per method; headline + per-channel facets."""
    agg = _agg(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, axis in zip(axes, [HEADLINE_AXIS, "channel"]):
        sub = agg[(agg["axis"] == axis) & (agg["rotation"] == HEADLINE_ROT)]
        for method, g in sub.groupby("method"):
            g = g.sort_values("bits")
            ax.plot(g["bits"], g[metric], marker="o",
                    color=METHOD_COLORS.get(method), label=METHOD_LABEL.get(method, method))
        ax.set_yscale("log")
        ax.set_xlabel("bits / coordinate")
        ax.set_title(f"{axis}-wise quant (rotation={HEADLINE_ROT})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(f"NUQ vs uniform — {ylabel} (mean over layers)")
    reporting.save_fig(fig, "m12_nuq_vs_uniform")


def write_table(df: pd.DataFrame) -> None:
    agg = _agg(df)
    cols = ["rotation", "axis", "bits", "effective_bits", "method", "key_mse",
            "ip_rmse", "attn_kl", "codebook_bits_per_value"]
    cols = [c for c in cols if c in agg.columns]
    table = agg[cols].sort_values([c for c in ["axis", "rotation", "bits", "method"]
                                   if c in agg.columns])
    reporting.write_markdown_table(table, "m12_nuq", columns=cols, float_format="{:.6f}")


def gate(df: pd.DataFrame) -> str:
    agg = _agg(df)
    lines = ["# M12 gate summary\n",
             f"_Headline regime: **{HEADLINE_AXIS}-wise** quant, rotation="
             f"**{HEADLINE_ROT}** (heavy-tailed coordinates stretch a uniform grid, "
             "so fitted NUQ levels help most)._\n"]

    metric = "attn_kl" if "attn_kl" in agg.columns else "ip_rmse"
    head = agg[(agg["axis"] == HEADLINE_AXIS) & (agg["rotation"] == HEADLINE_ROT)]

    def _val(bits, method):
        sub = head[(head["bits"] == bits) & (head["method"] == method)][metric]
        return sub.mean() if len(sub) else float("nan")

    ok_low = True
    have_low = False
    for bits in sorted(head["bits"].unique()):
        uni = _val(bits, "uniform")
        kqu = _val(bits, "nuq-quantile")
        kkm = _val(bits, "nuq-kmeans")
        better = kkm < uni
        ratio = kkm / uni if uni else float("nan")
        lines.append(f"- bits={bits}: {metric} uniform={uni:.4e} → nuq-quantile={kqu:.4e} "
                     f"→ nuq-kmeans={kkm:.4e} ({ratio:.2f}×, "
                     f"{'NUQ better ✅' if better else 'not better ❌'})")
        if bits <= LOW_BITS and uni == uni and kkm == kkm:  # finite
            have_low = True
            ok_low = ok_low and better

    # Robust per-(layer,bits) win-rate at low bits (≤ LOW_BITS), headline regime.
    per = df[(df["axis"] == HEADLINE_AXIS) & (df["rotation"] == HEADLINE_ROT)
             & (df["bits"] <= LOW_BITS)] if {"axis", "rotation"} <= set(df.columns) else df.iloc[0:0]
    win_txt = ""
    if not per.empty and "method" in per.columns:
        piv = per.pivot_table(index=["layer", "bits"], columns="method", values=metric)
        if {"uniform", "nuq-kmeans"} <= set(piv.columns):
            wins = int((piv["nuq-kmeans"] < piv["uniform"]).sum())
            n = len(piv)
            win_txt = f" Per-(layer,bits) cells at ≤{LOW_BITS} bits where NUQ-kmeans < uniform: **{wins}/{n}**."

    # Codebook overhead (honest cost).
    cb_txt = ""
    if "codebook_bits_per_value" in head.columns:
        cb = head[head["method"] == "nuq-kmeans"]["codebook_bits_per_value"]
        if len(cb):
            cb_txt = (f" NUQ stores a per-group fp16 codebook adding ≈"
                      f"{cb.mean():.3f} bits/value overhead (uniform: 0).")

    lines.append("")
    lines.append(
        "**Finding.** NUQ fits reconstruction levels to the data density "
        "(quantile init, then 1-D k-means / Lloyd–Max), so heavy-tailed key "
        "coordinates get fine resolution near the mode and coarse resolution in "
        "the tails. It helps most at low bits and in the per-token regime, where "
        "a uniform min/max grid wastes levels on the tails." + cb_txt +
        " Full cache integration needs to store and reload that codebook per "
        "group, so it is future work; this milestone is the numerical case for "
        "the win.")

    verdict = "PASS" if (have_low and ok_low) else "REVIEW"
    lines.append(
        f"\n**Gate: {verdict}** — NUQ (k-means) {metric} < uniform at matched "
        f"low bits (≤{LOW_BITS}) for the headline setting ({ok_low})." + win_txt)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = pathlib.Path(args.results)

    df = pd.read_csv(results / "turbo_nuq.csv")

    plot_nuq_vs_uniform(df, "attn_kl", "attention KL vs exact")
    write_table(df)

    note = gate(df)
    (reporting.TABLES_DIR / "m12_gate.md").write_text(note, encoding="utf-8")
    print(note)
    print("M12 report written to results/plots and results/tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
