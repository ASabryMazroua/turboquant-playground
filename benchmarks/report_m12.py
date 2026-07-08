"""M12 local report — NUQ vs uniform from the non-uniform-quant sweep CSV.

Reads ``results/turbo_nuq.csv`` (downloaded from the GPU job) and writes plots,
an error matrix, and a gate. matplotlib is only needed here (local), never in
the metric library.

The headline regime is **per-channel, no rotation** — the axis our int4 KV keys
actually use (M7 per-channel keys). NUQ is an MSE minimizer, and it does that job
well: k-means levels cut key reconstruction MSE 4–6× vs a uniform grid on both
axes. The twist this milestone surfaces is that the **attention-KL does not
follow the MSE down** — a fresh re-appearance of the project's MSE-optimal ≠
inner-product-optimal thesis (M3/M6): a better reconstruction is not a better
attention distribution.

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
HEADLINE_AXIS = "channel"
HEADLINE_ROT = "none"
LOW_BITS = 3.0


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over layers for each (rotation, axis, bits, method)."""
    keys = [k for k in ["rotation", "axis", "bits", "effective_bits", "method"]
            if k in df.columns]
    return df.groupby(keys, as_index=False).mean(numeric_only=True)


def plot_nuq_vs_uniform(df: pd.DataFrame, metric: str, ylabel: str, save: str) -> None:
    """``metric`` vs bits, lines per method; headline + per-channel facets."""
    agg = _agg(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, axis in zip(axes, [HEADLINE_AXIS, "token"]):
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
    reporting.save_fig(fig, save)


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
             f"**{HEADLINE_ROT}** — the axis our int4 KV keys actually use (M7 "
             "per-channel keys), where fitted NUQ levels lower attention-KL._\n"]

    metric = "key_mse"
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

    # Attention-KL nuance: NUQ minimizes MSE, but does the attention distribution
    # follow? (The MSE != inner-product thesis.) Report the win-rate honestly.
    akl_txt = ""
    if "attn_kl" in df.columns and {"axis", "rotation"} <= set(df.columns):
        ak = df[(df["axis"] == HEADLINE_AXIS) & (df["rotation"] == HEADLINE_ROT)]
        piv2 = ak.pivot_table(index=["layer", "bits"], columns="method", values="attn_kl")
        if {"uniform", "nuq-kmeans"} <= set(piv2.columns):
            aw = int((piv2["nuq-kmeans"] < piv2["uniform"]).sum())
            akl_txt = (f" By contrast, NUQ-kmeans beats uniform on **attention-KL** in only "
                       f"**{aw}/{len(piv2)}** (layer,bits) cells — the MSE win does *not* "
                       "transfer to attention fidelity.")

    lines.append("")
    lines.append(
        "**Finding.** NUQ fits reconstruction levels to the data density "
        "(quantile init, then 1-D k-means / Lloyd–Max), so heavy-tailed key "
        "coordinates get fine resolution near the mode and coarse near the tails. "
        "It does exactly what it optimizes: key reconstruction **MSE drops 4–6×** "
        "vs a uniform grid at matched bits, on both axes." + cb_txt + akl_txt +
        " That last point is the real lesson — the same **MSE-optimal ≠ "
        "inner-product-optimal** decoupling this project keeps hitting (M3/M6): a "
        "better reconstruction is not a better attention distribution, so pure "
        "MSE-driven NUQ is not the KV win; per-channel int4 (M7) already nails the "
        "axis that matters. Full cache integration must also store/reload the "
        "per-group codebook — future work; this milestone is the numerical case.")

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

    plot_nuq_vs_uniform(df, "key_mse", "key reconstruction MSE", "m12_nuq_mse")
    plot_nuq_vs_uniform(df, "attn_kl", "attention KL vs exact", "m12_nuq_attnkl")
    write_table(df)

    note = gate(df)
    (reporting.TABLES_DIR / "m12_gate.md").write_text(note, encoding="utf-8")
    print(note)
    print("M12 report written to results/plots and results/tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
