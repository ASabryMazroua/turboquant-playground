"""M4 local report — render plots & tables from the patched-attention CSVs.

Reads ``results/{turbo_e2e,turbo_layer_kl,turbo_ops}.csv`` (downloaded from the
GPU job) and writes the M4 artifacts: quality-vs-rotation×ctx, perplexity ratio,
decode-ms/tok & peak-mem (BF16 vs TurboKV), a per-layer attention-output
divergence heatmap, the decode op breakdown, comparison tables and the gate.

    python benchmarks/report_m4.py --results results
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

ROT_COLORS = {"none": "#d62728", "dense": "#1f77b4", "rht": "#2ca02c"}
ROT_LABEL = {"none": "no rotation", "dense": "dense orthogonal", "rht": "RHT"}

# Coherence thresholds (mirror M3): a rotation is coherent if it keeps the
# next-token distribution close to BF16 and barely inflates perplexity.
KL_TOL = 0.5
MATCH_TOL = 0.90
PPL_TOL = 1.20


def _has_sink(df: pd.DataFrame) -> bool:
    return "sink_length" in df.columns and df["sink_length"].nunique() > 1


def _primary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for the headline plots: the largest sink (the intended fix) if a
    sink sweep is present, else the frame unchanged."""
    if "sink_length" in df.columns:
        return df[df["sink_length"] == df["sink_length"].max()]
    return df


def plot_sink_sweep(e2e: pd.DataFrame) -> None:
    """Perplexity ratio vs BF16-sink length, per rotation × ctx — the sink test."""
    if not _has_sink(e2e):
        return
    ctxs = sorted(e2e["ctx"].unique())
    rots = [r for r in ["none", "rht", "dense"] if r in set(e2e["rotation"])]
    fig, axes = plt.subplots(1, len(ctxs), figsize=(5 * len(ctxs), 4.2), sharey=True)
    if len(ctxs) == 1:
        axes = [axes]
    for ax, c in zip(axes, ctxs):
        for rot in rots:
            g = e2e[(e2e.ctx == c) & (e2e.rotation == rot)].sort_values("sink_length")
            ax.plot(g["sink_length"], g["ppl_ratio"], marker="o",
                    color=ROT_COLORS.get(rot), label=ROT_LABEL.get(rot, rot))
        ax.axhline(PPL_TOL, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(f"ctx={c}")
        ax.set_xlabel("BF16 attention-sink tokens")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("perplexity ratio (turbo / BF16, log)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("M4 attention-sink test: ppl ratio vs #sink tokens kept in BF16", y=1.02)
    reporting.save_fig(fig, "m4_sink_sweep")


def plot_corpus_comparison(rdir: pathlib.Path) -> None:
    """Synthetic-tiled vs natural-WikiText ppl_ratio — the M4 validation-C anchor."""
    wpath = rdir / "turbo_e2e_wikitext.csv"
    spath = rdir / "turbo_e2e.csv"
    if not (wpath.exists() and spath.exists()):
        return
    s = reporting.load_results(spath)
    if "sink_length" in s.columns:
        s = s[s["sink_length"] == 0]
    w = reporting.load_results(wpath)
    ctxs = sorted(set(s["ctx"]) & set(w["ctx"]))
    rots = [r for r in ["none", "rht", "dense"] if r in set(w["rotation"])]
    fig, axes = plt.subplots(1, len(ctxs), figsize=(5 * len(ctxs), 4.2), sharey=True)
    if len(ctxs) == 1:
        axes = [axes]
    x = np.arange(len(rots))
    for ax, c in zip(axes, ctxs):
        sv = [float(s[(s.ctx == c) & (s.rotation == r)]["ppl_ratio"].iloc[0]) for r in rots]
        wv = [float(w[(w.ctx == c) & (w.rotation == r)]["ppl_ratio"].iloc[0]) for r in rots]
        ax.bar(x - 0.2, sv, 0.4, label="synthetic (tiled)", color="#ff7f0e")
        ax.bar(x + 0.2, wv, 0.4, label="WikiText (natural)", color="#1f77b4")
        ax.axhline(PPL_TOL, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(rots)
        ax.set_title(f"ctx={c}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("perplexity ratio (turbo / BF16, log)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("M4-C: int4 KV quality — synthetic-tiled vs natural WikiText", y=1.02)
    reporting.save_fig(fig, "m4_corpus_comparison")


def plot_quality(e2e: pd.DataFrame) -> None:
    """Grouped bars: teacher-forced KL per rotation, one cluster per ctx (log y)."""
    e2e = _primary(e2e)
    ctxs = sorted(e2e["ctx"].unique())
    rots = [r for r in ["none", "rht", "dense"] if r in set(e2e["rotation"])]
    x = np.arange(len(ctxs))
    w = 0.8 / max(len(rots), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, rot in enumerate(rots):
        vals = [float(e2e[(e2e.ctx == c) & (e2e.rotation == rot)]["tf_kl"].iloc[0]) for c in ctxs]
        ax.bar(x + i * w, vals, w, label=ROT_LABEL.get(rot, rot), color=ROT_COLORS.get(rot))
    ax.axhline(KL_TOL, color="k", ls="--", lw=1, alpha=0.6, label=f"coherence tol {KL_TOL}")
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(rots) - 1) / 2)
    ax.set_xticklabels([f"ctx={c}" for c in ctxs])
    ax.set_ylabel("teacher-forced KL vs BF16 (log)")
    ax.set_title("M4 quality: next-token KL by rotation × context")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m4_quality_vs_rotation")


def plot_perplexity(e2e: pd.DataFrame) -> None:
    """Perplexity ratio (turbo / BF16) per rotation × ctx."""
    e2e = _primary(e2e)
    ctxs = sorted(e2e["ctx"].unique())
    rots = [r for r in ["none", "rht", "dense"] if r in set(e2e["rotation"])]
    x = np.arange(len(ctxs))
    w = 0.8 / max(len(rots), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, rot in enumerate(rots):
        vals = [float(e2e[(e2e.ctx == c) & (e2e.rotation == rot)]["ppl_ratio"].iloc[0]) for c in ctxs]
        bars = ax.bar(x + i * w, vals, w, label=ROT_LABEL.get(rot, rot), color=ROT_COLORS.get(rot))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(PPL_TOL, color="k", ls="--", lw=1, alpha=0.6, label=f"tol {PPL_TOL}×")
    ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x + w * (len(rots) - 1) / 2)
    ax.set_xticklabels([f"ctx={c}" for c in ctxs])
    ax.set_ylabel("perplexity ratio (turbo / BF16)")
    ax.set_title("M4 perplexity inflation by rotation × context")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m4_perplexity_ratio")


def plot_cost(e2e: pd.DataFrame) -> None:
    """Decode ms/tok and peak-mem bars: BF16 vs TurboKV (RHT) per ctx."""
    e2e = _primary(e2e)
    rot = "rht" if "rht" in set(e2e["rotation"]) else sorted(e2e["rotation"])[0]
    sub = e2e[e2e.rotation == rot].sort_values("ctx")
    ctxs = [int(c) for c in sub["ctx"]]
    x = np.arange(len(ctxs))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    axL.bar(x - 0.2, sub["decode_ms_bf16"], 0.4, label="BF16", color="#7f7f7f")
    axL.bar(x + 0.2, sub["decode_ms_turbo"], 0.4, label=f"TurboKV ({rot})", color="#2ca02c")
    axL.set_xticks(x); axL.set_xticklabels([f"ctx={c}" for c in ctxs])
    axL.set_ylabel("decode latency (ms/token)")
    axL.set_title("Decode latency")
    axL.grid(True, axis="y", alpha=0.3); axL.legend(fontsize=8)

    axR.bar(x - 0.2, sub["peak_mb_bf16"], 0.4, label="BF16", color="#7f7f7f")
    axR.bar(x + 0.2, sub["peak_mb_turbo"], 0.4, label=f"TurboKV ({rot})", color="#1f77b4")
    axR.set_xticks(x); axR.set_xticklabels([f"ctx={c}" for c in ctxs])
    axR.set_ylabel("peak allocated (MB)")
    axR.set_title("Peak memory")
    axR.grid(True, axis="y", alpha=0.3); axR.legend(fontsize=8)
    fig.suptitle("M4 cost: BF16 vs TurboKV", y=1.02)
    reporting.save_fig(fig, "m4_cost_bf16_vs_turbo")


def plot_layer_divergence(layer: pd.DataFrame) -> None:
    """Heatmap of per-layer attention-output relerr (rows=layer, cols=rotation×ctx)."""
    layer = layer.copy()
    layer["col"] = layer["rotation"] + "\nctx=" + layer["ctx"].astype(str)
    pivot = layer.pivot_table(index="layer", columns="col", values="attn_out_relerr")
    fig, ax = plt.subplots(figsize=(1.4 * pivot.shape[1] + 2, 0.28 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", origin="lower")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(0, pivot.shape[0], 2)); ax.set_yticklabels(range(0, pivot.shape[0], 2))
    ax.set_ylabel("layer"); ax.set_title("Per-layer attention-output relerr (turbo vs BF16)")
    fig.colorbar(im, ax=ax, label="relative L2 error")
    reporting.save_fig(fig, "m4_layer_divergence")


def plot_ops(ops: pd.DataFrame) -> None:
    """Top CUDA ops on one decode step (torch.profiler)."""
    sub = ops.sort_values("cuda_us", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sub["op"], sub["cuda_us"], color="#1f77b4")
    ax.set_xlabel("self CUDA time (µs)")
    ax.set_title("M4 decode-step op breakdown (torch.profiler)")
    ax.grid(True, axis="x", alpha=0.3)
    reporting.save_fig(fig, "m4_decode_ops")


def write_tables(e2e: pd.DataFrame) -> None:
    cols = ["ctx", "rotation"]
    if "sink_length" in e2e.columns:
        cols.append("sink_length")
    cols += ["tf_kl", "tf_argmax_match", "ppl_bf16", "ppl_turbo",
             "ppl_ratio", "decode_ms_bf16", "decode_ms_turbo", "peak_mb_bf16", "peak_mb_turbo"]
    sort_cols = [c for c in ["ctx", "rotation", "sink_length"] if c in e2e.columns]
    reporting.write_markdown_table(e2e.sort_values(sort_cols), "m4_e2e",
                                   columns=cols, float_format="{:.4f}")


def gate(e2e: pd.DataFrame) -> str:
    sink_note = ""
    if "sink_length" in e2e.columns:
        best_sink = e2e["sink_length"].max()
        rht = e2e[(e2e.rotation == "rht") & (e2e.sink_length == best_sink)]
        sink_note = f" (RHT, sink={best_sink})"
    else:
        rht = e2e[e2e.rotation == "rht"]
    coherent_all = bool(
        (rht["tf_kl"] < KL_TOL).all()
        and (rht["tf_argmax_match"] > MATCH_TOL).all()
        and (rht["ppl_ratio"] < PPL_TOL).all()
    ) if len(rht) else False

    lines = ["# M4 gate — patched Qwen2 attention (rotate-query int4 KV)\n"]
    lines.append(f"- thresholds: tf_kl<{KL_TOL}, argmax-match>{MATCH_TOL}, ppl_ratio<{PPL_TOL}{sink_note}\n")
    sort_cols = [c for c in ["ctx", "rotation", "sink_length"] if c in e2e.columns]
    for _, r in e2e.sort_values(sort_cols).iterrows():
        ok = (r.tf_kl < KL_TOL and r.tf_argmax_match > MATCH_TOL and r.ppl_ratio < PPL_TOL)
        flag = "coherent" if ok else "INCOHERENT"
        sk = f" sink={int(r.sink_length)}" if "sink_length" in e2e.columns else ""
        lines.append(f"- ctx={int(r.ctx)} {r.rotation}{sk}: tf_kl={r.tf_kl:.3e} "
                     f"match={r.tf_argmax_match:.3f} ppl_ratio={r.ppl_ratio:.3f} "
                     f"decode {r.decode_ms_turbo:.2f}ms/tok (bf16 {r.decode_ms_bf16:.2f}) "
                     f"peak {r.peak_mb_turbo:.0f}MB (bf16 {r.peak_mb_bf16:.0f}) -> {flag}")
    lines.append("")
    lines.append(
        "**Finding.** The rotate-query *patch* is correct (pytest 74/74; the "
        "per-layer probe with a non-quantized cache shows `none` bit-exact and "
        "`rht`/`dense` only a stable ~3.5% bf16 rotated-basis floor). But the "
        "held-out **novel-text** eval (valid reference, ppl_bf16≈6.5) shows the "
        "4-bit `TurboKVCache` is **pervasively lossy**: median next-token KL ~1–6 "
        "and ppl inflated 4–260×. This overturns M3's 'near-lossless' result, "
        "which was an artifact of the repeated-text eval — induction/copy heads "
        "hide KV corruption on repeated text but not on novel text. Rotation "
        "helps at ctx=4096 (rht ppl_ratio 5.9 < dense 11.0 < none 58.7, "
        "reproducing the M2 per-token result) but the relationship **inverts at "
        "long context**: at 8k–16k rht degrades badly (rht spikes to ppl_ratio "
        "1389× at 8k while none is 4.4×) — rotation spreads each logit into a sum "
        "of many noisy int4 terms, so over thousands of keys a spurious maximum "
        "can dominate the softmax. BF16 **attention-sink** preservation (sink=4) "
        "was tested and gives only minor relief (none 58.7→17.8× at 4k, "
        "166→85× at 16k; rht unchanged) — so the sink is **not** the primary "
        "cause. Conclusion: per-token int4 KV is insufficient for novel-text "
        "fidelity; the principled fix is the **QJL +1-bit residual (M6)** — the "
        "paper's unbiased-inner-product correction, since 4-bit MSE quant is "
        "biased — and/or per-channel key quantization (KIVI).\n")
    verdict = "PASS" if coherent_all else "FAIL (quality) — patch validated, int4-only insufficient"
    lines.append(f"## Verdict: {verdict}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)

    e2e = reporting.load_results(rdir / "turbo_e2e.csv")
    plot_quality(e2e)
    plot_perplexity(e2e)
    plot_cost(e2e)
    plot_sink_sweep(e2e)
    plot_corpus_comparison(rdir)
    write_tables(e2e)

    layer_path = rdir / "turbo_layer_kl.csv"
    if layer_path.exists():
        plot_layer_divergence(reporting.load_results(layer_path))
    ops_path = rdir / "turbo_ops.csv"
    if ops_path.exists():
        plot_ops(reporting.load_results(ops_path))

    note = gate(e2e)
    (reporting.TABLES_DIR / "m4_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
