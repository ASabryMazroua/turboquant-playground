"""M16 local report — wiring QJL into end-to-end generation.

Reads ``results/turbo_e2e_qjl.csv`` (WikiText eval) and compares the three key
caches at matched context: **per-token** int4 (the M4 disaster), **per-channel**
int4 (the M7 redemption), and **qjl** (the unbiased inner-product sign sketch of
M11, now used in real generation). The point of M16 is to close the loop: does
the paper's unbiased-IP correction actually work end-to-end, and how does it
stack up against the KIVI per-channel fix?

    python benchmarks/report_m16.py --results results
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from turbo_kv import reporting  # noqa: E402

ORDER = ["per_token", "per_channel", "qjl"]
COLOR = {"per_token": "#d62728", "per_channel": "#2ca02c", "qjl": "#1f77b4"}
LABEL = {
    "per_token": "per-token int4 (M4 disaster)",
    "per_channel": "per-channel int4 (M7 fix)",
    "qjl": "QJL sign sketch (M16)",
}
PPL_TOL = 1.20


def _qjl_key_bits(row, head_dim: int = 64) -> float:
    """Realized QJL key bits/value: m sign bits + bf16 norm + fp16 outliers."""
    m = float(row.get("qjl_m", 0) or 0)
    n_out = float(row.get("qjl_outliers", 0) or 0)
    idx_bits = math.ceil(math.log2(head_dim)) if head_dim > 1 else 0
    return m / head_dim + 16.0 / head_dim + n_out * (16.0 + idx_bits) / head_dim


def _best_qjl(e2e, c):
    """The lowest-ppl_ratio qjl row at context ``c`` (None if no qjl rows)."""
    sub = e2e[(e2e.ctx == c) & (e2e.key_quant == "qjl")]
    if not len(sub):
        return None
    return sub.loc[sub["ppl_ratio"].idxmin()]


def _ratio(e2e, q, c):
    if q == "qjl":
        row = _best_qjl(e2e, c)
        return float(row["ppl_ratio"]) if row is not None else float("nan")
    sub = e2e[(e2e.ctx == c) & (e2e.key_quant == q)]
    if not len(sub):
        return float("nan")
    # Pin the int4 baselines to the M7-canonical rotation=none when available.
    if "rotation" in sub.columns and (sub["rotation"] == "none").any():
        sub = sub[sub["rotation"] == "none"]
    return float(sub["ppl_ratio"].iloc[0])


def plot_compare(e2e) -> None:
    ctxs = sorted(e2e["ctx"].unique())
    quants = [q for q in ORDER if q in set(e2e["key_quant"])]
    x = np.arange(len(ctxs))
    w = 0.8 / max(len(quants), 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, q in enumerate(quants):
        vals = [_ratio(e2e, q, c) for c in ctxs]
        label = LABEL.get(q, q)
        if q == "qjl":  # annotate the best config used per ctx
            best = _best_qjl(e2e, ctxs[-1])
            if best is not None:
                label += (f" — best {best.get('rotation', '?')} m={int(best['qjl_m'])}"
                          f",o={int(best['qjl_outliers'])}")
        bars = ax.bar(x + i * w, vals, w, label=label, color=COLOR.get(q))
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2g}×",
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.7)
    ax.axhline(PPL_TOL, color="k", ls="--", lw=1, alpha=0.6, label=f"near-lossless ≤ {PPL_TOL}×")
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(quants) - 1) / 2)
    ax.set_xticklabels([f"ctx={c}" for c in ctxs])
    ax.set_ylabel("perplexity ratio (cache / BF16, log)")
    ax.set_title("M16: QJL keys end-to-end vs per-token / per-channel int4 (WikiText)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m16_qjl_e2e")


def plot_qjl_ablation(e2e) -> None:
    """At the largest ctx, show every QJL (rotation, m, outliers) config vs the
    int4 baselines — the variance mechanism: outliers and larger m help, but even
    the best config (m=512, 8 outliers ≈ 11 bits/val) can't reach per-channel int4."""
    c = sorted(e2e["ctx"].unique())[-1]
    qj = e2e[(e2e.ctx == c) & (e2e.key_quant == "qjl")].copy()
    if not len(qj):
        return
    rots = sorted(qj["rotation"].unique())
    combos = sorted({(int(r.qjl_m), int(r.qjl_outliers)) for r in qj.itertuples()})
    x = np.arange(len(combos))
    w = 0.8 / max(len(rots), 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    rot_color = {"none": "#1f77b4", "rht": "#ff7f0e"}
    for i, rot in enumerate(rots):
        vals = []
        for (m, o) in combos:
            sub = qj[(qj.rotation == rot) & (qj.qjl_m == m) & (qj.qjl_outliers == o)]
            vals.append(float(sub["ppl_ratio"].iloc[0]) if len(sub) else float("nan"))
        bars = ax.bar(x + i * w, vals, w, label=f"rotation={rot}",
                      color=rot_color.get(rot, None))
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}×",
                        ha="center", va="bottom", fontsize=7)
    ax.axhline(_ratio(e2e, "per_token", c), color="#d62728", ls="--", lw=1.2,
               label=f"per-token int4 ({_ratio(e2e, 'per_token', c):.0f}×)")
    ax.axhline(_ratio(e2e, "per_channel", c), color="#2ca02c", ls="--", lw=1.2,
               label=f"per-channel int4 ({_ratio(e2e, 'per_channel', c):.2f}×)")
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(rots) - 1) / 2)
    ax.set_xticklabels([f"m={m}\no={o}\n≈{m/64 + 16/64 + o*22/64:.1f}b" for (m, o) in combos],
                       fontsize=8)
    ax.set_ylabel("perplexity ratio (QJL / BF16, log)")
    ax.set_title(f"M16 QJL ablation @ ctx={c}: outliers & m help, but variance stays fatal")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m16_qjl_ablation")


def write_table(e2e) -> None:
    cols = ["ctx", "rotation", "key_quant", "qjl_m", "qjl_outliers", "ppl_bf16", "ppl_turbo",
            "ppl_ratio", "tf_kl", "tf_argmax_match", "peak_mb_bf16", "peak_mb_turbo"]
    cols = [c for c in cols if c in e2e.columns]
    df = e2e[e2e.key_quant.isin(ORDER)].copy()
    df["__o"] = df["key_quant"].map({q: i for i, q in enumerate(ORDER)})
    sort_keys = ["ctx", "__o"] + (["ppl_ratio"] if "ppl_ratio" in df.columns else [])
    df = df.sort_values(sort_keys)
    reporting.write_markdown_table(df, "m16_qjl_e2e", columns=cols, float_format="{:.4f}")


def gate(e2e) -> str:
    ctxs = sorted(e2e["ctx"].unique())

    # Core claim: the BEST QJL config (unbiased IP) should at least beat the
    # per-token int4 disaster end-to-end. Reported honestly either way.
    beats_pt = all(_ratio(e2e, "qjl", c) < _ratio(e2e, "per_token", c)
                   for c in ctxs if _ratio(e2e, "per_token", c) == _ratio(e2e, "per_token", c))
    lines = ["# M16 gate — QJL wired into end-to-end generation\n"]
    lines.append("- gate: the best `qjl` key cache ppl_ratio < `per_token` int4 ppl_ratio at "
                 "every context (the paper's unbiased inner-product correction must at least "
                 "rescue the per-token disaster in real generation)\n")
    for c in ctxs:
        best = _best_qjl(e2e, c)
        cfg = (f" (best {best.get('rotation', '?')} m={int(best['qjl_m'])},o={int(best['qjl_outliers'])}, "
               f"≈{_qjl_key_bits(best):.1f} key-bits/val)") if best is not None else ""
        lines.append(
            f"- ctx={int(c)}: per-token {_ratio(e2e, 'per_token', c):.2f}×  ·  "
            f"per-channel {_ratio(e2e, 'per_channel', c):.3f}×  ·  "
            f"**qjl {_ratio(e2e, 'qjl', c):.2f}×**{cfg}")

    # Memory angle: the manual QJL attention can't use flash-attention, so it
    # materialises the score matrix — peak memory vs the BF16 baseline it competes with.
    cmax = ctxs[-1]
    bestmax = _best_qjl(e2e, cmax)
    mem_note = ""
    if bestmax is not None and "peak_mb_turbo" in e2e.columns:
        qp = float(bestmax["peak_mb_turbo"])
        bp = float(e2e[(e2e.ctx == cmax) & (e2e.key_quant == "per_channel")]["peak_mb_bf16"].iloc[0])
        mem_note = (f" At ctx={int(cmax)} the QJL path peaks at **{qp:.0f} MB** vs **{bp:.0f} MB** "
                    f"for the BF16 baseline ({qp / max(bp, 1e-9):.1f}×): the custom attention "
                    "that the sketch forces (no key reconstruction → no flash-attention) "
                    "materialises the score matrix, forfeiting the memory win a KV cache exists for.")

    # Ablation lever: the single biggest error reduction comes from fp16 outliers.
    abl = ""
    def _qcfg(c, m, o):
        s = e2e[(e2e.ctx == c) & (e2e.key_quant == "qjl") & (e2e.qjl_m == m)
                & (e2e.qjl_outliers == o)]
        # take the better rotation for this (m,o) so the lever isn't rotation-confounded
        return float(s["ppl_ratio"].min()) if len(s) else float("nan")
    worst = _qcfg(cmax, 256, 0)
    bestcfg = _ratio(e2e, "qjl", cmax)
    o0 = _qcfg(cmax, 512, 0)
    o8 = _qcfg(cmax, 512, 8)
    if worst == worst and o0 == o0 and o8 == o8:
        abl = (f" The ablation is clean: at ctx={int(cmax)}, going m=256→512 and adding 8 fp16 "
               f"outliers cuts QJL from **{worst:.0f}×** to **{bestcfg:.0f}×** (outliers alone: "
               f"m=512 {o0:.0f}×→{o8:.0f}×, ~{o0 / max(o8, 1e-9):.0f}×) — variance ~1/m and the "
               f"outlier side-channel both help, exactly as the theory predicts, yet the best "
               f"config (~{_qjl_key_bits(bestmax):.0f} bits/val, 2.7× int4's cost) is still "
               f"~{bestcfg / max(_ratio(e2e, 'per_channel', cmax), 1e-9):.0f}× worse than 4-bit per-channel.")
    lines.append("")
    lines.append(
        "**Finding.** M6/M11 validated the QJL unbiased inner-product estimator only as a "
        "numeric study; M16 wires it into real Qwen2 generation via a custom attention path. "
        + ("Wired in, the best QJL config **beats** per-token int4 — the unbiased estimate "
           "survives the outlier-channel inflation that wrecks per-token MSE quant. "
           if beats_pt else
           "Wired in, QJL is **catastrophically worse** than even per-token int4 (the M4 "
           "disaster): the estimator is unbiased but **high-variance**, and softmax attention "
           "over thousands of keys is exquisitely sensitive to per-logit variance — every "
           "noisy score is a chance to spuriously win the max, so attention scatters and "
           "perplexity explodes.")
        + abl
        + mem_note
        + " Per-channel int4 (M7) remains the right end-to-end fix; QJL's home is the "
        "retrieval / no-scale regime, not dense KV attention at a small head_dim — extending "
        "the M11 PARTIAL verdict from a numeric to an end-to-end result.\n")
    lines.append(f"## Verdict: {'PASS' if beats_pt else 'FAIL'}  (best qjl beats per-token at every ctx={beats_pt})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)
    e2e = reporting.load_results(rdir / "turbo_e2e_qjl.csv")
    plot_compare(e2e)
    plot_qjl_ablation(e2e)
    write_table(e2e)
    note = gate(e2e)
    (reporting.TABLES_DIR / "m16_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
