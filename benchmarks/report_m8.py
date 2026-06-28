"""M8 local report — pre-RoPE key quantization (the KVQuant fix).

Reads ``results/turbo_e2e_prerope.csv`` (WikiText eval, rotation=none,
key_quant=per_channel) and compares **post-RoPE** (M7, quantize keys after RoPE)
vs **pre-RoPE** (KVQuant: quantize the raw key, re-apply RoPE on reconstruction).
RoPE injects position-dependent variation that hurts quantization, so pre-RoPE
should be at least as good as post-RoPE and ideally better.

Expected CSV columns (the AML job writes ``turbo_e2e.csv``; rename on download to
``turbo_e2e_prerope.csv``):
    ctx, key_quant, rope_mode, ppl_bf16, ppl_turbo, ppl_ratio, tf_kl, tf_argmax_match

    python benchmarks/report_m8.py --results results
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from turbo_kv import reporting  # noqa: E402

COLOR = {"post": "#ff7f0e", "pre": "#1f77b4"}
LABEL = {"post": "post-RoPE keys (M7)", "pre": "pre-RoPE keys (KVQuant fix)"}
KL_TOL = 1.10  # pre-RoPE tf_kl must be <= post-RoPE tf_kl * KL_TOL (within noise)


def _per_channel(e2e):
    return e2e[e2e["key_quant"] == "per_channel"] if "key_quant" in e2e else e2e


def plot_prerope(e2e) -> None:
    e2e = _per_channel(e2e)
    ctxs = sorted(e2e["ctx"].unique())
    modes = [m for m in ["post", "pre"] if m in set(e2e["rope_mode"])]
    x = np.arange(len(ctxs))
    w = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, m in enumerate(modes):
        vals = [float(e2e[(e2e.ctx == c) & (e2e.rope_mode == m)]["tf_kl"].iloc[0]) for c in ctxs]
        bars = ax.bar(x + i * w, vals, w, label=LABEL.get(m, m), color=COLOR.get(m))
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2e}", ha="center",
                    va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(modes) - 1) / 2)
    ax.set_xticklabels([f"ctx={c}" for c in ctxs])
    ax.set_ylabel("teacher-forced KL(ref ‖ int4) (log)")
    ax.set_title("M8: pre-RoPE vs post-RoPE per-channel key quantization (WikiText)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m8_pre_rope")


def write_table(e2e) -> None:
    cols = ["ctx", "key_quant", "rope_mode", "ppl_bf16", "ppl_turbo", "ppl_ratio",
            "tf_kl", "tf_argmax_match"]
    reporting.write_markdown_table(_per_channel(e2e).sort_values(["ctx", "rope_mode"]),
                                   "m8_pre_rope", columns=cols, float_format="{:.4f}")


def gate(e2e) -> str:
    e2e = _per_channel(e2e)
    post = e2e[e2e.rope_mode == "post"]
    pre = e2e[e2e.rope_mode == "pre"]
    ctxs = sorted(e2e["ctx"].unique())
    ok = True
    lines = ["# M8 gate — pre-RoPE key quantization (KVQuant fix)\n"]
    lines.append(f"- gate: pre-RoPE tf_kl <= post-RoPE tf_kl x {KL_TOL} at every context "
                 "(pre-RoPE does not hurt, ideally helps)\n")
    for c in ctxs:
        kp = float(post[post.ctx == c]["tf_kl"].iloc[0]) if len(post[post.ctx == c]) else float("nan")
        kr = float(pre[pre.ctx == c]["tf_kl"].iloc[0]) if len(pre[pre.ctx == c]) else float("nan")
        good = kr <= kp * KL_TOL
        ok = ok and good
        lines.append(f"- ctx={int(c)}: post tf_kl {kp:.3e}  ->  pre **{kr:.3e}**  "
                     f"({kp / max(kr, 1e-12):.2f}x {'better' if kr < kp else 'worse'})")
    lines.append("")
    lines.append(
        "**Finding.** M7 recovered near-lossless int4 KV with per-channel key "
        "quantization, but still quantized keys *post*-RoPE. RoPE mixes adjacent "
        "channels with position-dependent rotations, smearing the per-channel "
        "statistics that per-channel quantization relies on. KVQuant's fix is to "
        "quantize the **raw pre-RoPE key** and re-apply RoPE to the reconstructed "
        "key at attention time (we store a tiny int32 positions buffer to do so, "
        "preserving the ~4x memory win). Pre-RoPE keys should match or beat "
        "post-RoPE on teacher-forced KL at every context.\n")
    lines.append(f"## Verdict: {'PASS' if ok else 'FAIL'}  (pre-RoPE does not hurt={ok})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir = pathlib.Path(args.results)
    e2e = reporting.load_results(rdir / "turbo_e2e_prerope.csv")
    plot_prerope(e2e)
    write_table(e2e)
    note = gate(e2e)
    (reporting.TABLES_DIR / "m8_gate.md").write_text(note + "\n", encoding="utf-8")
    try:
        print(note)
    except UnicodeEncodeError:
        print(note.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
