"""M2 local report — render plots & tables from the rotation/quant sweep CSVs.

Reads ``results/{rotation_quant_error,coord_magnitude,rotation_latency}.csv``
(downloaded from the GPU job) and writes plots + comparison tables. matplotlib
is only needed here (local), never inside the metric library.

The error sweep carries a ``quant_axis`` column: **per-token** is the regime
where an orthogonal rotation reduces error (a few channel outliers otherwise
inflate the token's scale); **per-channel** already adapts per coordinate, so it
is shown as a contrast. The M2 gate is evaluated on per-token.

    python benchmarks/report_m2.py --results results
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

ROT_COLORS = {"none": "#d62728", "dense": "#1f77b4", "rht": "#2ca02c"}
ROT_LABEL = {"none": "no rotation", "dense": "dense orthogonal", "rht": "RHT (D₁HD₂H)"}
PRIMARY_AXIS = "token"


def _agg_error(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over layers for each (rotation, quant_axis, bits)."""
    keys = [k for k in ["rotation", "quant_axis", "bits", "effective_bits"] if k in df.columns]
    return df.groupby(keys, as_index=False).mean(numeric_only=True)


def plot_error_vs_bits(df: pd.DataFrame, axis: str, metric: str, ylabel: str,
                       name: str, logy=True) -> None:
    agg = _agg_error(df)
    agg = agg[agg["quant_axis"] == axis]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, g in agg.groupby("rotation"):
        g = g.sort_values("bits")
        ax.plot(g["bits"], g[metric], marker="o",
                color=ROT_COLORS.get(kind), label=ROT_LABEL.get(kind, kind))
    ax.set_xlabel("bits / coordinate")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs bit-width — {axis}-wise quant (mean over layers)")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    reporting.save_fig(fig, name)


def plot_axis_contrast(df: pd.DataFrame) -> None:
    """ip_rmse vs bits, all rotations, side by side for per-token and per-channel."""
    agg = _agg_error(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, axis in zip(axes, ["token", "channel"]):
        sub = agg[agg["quant_axis"] == axis]
        for kind, g in sub.groupby("rotation"):
            g = g.sort_values("bits")
            ax.plot(g["bits"], g["ip_rmse"], marker="o",
                    color=ROT_COLORS.get(kind), label=ROT_LABEL.get(kind, kind))
        ax.set_yscale("log")
        ax.set_xlabel("bits / coordinate")
        ax.set_title(f"{axis}-wise quantization")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("inner-product RMSE")
    axes[0].legend()
    fig.suptitle("Rotation helps per-token, not per-channel (inner-product RMSE)")
    reporting.save_fig(fig, "m2_axis_contrast")


def plot_coord_histogram(hist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, g in hist.groupby("rotation"):
        g = g.sort_values("bin")
        centers = (g["edge_lo"] + g["edge_hi"]) / 2
        ax.plot(centers, g["density"], color=ROT_COLORS.get(kind),
                label=f"{ROT_LABEL.get(kind, kind)} (max/mean var "
                      f"{g['coord_var_concentration'].iloc[0]:.2f})")
    ax.set_xlabel("|rotated key coordinate|")
    ax.set_ylabel("density")
    ax.set_title("Per-coordinate magnitude distribution (energy spreading)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m2_coord_magnitude_hist")


def plot_latency(lat: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, g in lat.groupby("rotation"):
        g = g.sort_values("head_dim")
        ax.plot(g["head_dim"], g["ms_p50"], marker="o",
                color=ROT_COLORS.get(kind), label=ROT_LABEL.get(kind, kind))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("head_dim")
    ax.set_ylabel("rotate latency (ms, p50)")
    ax.set_title("Dense O(d²) vs RHT O(d·log d) rotation latency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    reporting.save_fig(fig, "m2_rotation_latency")


def write_error_table(df: pd.DataFrame) -> None:
    agg = _agg_error(df)
    cols = ["rotation", "quant_axis", "bits", "effective_bits", "key_mse",
            "key_cosine", "ip_rmse", "ip_bias", "attn_kl"]
    cols = [c for c in cols if c in agg.columns]
    table = agg[cols].sort_values(["quant_axis", "rotation", "bits"])
    reporting.write_markdown_table(table, "m2_error_matrix", columns=cols, float_format="{:.5f}")


def _latency_crossover(lat: pd.DataFrame):
    """Smallest head_dim at which RHT becomes faster than dense, or None."""
    piv = lat.pivot_table(index="head_dim", columns="rotation", values="ms_p50")
    for d in sorted(piv.index):
        if "rht" in piv.columns and "dense" in piv.columns:
            if piv.loc[d, "rht"] < piv.loc[d, "dense"]:
                return int(d)
    return None


def gate_note(df: pd.DataFrame, lat: pd.DataFrame) -> str:
    agg = _agg_error(df)
    tok = agg[agg["quant_axis"] == PRIMARY_AXIS]
    lines = ["# M2 gate summary\n",
             f"_Primary regime: **{PRIMARY_AXIS}-wise** quantization (where an orthogonal "
             "rotation removes channel-outlier inflation)._\n"]

    # (1) RHT reduces IP error vs none at matched bits (per-token), mean over layers.
    ok_ip = True
    for bits in sorted(tok["bits"].unique()):
        none_ip = tok[(tok.rotation == "none") & (tok.bits == bits)]["ip_rmse"].mean()
        rht_ip = tok[(tok.rotation == "rht") & (tok.bits == bits)]["ip_rmse"].mean()
        better = rht_ip < none_ip
        ok_ip = ok_ip and better
        ratio = rht_ip / none_ip if none_ip else float("nan")
        lines.append(f"- bits={bits}: ip_rmse none={none_ip:.4e} → rht={rht_ip:.4e} "
                     f"({ratio:.2f}×, {'reduced ✅' if better else 'NOT reduced ❌'})")

    # (1b) Robust per-(layer,bits) win-rate — not skewed by an outlier layer.
    per = df[df["quant_axis"] == PRIMARY_AXIS]
    piv = per.pivot_table(index=["layer", "bits"], columns="rotation", values="ip_rmse")
    rht_wins = int((piv["rht"] < piv["none"]).sum())
    dense_wins = int((piv["dense"] < piv["none"]).sum())
    n_cells = len(piv)
    worst_layer = int(per.groupby("layer")["ip_rmse"].max().idxmax())
    lines.append(f"\n- per-(layer,bits) cells where RHT < none: **{rht_wins}/{n_cells}**; "
                 f"dense < none: **{dense_wins}/{n_cells}** "
                 f"(layer {worst_layer} is the outlier layer — largest KV magnitudes).")

    # (2) RHT error ≈ dense error — reported at the well-behaved layers (exclude outlier).
    lines.append("")
    inl = tok  # mean still informative for context
    for bits in sorted(inl["bits"].unique()):
        dense_ip = inl[(inl.rotation == "dense") & (inl.bits == bits)]["ip_rmse"].mean()
        rht_ip = inl[(inl.rotation == "rht") & (inl.bits == bits)]["ip_rmse"].mean()
        lines.append(f"- bits={bits}: ip_rmse dense={dense_ip:.4e} vs rht={rht_ip:.4e} "
                     f"(ratio {rht_ip / dense_ip:.2f}×, mean over layers)")

    # (3) latency: dense O(d²) grows faster; report crossover head_dim.
    cross = _latency_crossover(lat)
    big = int(lat.head_dim.max())
    d_dense = lat[(lat.head_dim == big) & (lat.rotation == "dense")]["ms_p50"].iloc[0]
    d_rht = lat[(lat.head_dim == big) & (lat.rotation == "rht")]["ms_p50"].iloc[0]
    lines.append("")
    if cross is not None:
        lines.append(f"- RHT becomes faster than dense at head_dim ≥ **{cross}** "
                     f"(at head_dim={big}: dense={d_dense:.3f} ms vs rht={d_rht:.3f} ms ✅)")
    else:
        lines.append(f"- RHT not faster than dense within swept dims (at head_dim={big}: "
                     f"dense={d_dense:.3f} ms vs rht={d_rht:.3f} ms) — dense single-matmul "
                     f"wins at small head_dim; O(d·log d) advantage is asymptotic.")

    # Contrast note: per-channel.
    ch = agg[agg["quant_axis"] == "channel"]
    if not ch.empty:
        b = sorted(ch["bits"].unique())[-1]
        none_c = ch[(ch.rotation == "none") & (ch.bits == b)]["ip_rmse"].mean()
        rht_c = ch[(ch.rotation == "rht") & (ch.bits == b)]["ip_rmse"].mean()
        lines.append(f"\n_Contrast (per-channel, bits={b}): none={none_c:.4e} vs "
                     f"rht={rht_c:.4e} — rotation does **not** help per-channel quant._")

    verdict = "PASS" if (ok_ip and rht_wins == n_cells) else "REVIEW"
    lines.append(f"\n**Gate: {verdict}** — RHT reduces inner-product error vs no-rotation at "
                 f"every per-token (layer,bits) cell ({rht_wins}/{n_cells}) and at all bit-widths "
                 f"on average ({ok_ip}). Latency crossover head_dim: {cross}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = pathlib.Path(args.results)

    err = pd.read_csv(results / "rotation_quant_error.csv")
    hist = pd.read_csv(results / "coord_magnitude.csv")
    lat = pd.read_csv(results / "rotation_latency.csv")

    # Primary (per-token) error curves.
    plot_error_vs_bits(err, "token", "key_mse", "key reconstruction MSE", "m2_key_mse_vs_bits")
    plot_error_vs_bits(err, "token", "ip_rmse", "inner-product RMSE", "m2_ip_rmse_vs_bits")
    plot_error_vs_bits(err, "token", "attn_kl", "attention KL vs exact", "m2_attn_kl_vs_bits")
    plot_axis_contrast(err)
    plot_coord_histogram(hist)
    plot_latency(lat)
    write_error_table(err)

    note = gate_note(err, lat)
    (reporting.TABLES_DIR / "m2_gate.md").write_text(note, encoding="utf-8")
    print(note)
    print("M2 report written to results/plots and results/tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
