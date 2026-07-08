"""M1 reporting: turn the A100 baseline CSVs into the PLAN §8·M1 plots + tables.

Run **locally** after downloading the GPU job's ``outputs/`` into ``results/``:

    python benchmarks/report_m1.py --results results

Produces (skipping any whose inputs are missing):
  * memory vs context           (per cache)
  * decode ms/tok vs context    (per cache, p10–p90 band)
  * tokens/s vs batch           (per cache)
  * SM-util vs time             (NVML, representative decode)
  * GPU memory vs time          (NVML, proxy memory timeline)
  * top-op breakdown            (torch.profiler)
  * baseline matrix markdown table + memory-bound crossover note.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turbo_kv import reporting  # noqa: E402

_COLORS = {"dynamic_bf16": "#1f77b4", "static_bf16": "#2ca02c", "quantized_int4": "#d62728"}


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.get("status", "ok").astype(str).str.startswith("ok")].copy()


def plot_memory_vs_context(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cache, g in _ok(df).groupby("cache"):
        for batch, gb in g.groupby("batch"):
            gb = gb.sort_values("context")
            ax.plot(gb["context"], gb["peak_mem_mb"], marker="o",
                    color=_COLORS.get(cache), linestyle="-" if batch == 1 else "--",
                    label=f"{cache} b{batch}")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak allocated memory (MB)")
    ax.set_title("Peak memory vs context")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    reporting.save_fig(fig, "m1_memory_vs_context")


def plot_decode_vs_context(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cache, g in _ok(df[df["batch"] == 1]).groupby("cache"):
        g = g.sort_values("context")
        ax.plot(g["context"], g["decode_ms_tok_p50"], marker="o",
                color=_COLORS.get(cache), label=cache)
        ax.fill_between(g["context"], g["decode_ms_tok_p10"], g["decode_ms_tok_p90"],
                        color=_COLORS.get(cache), alpha=0.15)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("decode latency (ms / token)")
    ax.set_title("Decode ms/token vs context (batch=1, p10–p90)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m1_decode_vs_context")


def plot_tokens_per_s_vs_batch(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    big_ctx = _ok(df)["context"].max()
    sub = _ok(df[df["context"] == big_ctx])
    for cache, g in sub.groupby("cache"):
        g = g.sort_values("batch")
        ax.plot(g["batch"], g["tokens_per_s"], marker="o", color=_COLORS.get(cache), label=cache)
    ax.set_xlabel("batch size")
    ax.set_ylabel("tokens / second")
    ax.set_title(f"Decode throughput vs batch (context={big_ctx})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m1_tokens_per_s_vs_batch")


def plot_nvml(results: pathlib.Path) -> None:
    nvml_csv = results / "nvml_decode.csv"
    if not nvml_csv.exists():
        return
    n = pd.read_csv(nvml_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n["t_s"], n["util_gpu_pct"], color="#9467bd")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SM utilization (%)")
    ax.set_title("GPU SM-utilization during decode (memory-bound ⇒ low)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    reporting.save_fig(fig, "m1_sm_util_vs_time")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(n["t_s"], n["mem_used_mb"], color="#8c564b")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("GPU memory used (MB)")
    ax2.set_title("GPU memory vs time (decode)")
    ax2.grid(True, alpha=0.3)
    reporting.save_fig(fig2, "m1_memory_timeline")


def plot_quality_vs_context(df: pd.DataFrame) -> None:
    """Teacher-forced next-token KL vs the BF16 reference (cache equivalence/quality)."""
    if "tf_logit_kl" not in df.columns:
        return
    sub = _ok(df[df["batch"] == 1]).copy()
    sub = sub[sub["cache"] != "dynamic_bf16"]  # KL vs itself is 0 by construction
    sub = sub[pd.to_numeric(sub["tf_logit_kl"], errors="coerce").notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for cache, g in sub.groupby("cache"):
        g = g.sort_values("context")
        ax.plot(g["context"], g["tf_logit_kl"].clip(lower=1e-12), marker="o",
                color=_COLORS.get(cache), label=cache)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("teacher-forced next-token KL vs BF16")
    ax.set_title("Cache quality: per-step KL vs BF16 (batch=1, lower=better)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    reporting.save_fig(fig, "m1_quality_kl_vs_context")


def plot_op_breakdown(results: pathlib.Path) -> None:
    op_csv = results / "op_breakdown.csv"
    if not op_csv.exists():
        return
    ops = pd.read_csv(op_csv).sort_values("self_cuda_ms", ascending=True).tail(12)
    labels = ops["op"].map(lambda s: (s[:44] + "…") if len(str(s)) > 45 else str(s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, ops["self_cuda_ms"], color="#17becf")
    ax.set_xlabel("self CUDA time (ms, summed over decode)")
    ax.set_title("Top decode operators (torch.profiler)")
    ax.grid(True, axis="x", alpha=0.3)
    reporting.save_fig(fig, "m1_op_breakdown")


def write_matrix_table(df: pd.DataFrame) -> None:
    keep = ["cache", "context", "batch", "peak_mem_mb", "decode_ms_tok_p50",
            "tokens_per_s", "tf_argmax_match", "tf_logit_kl", "exact_match_vs_bf16", "status"]
    cols = [c for c in keep if c in df.columns]
    table = df[cols].sort_values(["cache", "batch", "context"])
    reporting.write_markdown_table(table, "m1_baseline_matrix", columns=cols)


def memory_bound_note(df: pd.DataFrame) -> str:
    """Find the context where decode first becomes memory-bound for BF16."""
    g = _ok(df[(df["cache"] == "dynamic_bf16") & (df["batch"] == 1)]).sort_values("context")
    if g.empty:
        return "no dynamic_bf16 batch=1 rows to assess memory-bound onset."
    base = g["decode_ms_tok_p50"].iloc[0]
    onset = None
    for _, r in g.iterrows():
        if r["decode_ms_tok_p50"] >= 1.20 * base:
            onset = int(r["context"])
            break
    lines = ["context, decode_ms_tok_p50, x_vs_smallest"]
    for _, r in g.iterrows():
        lines.append(f"{int(r['context'])}, {r['decode_ms_tok_p50']:.4f}, {r['decode_ms_tok_p50'] / base:.2f}x")
    verdict = (f"Decode becomes memory-bound (>20% latency growth vs smallest ctx) at "
               f"context = {onset} tokens." if onset else
               "Decode latency stayed within 20% across the swept contexts (compute/overhead-bound here).")
    note = "# M1 memory-bound crossover (dynamic_bf16, batch=1)\n\n" + "\n".join(lines) + "\n\n" + verdict + "\n"
    (reporting.TABLES_DIR / "m1_memory_bound.md").write_text(note, encoding="utf-8")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="dir containing baseline.csv etc.")
    args = ap.parse_args()
    results = pathlib.Path(args.results)

    baseline = results / "baseline.csv"
    if not baseline.exists():
        print(f"ERROR: {baseline} not found — download the GPU job outputs first.")
        return 2
    df = pd.read_csv(baseline)

    plot_memory_vs_context(df)
    plot_decode_vs_context(df)
    plot_tokens_per_s_vs_batch(df)
    plot_quality_vs_context(df)
    plot_nvml(results)
    plot_op_breakdown(results)
    write_matrix_table(df)
    verdict = memory_bound_note(df)

    print("M1 report written to results/plots and results/tables.")
    print("Decision:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
