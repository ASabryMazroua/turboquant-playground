"""Reporting helpers: turn results CSVs into committed comparison tables + plots.

Every milestone writes rows via :func:`append_row` to ``results/*.csv`` and then
renders artifacts through the helpers here, so figures and tables always
regenerate from the CSVs (no hand-built artifacts). A headless matplotlib
backend is used so this runs unchanged inside an AML job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
TRACES_DIR = RESULTS_DIR / "traces"

for _d in (RESULTS_DIR, PLOTS_DIR, TABLES_DIR, TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def append_row(csv_path: str | Path, row: Mapping) -> Path:
    """Append one row to a CSV, creating it with headers if it does not exist."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(row)])
    frame.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    return csv_path


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load a results CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def write_markdown_table(
    df: pd.DataFrame,
    name: str,
    columns: Sequence[str] | None = None,
    float_format: str = "{:.3f}",
) -> Path:
    """Render a DataFrame to a committed markdown table under ``results/tables/``.

    Implemented without ``tabulate`` so the only hard dep is pandas.
    """
    if columns is not None:
        df = df[list(columns)]
    cols = [str(c) for c in df.columns]

    def fmt(value) -> str:
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in df.columns) + " |")

    out = TABLES_DIR / (name if name.endswith(".md") else f"{name}.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def save_fig(fig, name: str) -> list[Path]:
    """Save a matplotlib figure as both PNG and SVG under ``results/plots/``."""
    import matplotlib.pyplot as plt

    stem = name.rsplit(".", 1)[0] if name.endswith((".png", ".svg")) else name
    paths: list[Path] = []
    for ext in ("png", "svg"):
        p = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    *,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    logy: bool = False,
    name: str | None = None,
):
    """Generic grouped line plot. Saves if ``name`` is given, else returns the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if hue:
        for key, g in df.groupby(hue):
            g = g.sort_values(x)
            ax.plot(g[x], g[y], marker="o", label=str(key))
        ax.legend(title=hue)
    else:
        g = df.sort_values(x)
        ax.plot(g[x], g[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, name) if name else fig
