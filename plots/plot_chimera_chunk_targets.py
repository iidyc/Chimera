#!/usr/bin/env python3
"""Plot chimera chunk-size sensitivity from the updated profiling summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_paths import latest_required


DEFAULT_DATASET_ORDER = ["lotte", "hotpot"]
FULL_DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}
TARGET_ORDER = ["0.90", "0.95"]
TARGET_LABELS = {
    "0.90": "Recall@100=0.90",
    "0.95": "Recall@100=0.95",
}
BLUE = "#0072B2"
BEST_RED = "#D62728"
LEGACY_CHIMERA_SYSTEM = "v" "8"
CHIMERA_SYSTEM_ALIASES = {"chimera", "gpu_search", LEGACY_CHIMERA_SYSTEM}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot Chimera QPS versus overlap chunk count.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Chimera chunk target summary CSV. Defaults to latest profiling/chimera_chunk_targets summary.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=24)
    parser.add_argument(
        "--plot",
        choices=["all", "focus", "stacked", "both"],
        default="both",
        help="Which figure to generate.",
    )
    parser.add_argument(
        "--include-msmarco",
        action="store_true",
        help="Include MS MARCO in the all-grid plot. Disabled by default.",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="Chunk values to include. Defaults to 1 through 8.",
    )
    parser.add_argument(
        "--focus-target",
        choices=TARGET_ORDER,
        default="0.95",
        help="Recall@100 target for the one-row focus plot.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") not in CHIMERA_SYSTEM_ALIASES:
                continue
            row = dict(row)
            row["target_recall"] = f"{float(row['target_recall']):.2f}"
            row["chunks"] = str(int(row["chunks"]))
            rows.append(row)
    required = {"dataset", "target_recall", "chunks", "qps_mean", "qps_std"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"input CSV is missing required columns: {sorted(missing)}")
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_recall"])].append(row)
    for series in grouped.values():
        series.sort(key=lambda row: int(row["chunks"]))
    return dict(grouped)


def qps(row: dict[str, str]) -> float:
    return float(row["qps_mean"])


def qps_std(row: dict[str, str]) -> float:
    return float(row.get("qps_std") or 0.0)


def best_row(series: list[dict[str, str]]) -> dict[str, str]:
    return max(series, key=qps)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6, alpha=0.65)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)


def set_rcparams(font_size: float) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "normal",
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size + 1,
            "ytick.labelsize": font_size + 1,
            "legend.fontsize": font_size,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def chunk_one_row(series: list[dict[str, str]]) -> dict[str, str] | None:
    for row in series:
        if int(row["chunks"]) == 1:
            return row
    return None


def draw_series(ax: plt.Axes, series: list[dict[str, str]], *, show_error: bool, annotate_speedup: bool) -> None:
    xs = [int(row["chunks"]) for row in series]
    ys = [qps(row) for row in series]
    yerr = [qps_std(row) for row in series]
    best = best_row(series)
    best_x = int(best["chunks"])
    best_y = qps(best)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.22, y_max * 0.055)

    ax.axvspan(best_x - 0.22, best_x + 0.22, color=BEST_RED, alpha=0.10, zorder=0)
    ax.errorbar(
        xs,
        ys,
        yerr=yerr if show_error else None,
        color=BLUE,
        marker="s",
        linewidth=2.5,
        markersize=11.5,
        capsize=3 if show_error else 0,
    )
    ax.scatter(
        [best_x],
        [best_y],
        color=BEST_RED,
        marker="*",
        edgecolor="white",
        linewidth=1.4,
        s=900,
        zorder=4,
    )
    baseline = chunk_one_row(series)
    if annotate_speedup and baseline is not None:
        speedup = best_y / qps(baseline)
        label_on_left = best_x >= xs[-2]
        ax.annotate(
            f"{speedup:.2f}x",
            xy=(best_x, best_y),
            xytext=(-12, 10) if label_on_left else (8, 10),
            textcoords="offset points",
            color="#000000",
            fontsize=plt.rcParams["font.size"] + 3,
            fontweight="bold",
            ha="right" if label_on_left else "left",
            va="bottom",
            zorder=5,
        )
    ax.set_ylim(y_min - y_pad * 0.35, y_max + y_pad)
    style_axis(ax)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_all_grid(
    rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
    datasets: list[str],
) -> list[Path]:
    set_rcparams(font_size)
    grouped = group_rows(rows)
    fig_height = 4.05 * len(datasets)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13.4, fig_height), sharey=False, squeeze=False)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, target in enumerate(TARGET_ORDER):
            ax = axes[row_idx][col_idx]
            series = grouped[(dataset, target)]
            draw_series(ax, series, show_error=False, annotate_speedup=True)
            ax.set_title(f"{DATASET_LABELS[dataset]} {TARGET_LABELS[target]}", pad=10)
            ax.set_xticks([int(row["chunks"]) for row in series])
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Number of Overlap Chunks")
            if col_idx == 0:
                ax.set_ylabel("Throughput")

    best_handle = Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor=BEST_RED,
        markeredgecolor="white",
        markeredgewidth=1.4,
        markersize=24,
        label="Best chunks",
    )
    for ax in axes.flat:
        ax.legend(handles=[best_handle], loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.945, bottom=0.12, wspace=0.20, hspace=0.58)
    dataset_part = "all" if "msmarco" in datasets else "lotte_hotpot"
    return save_figure(fig, output_dir, f"chimera_chunk_targets_qps_{dataset_part}_r090_r095_grid", formats, dpi)


def plot_focus_one_row(
    rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
    chunks: list[int],
    target: str,
) -> list[Path]:
    set_rcparams(font_size)
    grouped = group_rows(rows)
    focus = [
        ("lotte", chunks),
        ("hotpot", chunks),
        ("msmarco", chunks),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), sharey=False)

    best_handle = Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor=BEST_RED,
        markeredgecolor="white",
        markeredgewidth=1.4,
        markersize=30,
        label="Best chunks",
    )

    for ax, (dataset, chunk_filter) in zip(axes, focus):
        series = [row for row in grouped[(dataset, target)] if int(row["chunks"]) in set(chunk_filter)]
        draw_series(ax, series, show_error=False, annotate_speedup=True)
        ax.set_title(f"{DATASET_LABELS[dataset]} {TARGET_LABELS[target]}", pad=10)
        ax.set_xlabel("Number of Overlap Chunks")
        ax.set_ylabel("Throughput")
        ax.set_xticks(chunk_filter)
        ax.legend(handles=[best_handle], loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False)

    fig.subplots_adjust(left=0.070, right=0.975, top=0.80, bottom=0.16, wspace=0.34)
    target_part = "r" + target.replace(".", "")
    return save_figure(fig, output_dir, f"chimera_chunk_targets_qps_all_datasets_{target_part}_one_row", formats, dpi)


def plot_focus_stacked(
    rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
    chunks: list[int],
) -> list[Path]:
    set_rcparams(font_size)
    grouped = group_rows(rows)
    datasets = ["lotte", "hotpot", "msmarco"]
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.4), sharey=False, squeeze=False)

    best_handle = Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor=BEST_RED,
        markeredgecolor="white",
        markeredgewidth=1.4,
        markersize=30,
        label="Best chunks",
    )

    for row_idx, target in enumerate(TARGET_ORDER):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            series = [row for row in grouped[(dataset, target)] if int(row["chunks"]) in set(chunks)]
            draw_series(ax, series, show_error=False, annotate_speedup=True)
            ax.set_title(f"{DATASET_LABELS[dataset]} {TARGET_LABELS[target]}", pad=10)
            ax.set_xlabel("Number of Overlap Chunks")
            if col_idx == 0:
                ax.set_ylabel("Throughput")
            ax.set_xticks(chunks)
            ax.legend(handles=[best_handle], loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False)

    fig.subplots_adjust(left=0.070, right=0.975, top=0.90, bottom=0.09, wspace=0.34, hspace=0.48)
    return save_figure(fig, output_dir, "chimera_chunk_targets_qps_all_datasets_r090_r095_stacked", formats, dpi)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input or latest_required(
        repo_root,
        "profiling/chimera_chunk_targets",
        "chimera_chunk_targets_summary_*.csv",
    )
    rows = read_rows(input_path)
    chunk_set = set(args.chunks)
    rows = [row for row in rows if int(row["chunks"]) in chunk_set]
    grid_datasets = FULL_DATASET_ORDER if args.include_msmarco else DEFAULT_DATASET_ORDER
    outputs: list[Path] = []
    if args.plot in {"all", "both"}:
        outputs.extend(plot_all_grid(rows, args.output_dir, args.formats, args.dpi, args.font_size, grid_datasets))
    if args.plot in {"focus", "both"}:
        outputs.extend(
            plot_focus_one_row(
                rows,
                args.output_dir,
                args.formats,
                args.dpi,
                args.font_size,
                args.chunks,
                args.focus_target,
            )
        )
    if args.plot in {"stacked", "both"}:
        outputs.extend(plot_focus_stacked(rows, args.output_dir, args.formats, args.dpi, args.font_size, args.chunks))
    print(f"input={input_path}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
