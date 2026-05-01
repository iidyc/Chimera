#!/usr/bin/env python3
"""Plot chimera configuration variant performance from unity benchmark data."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_paths import latest_required


DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
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
SYSTEM_ORDER = ["chimera", "chimera_nolut", "chimera_nosum", "chimera_nolut_nosum"]
SYSTEM_LABELS = {
    "chimera": "Chimera",
    "chimera_nolut": "w/o LUT",
    "chimera_nosum": "w/o Score",
    "chimera_nolut_nosum": "w/o LUT\nw/o Score",
}
COLORS = {
    "chimera": "#4C78A8",
    "chimera_nolut": "#B279A2",
    "chimera_nosum": "#F58518",
    "chimera_nolut_nosum": "#54A24B",
}
HATCHES = {
    "chimera": "..",
    "chimera_nolut": "xx",
    "chimera_nosum": "",
    "chimera_nolut_nosum": "oo",
}
SYSTEM_ALIASES = {
    "chimera": "chimera",
    "gpu_search": "chimera",
    "v" "8": "chimera",
    "chimera_nolut": "chimera_nolut",
    "gpu_search_nolut": "chimera_nolut",
    "v" "8_nolut": "chimera_nolut",
    "chimera_nosum": "chimera_nosum",
    "gpu_search_nosum": "chimera_nosum",
    "v" "8_nosum": "chimera_nosum",
    "chimera_nolut_nosum": "chimera_nolut_nosum",
    "gpu_search_nolut_nosum": "chimera_nolut_nosum",
    "v" "8_nolut_nosum": "chimera_nolut_nosum",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot chimera variant target throughput.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Chimera variant target summary CSV. Defaults to latest profiling/chimera_variant_targets summary.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=21)
    return parser.parse_args()


def latest_summary(repo_root: Path) -> Path:
    return latest_required(
        repo_root,
        "profiling/chimera_variant_targets",
        "chimera_variant_targets_summary_*.csv",
    )


def read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    def numeric(row: dict[str, str], *names: str) -> float:
        for name in names:
            value = row.get(name, "")
            if value != "":
                return float(value)
        raise KeyError(f"none of these columns are present with values: {names}")

    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            system = SYSTEM_ALIASES.get(row["system"])
            if system is None:
                continue
            target = f"{float(row['target_recall']):.2f}"
            key = (row["dataset"], target, system)
            rows[key] = {
                "qps": numeric(row, "qps", "qps_mean"),
                "latency_ms": numeric(row, "avg_latency_ms", "avg_latency_ms_mean"),
                "recall_at_100": numeric(row, "recall_at_100", "recall_at_100_mean"),
                "gpu_mem_peak_mib": next(
                    (
                        float(row[name])
                        for name in ("gpu_mem_peak_mib", "gpu_mem_peak_mib_mean", "gpu_mem_peak_mib_max")
                        if row.get(name, "") != ""
                    ),
                    math.nan,
                ),
            }
    return rows


def plot(
    rows: dict[tuple[str, str, str], dict[str, float]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
    metric: str,
    ylabel: str,
    output_stem: str,
    baseline_system: str,
    highlight_system: str,
    share_row_y: bool = False,
) -> list[Path]:
    missing = [
        (dataset, target, system)
        for target in TARGET_ORDER
        for dataset in DATASET_ORDER
        for system in SYSTEM_ORDER
        if (dataset, target, system) not in rows
    ]
    if missing:
        raise ValueError(f"missing rows: {missing}")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size + 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(13.4, 13.2), sharey=False)
    x_positions = [idx * 1.55 for idx in range(len(SYSTEM_ORDER))]
    bar_colors = [COLORS[system] for system in SYSTEM_ORDER]
    bar_hatches = [HATCHES[system] for system in SYSTEM_ORDER]
    bar_labels = [SYSTEM_LABELS[system] for system in SYSTEM_ORDER]

    for row_idx, dataset in enumerate(DATASET_ORDER):
        row_ymax = None
        if share_row_y:
            row_values = [
                rows[(dataset, target, system)][metric]
                for target in TARGET_ORDER
                for system in SYSTEM_ORDER
            ]
            row_ymax = max(row_values) * 1.28
        for col_idx, target in enumerate(TARGET_ORDER):
            ax = axes[row_idx][col_idx]
            values = [rows[(dataset, target, system)][metric] for system in SYSTEM_ORDER]
            bars = ax.bar(
                x_positions,
                values,
                color=bar_colors,
                edgecolor="#222222",
                linewidth=0.9,
                width=0.84,
            )
            for bar, hatch in zip(bars, bar_hatches):
                bar.set_hatch(hatch)
            baseline = rows[(dataset, target, baseline_system)][metric]
            for bar, value, system in zip(bars, values, SYSTEM_ORDER):
                ratio = value / baseline
                label = "1x" if system == baseline_system else f"{ratio:.2f}x"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=font_size if system == highlight_system else font_size - 1,
                    fontweight="bold" if system == highlight_system else "normal",
                    color="#222222",
                    rotation=0,
                    clip_on=False,
                )
            ax.set_title(f"{DATASET_LABELS[dataset]} {TARGET_LABELS[target]}", fontweight="normal", pad=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(bar_labels, rotation=0, ha="center")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.set_ylim(0, row_ymax if row_ymax is not None else max(values) * 1.28)
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[system], edgecolor="#222222", hatch=HATCHES[system])
        for system in SYSTEM_ORDER
    ]
    fig.legend(
        handles,
        bar_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(SYSTEM_ORDER),
        columnspacing=1.6,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.955, bottom=0.16, wspace=0.16, hspace=0.66)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        out = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(out, dpi=dpi)
        outputs.append(out)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input or latest_summary(repo_root)
    rows = read_rows(input_path)
    outputs = []
    outputs.extend(
        plot(
            rows,
            args.output_dir,
            args.formats,
            args.dpi,
            args.font_size,
            metric="qps",
            ylabel="Throughput",
            output_stem="chimera_variant_targets_qps",
            baseline_system="chimera_nolut_nosum",
            highlight_system="chimera",
        )
    )
    outputs.extend(
        plot(
            rows,
            args.output_dir,
            args.formats,
            args.dpi,
            args.font_size,
            metric="latency_ms",
            ylabel="Latency (ms)",
            output_stem="chimera_variant_targets_latency",
            baseline_system="chimera",
            highlight_system="chimera_nolut_nosum",
            share_row_y=True,
        )
    )
    print(f"input={input_path}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
