#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = {"lotte": 0, "hotpot": 1, "msmarco": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stacked bar plots from cpu_search_v3 profile summary CSV output."
        )
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        required=True,
        help="CSV produced by parse_cpu_search_v3_profile_logs.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for PNG/PDF outputs.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["k"]), DATASET_ORDER.get(row["dataset"], 99)))
    return rows


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.profile_csv)
    labels = [
        f"{row['dataset']}\n$k$={row['k']}\n{row['selection_label']}"
        for row in rows
    ]
    x = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    summary_components = [
        ("avg_query_setup_ms", "Query setup", "#B8C4BB"),
        ("avg_stage1_ms", "Stage 1", "#3A7D44"),
        ("avg_stage2_ms", "Stage 2", "#2A6F97"),
        ("avg_stage3_ms", "Stage 3", "#D17B0F"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(rows))
    for key, label, color in summary_components:
        values = np.array([float(row[key]) for row in rows])
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
        bottom += values

    for idx, row in enumerate(rows):
        stage12_pct = float(row["stage12_pct_end_to_end"])
        ax.text(
            x[idx],
            bottom[idx] + 0.12,
            f"S1+S2 {stage12_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Average time per query (ms)")
    ax.set_title("cpu_search_v3 average end-to-end breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, args.output_prefix.with_name(args.output_prefix.name + "_summary"))

    detail_specs = [
        (
            "Stage 1 detail",
            [
                ("avg_stage1_probe_ms", "HNSW probe", "#355070"),
                ("avg_stage1_prepare_ms", "Stage 1 prepare", "#6D597A"),
                ("avg_stage1_scan_ms", "Stage 1 scan", "#B56576"),
                ("avg_stage1_reduce_ms", "Stage 1 reduce", "#E56B6F"),
                ("avg_stage1_cleanup_ms", "Stage 1 cleanup", "#EAAC8B"),
            ],
        ),
        (
            "Stage 2 detail",
            [
                ("avg_stage2_lut_ms", "LUT build", "#1B4332"),
                ("avg_stage2_score_docs_ms", "Doc scoring", "#2D6A4F"),
                ("avg_stage2_select_topk_ms", "Select top-k", "#40916C"),
                ("avg_stage2_materialize_ms", "Materialize", "#74C69D"),
            ],
        ),
        (
            "Stage 3 detail",
            [
                ("avg_stage3_prepare_ms", "Stage 3 prepare", "#9C6644"),
                ("avg_stage3_score_docs_ms", "Stage 3 score docs", "#D4A373"),
                ("avg_stage3_select_topk_ms", "Stage 3 select top-k", "#E9C46A"),
            ],
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (title, components) in zip(axes, detail_specs):
        bottom = np.zeros(len(rows))
        for key, label, color in components:
            values = np.array([float(row[key]) for row in rows])
            ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
            bottom += values
        ax.set_title(title)
        ax.set_ylabel("ms")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=len(components), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()
    save_figure(fig, args.output_prefix.with_name(args.output_prefix.name + "_detail"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
