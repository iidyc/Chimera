#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COMPONENTS = [
    ("Candidate Generation", "candidate_generation_ms", "#4C78A8"),
    ("Candidate Refine", "refine_ms", "#54A24B"),
    ("Scoring", "rerank_ms", "#E45756"),
]


@dataclass
class CpuBreakdown:
    label: str
    threads: int
    target_recall: float
    recall: float
    avg_end_to_end_ms: float
    query_setup_ms: float
    probe_ms: float
    scan_ms: float
    refine_ms: float
    rerank_ms: float

    def value(self, key: str) -> float:
        return getattr(self, key)

    @property
    def candidate_generation_ms(self) -> float:
        return self.probe_ms + self.scan_ms


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a LoTTE cpu_search_v3 32-thread recall@0.9 latency-breakdown plot."
    )
    parser.add_argument(
        "--breakdown-csv",
        type=Path,
        default=repo_root / "profiling" / "cpu_search_v3" / "breakdown.csv",
    )
    parser.add_argument("--dataset", default="lotte")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "plot" / "cpu_search_v3_lotte_recall90_t32_breakdown",
        help="Output path prefix without file extension.",
    )
    return parser.parse_args()


def load_breakdown(args: argparse.Namespace) -> CpuBreakdown:
    target_recall = f"{args.target_recall:.2f}"
    with args.breakdown_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != args.dataset:
                continue
            if int(row["threads"]) != args.threads:
                continue
            if int(row["k"]) != args.topk:
                continue
            if row["target_recall"] != target_recall:
                continue
            return CpuBreakdown(
                label=row["selection_label"],
                threads=int(row["threads"]),
                target_recall=float(row["target_recall"]),
                recall=float(row["recall"]),
                avg_end_to_end_ms=float(row["avg_end_to_end_ms"]),
                query_setup_ms=float(row["avg_query_setup_ms"]),
                probe_ms=float(row["avg_query_setup_ms"]) + float(row["avg_stage1_probe_ms"]),
                scan_ms=(
                    float(row["avg_stage1_prepare_ms"])
                    + float(row["avg_stage1_scan_ms"])
                    + float(row["avg_stage1_reduce_ms"])
                    + float(row["avg_stage1_cleanup_ms"])
                ),
                refine_ms=float(row["avg_stage2_ms"]),
                rerank_ms=float(row["avg_stage3_ms"]),
            )
    raise ValueError(
        f"no row found for dataset={args.dataset}, k={args.topk}, "
        f"target_recall={target_recall}, threads={args.threads}"
    )


def add_segment_label(ax: plt.Axes, left: float, width: float, label: str, pct: float) -> None:
    if width < 7.0:
        return
    ax.text(
        left + width / 2.0,
        0,
        f"{label}\n{pct:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
        color="white",
        fontweight="bold",
    )


def add_total_label(ax: plt.Axes, total_ms: float) -> None:
    ax.text(
        101.0,
        0,
        f"Total: {total_ms:.1f} ms",
        ha="left",
        va="center",
        fontsize=15,
        color="black",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def plot(output_root: Path, breakdown: CpuBreakdown) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.2))
    left = 0.0
    for label, key, color in COMPONENTS:
        value = breakdown.value(key)
        if value <= 0.0:
            continue
        pct = value / breakdown.avg_end_to_end_ms * 100.0
        ax.barh(
            0,
            pct,
            left=left,
            height=0.58,
            color=color,
            edgecolor="#000000",
            linewidth=1.5,
        )
        add_segment_label(ax, left, pct, label, pct)
        left += pct

    ax.set_xlabel("Percentage of End-to-End Latency (Average)", fontsize=22, labelpad=10)
    ax.set_yticks([0])
    ax.set_yticklabels(["CPU"], fontsize=21)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", pad=1, length=0)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.55, 0.62)
    ax.set_title(
        "LoTTE: Latency Breakdown for CPU (Recall@100 >= 0.9)",
        fontsize=23,
        pad=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.28, top=0.80)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    breakdown = load_breakdown(args)
    plot(args.output_root, breakdown)
    print(f"[done] plot_png={args.output_root.with_suffix('.png')}")
    print(f"[done] plot_pdf={args.output_root.with_suffix('.pdf')}")
    print(
        f"[summary] dataset={args.dataset} k={args.topk} threads={breakdown.threads} "
        f"target_recall={breakdown.target_recall:.2f} recall={breakdown.recall:.6f} "
        f"avg_end_to_end_ms={breakdown.avg_end_to_end_ms:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
