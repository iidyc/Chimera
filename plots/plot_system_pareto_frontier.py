#!/usr/bin/env python3
"""Plot throughput/recall Pareto frontiers for PLAID, PLAID+, IGP, and Chimera."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from plot_paths import latest_required


DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}
TOPK_ORDER = ["10", "100"]
TOPK_LABELS = {
    "10": "Recall@10",
    "100": "Recall@100",
}
SYSTEM_ORDER = ["plaid", "plaid_plus", "igp", "chimera"]
SYSTEM_LABELS = {
    "plaid": "PLAID",
    "plaid_plus": "PLAID+ (GPU-resident Index)",
    "igp": "IGP (48 threads)",
    "chimera": "Chimera",
}
COLORS = {
    "plaid": "#4C78A8",
    "plaid_plus": "#B279A2",
    "igp": "#F58518",
    "chimera": "#54A24B",
}
MARKERS = {
    "plaid": "o",
    "plaid_plus": "s",
    "igp": "^",
    "chimera": "D",
}
LINESTYLES = {
    "plaid": "-",
    "plaid_plus": "-",
    "igp": "--",
    "chimera": "-",
}
LEGACY_CHIMERA_SYSTEM = "v" "8"
CHIMERA_SYSTEM_ALIASES = {"chimera", "gpu_search", LEGACY_CHIMERA_SYSTEM}


@dataclass(frozen=True)
class Point:
    system: str
    dataset: str
    topk: str
    recall: float
    qps: float
    label: str
    source: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot system Pareto frontiers over Recall@10 and Recall@100.")
    parser.add_argument("--plaid", type=Path, default=repo_root / "profiling" / "plaid" / "summary.csv")
    parser.add_argument("--plaid-plus", type=Path, default=repo_root / "profiling" / "plaid_plus" / "summary.csv")
    parser.add_argument("--igp", type=Path, default=repo_root / "profiling" / "igp" / "summary.csv")
    parser.add_argument(
        "--chimera",
        type=Path,
        default=None,
        help="Defaults to latest profiling/chimera_topk_baseline/chimera_topk_baseline_summary_*.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=27)
    parser.add_argument("--min-recall", type=float, default=0.825)
    parser.add_argument("--max-recall", type=float, default=0.975)
    return parser.parse_args()


def read_plaid_like(path: Path, system: str) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("k") not in TOPK_ORDER:
                continue
            points.append(
                Point(
                    system=system,
                    dataset=row["dataset"],
                    topk=row["k"],
                    recall=float(row["recall"]),
                    qps=float(row["qps"]),
                    label=row["label"],
                    source=row.get("source", str(path)),
                )
            )
    return points


def read_igp(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "val" or row.get("topk") not in TOPK_ORDER:
                continue
            if row.get("metric_name") != f"recall@{row['topk']}":
                continue
            points.append(
                Point(
                    system="igp",
                    dataset=row["dataset"],
                    topk=row["topk"],
                    recall=float(row["recall_mean"]),
                    qps=float(row["throughput_qps"]),
                    label=f"phi_pb={row['phi_pb']},phi_ref={row['phi_ref']}",
                    source=row.get("chunk_json", str(path)),
                )
            )
    return points


def read_chimera(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") not in CHIMERA_SYSTEM_ALIASES or row.get("k") != "100":
                continue
            for topk, recall_column in (("10", "recall_at_10_mean"), ("100", "recall_at_100_mean")):
                points.append(
                    Point(
                        system="chimera",
                        dataset=row["dataset"],
                        topk=topk,
                        recall=float(row[recall_column]),
                        qps=float(row["qps_mean"]),
                        label=f"{row['label']} (k=100 run)",
                        source=str(path),
                    )
                )
    return points


def pareto_frontier(points: list[Point]) -> list[Point]:
    best_by_recall: dict[float, Point] = {}
    for point in points:
        old = best_by_recall.get(point.recall)
        if old is None or point.qps > old.qps:
            best_by_recall[point.recall] = point

    candidates = sorted(best_by_recall.values(), key=lambda point: (-point.recall, -point.qps))
    frontier_desc: list[Point] = []
    best_qps = -1.0
    for point in candidates:
        if point.qps > best_qps:
            frontier_desc.append(point)
            best_qps = point.qps
    return sorted(frontier_desc, key=lambda point: point.recall)


def set_rcparams(font_size: float) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.titleweight": "normal",
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_frontier_csv(frontiers: dict[tuple[str, str, str], list[Point]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "system_pareto_frontier_topk10_topk100_qps_selected.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "topk", "system", "recall", "qps", "label", "source"])
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for topk in TOPK_ORDER:
                for system in SYSTEM_ORDER:
                    for point in frontiers.get((dataset, topk, system), []):
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "topk": topk,
                                "system": system,
                                "recall": f"{point.recall:.6f}",
                                "qps": f"{point.qps:.6f}",
                                "label": point.label,
                                "source": point.source,
                            }
                        )
    return path


def draw_series(ax: plt.Axes, series: list[Point], system: str) -> None:
    xs = [point.recall for point in series]
    ys = [point.qps for point in series]
    ax.plot(
        xs,
        ys,
        label=SYSTEM_LABELS[system],
        color=COLORS[system],
        marker=MARKERS[system],
        linestyle=LINESTYLES[system],
        linewidth=2.1,
        markersize=16.0,
        markeredgecolor="#222222",
        markeredgewidth=0.9,
    )


def plot(
    frontiers: dict[tuple[str, str, str], list[Point]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
    min_recall: float,
    max_recall: float,
) -> list[Path]:
    set_rcparams(font_size)
    fig, axes = plt.subplots(2, 3, figsize=(24.0, 8.0), sharey=True, squeeze=False)
    visible_ymax = max(
        (
            point.qps
            for series in frontiers.values()
            for point in series
            if min_recall <= point.recall <= max_recall
        ),
        default=1.0,
    )
    ylimit = visible_ymax * 1.12

    for row_idx, topk in enumerate(TOPK_ORDER):
        for col_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[row_idx][col_idx]
            for system in SYSTEM_ORDER:
                series = [
                    point
                    for point in frontiers.get((dataset, topk, system), [])
                    if min_recall <= point.recall <= max_recall
                ]
                if not series:
                    continue
                draw_series(ax, series, system)

            ax.set_title(DATASET_LABELS[dataset] if row_idx == 0 else "", pad=10, fontweight="normal")
            ax.set_xlabel(TOPK_LABELS[topk])
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="major", axis="both", color="#d0d0d0", linewidth=0.85, alpha=0.9)
            ax.grid(which="minor", axis="both", color="#d0d0d0", linewidth=0.85, alpha=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.set_xlim(min_recall, max_recall)
            ax.set_ylim(0, ylimit)
            ax.set_yticks([0, 250, 500, 750])
            if col_idx == 0:
                ax.set_ylabel("Throughput")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.532, 0.000),
        ncol=len(SYSTEM_ORDER),
        columnspacing=1.6,
        handlelength=1.8,
        markerscale=0.95,
    )
    fig.subplots_adjust(left=0.060, right=0.99, top=0.92, bottom=0.205, wspace=0.28, hspace=0.38)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        path = output_dir / f"system_pareto_frontier_topk10_topk100_qps.{suffix}"
        fig.savefig(path, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.chimera is None:
        args.chimera = latest_required(
            repo_root,
            "profiling/chimera_topk_baseline",
            "chimera_topk_baseline_summary_*.csv",
        )
    points = []
    points.extend(read_plaid_like(args.plaid, "plaid"))
    points.extend(read_plaid_like(args.plaid_plus, "plaid_plus"))
    points.extend(read_igp(args.igp))
    points.extend(read_chimera(args.chimera))

    frontiers: dict[tuple[str, str, str], list[Point]] = {}
    for dataset in DATASET_ORDER:
        for topk in TOPK_ORDER:
            for system in SYSTEM_ORDER:
                series = [
                    point
                    for point in points
                    if point.dataset == dataset and point.topk == topk and point.system == system
                ]
                frontiers[(dataset, topk, system)] = pareto_frontier(series)

    selected_csv = write_frontier_csv(frontiers, args.output_dir)
    outputs = plot(
        frontiers,
        args.output_dir,
        args.formats,
        args.dpi,
        args.font_size,
        args.min_recall,
        args.max_recall,
    )

    print(f"plaid={args.plaid}")
    print(f"plaid_plus={args.plaid_plus}")
    print(f"igp={args.igp}")
    print(f"chimera={args.chimera}")
    print(selected_csv)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
