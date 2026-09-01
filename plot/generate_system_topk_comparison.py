#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


DATASETS = ["lotte", "hotpot", "msmarco"]
TOPKS = [10, 100]
DATASET_LABELS = {
    "hotpot": "HotpotQA",
    "lotte": "LoTTE",
    "msmarco": "MSMARCO",
}

SYSTEM_STYLES = {
    # Palette aligned with the PLAID/PLAID+/Chimera latency-breakdown figure.
    "v6_lite": {"label": "Chimera", "color": "#2B78A6", "marker": "o"},
    "plaid": {"label": "PLAID", "color": "#E84D4D", "marker": "s"},
    "plaid_plus": {"label": "PLAID+", "color": "#59A14F", "marker": "^"},
    "cpu_search_v3": {"label": "Chimera-CPU", "color": "#F58518", "marker": "X"},
    "igp": {"label": "IGP", "color": "#B279A2", "marker": "D"},
}


@dataclass(frozen=True)
class Point:
    system: str
    dataset: str
    topk: int
    recall: float
    qps: float
    label: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot qps-vs-recall curves for v6_lite, PLAID, PLAID+, CPU Search v3, and IGP "
            "across Hotpot, LoTTE, and MSMARCO for topk=10 and topk=100."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling output root. Default: profiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot"),
        help="Directory where output files are written. Default: plot.",
    )
    parser.add_argument(
        "--output-stem",
        default="system_topk_comparison",
        help="Output filename stem. Default: system_topk_comparison.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.825,
        help="Drop plotted points at or below this recall. Default: 0.825.",
    )
    parser.add_argument(
        "--all-points",
        action="store_true",
        help="Plot all rows instead of each system's Pareto frontier.",
    )
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Include the Chimera-CPU series from profiling/cpu_search_v3/summary_t32.csv.",
    )
    return parser.parse_args()


def setup_matplotlib_cache(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    fontconfig_dir = cache_dir / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    os.environ["FONTCONFIG_CACHE"] = str(fontconfig_dir.resolve())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latest_csv(directory: Path, pattern: str = "benchmark_results*.csv") -> Path | None:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def float_value(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return float(value)
    return None


def load_benchmark_points(
    path: Path | None,
    system: str,
    dataset: str,
    topk: int,
) -> list[Point]:
    if path is None:
        return []

    points: list[Point] = []
    for idx, row in enumerate(read_csv_rows(path)):
        recall = float_value(row, "recall", "recall_mean")
        qps = float_value(row, "qps", "throughput_qps")
        if recall is None or qps is None:
            continue

        row_dataset = row.get("dataset", dataset).strip() or dataset
        row_topk = row.get("k", row.get("topk", str(topk))).strip() or str(topk)
        if row_dataset != dataset or int(row_topk) != topk:
            continue

        points.append(
            Point(
                system=system,
                dataset=dataset,
                topk=topk,
                recall=recall,
                qps=qps,
                label=row.get("label", row.get("phi_ref", str(idx))).strip(),
                source=path,
            )
        )
    return points


def load_benchmark_points_from_paths(
    paths: list[Path],
    system: str,
    dataset: str,
    topk: int,
) -> list[Point]:
    points: list[Point] = []
    seen: set[tuple[str, float, float]] = set()
    for path in paths:
        for point in load_benchmark_points(path, system, dataset, topk):
            key = (point.label, point.recall, point.qps)
            if key in seen:
                continue
            seen.add(key)
            points.append(point)
    return points


def load_search_py_points(
    paths: list[Path],
    system: str,
    dataset: str,
    topk: int,
) -> list[Point]:
    points: list[Point] = []
    seen: set[tuple[float, float]] = set()
    for path in paths:
        for idx, row in enumerate(read_csv_rows(path)):
            recall = float_value(row, "recall")
            qps = float_value(row, "qps")
            if recall is None or qps is None:
                continue
            key = (recall, qps)
            if key in seen:
                continue
            seen.add(key)
            points.append(
                Point(
                    system=system,
                    dataset=dataset,
                    topk=topk,
                    recall=recall,
                    qps=qps,
                    label=f"search_py_{idx}",
                    source=path,
                )
            )
    return points


def v6_lite_paths(profiling_root: Path, dataset: str, topk: int) -> list[Path]:
    paths: list[Path] = []
    directory = (
        profiling_root
        / "experiments"
        / "gpu_main_topk_vram16_lite_c4"
        / f"k{topk}"
        / dataset
        / "gpu_search_v6_lite_vram16_c4"
    )
    base_path = latest_csv(directory)
    if base_path is not None:
        paths.append(base_path)

    high_recall_dir = (
        profiling_root
        / "experiments"
        / "gpu_main_topk_vram16_lite_hotpot_highrecall"
        / f"k{topk}"
        / dataset
        / "gpu_search_v6_lite_vram16_hotpot_highrecall"
    )
    high_recall_path = latest_csv(high_recall_dir)
    if high_recall_path is not None:
        paths.append(high_recall_path)

    return paths


def plaid_path(profiling_root: Path, dataset: str, topk: int) -> Path | None:
    candidates = [
        profiling_root / "experiments" / "plaid_main_topk_local" / f"k{topk}" / dataset / "plaid" / "benchmark_results.csv",
        profiling_root / "experiments" / "plaid_local_topk" / f"k{topk}" / dataset / "plaid" / "benchmark_results.csv",
        profiling_root / "experiments" / "plaid_main_topk" / f"k{topk}" / dataset / "plaid" / "benchmark_results.csv",
    ]
    for path in candidates:
        if len(read_csv_rows(path)) > 0:
            return path
    return None


def plaid_search_paths(profiling_root: Path, dataset: str, topk: int) -> list[Path]:
    if dataset != "msmarco" or topk != 10:
        return []
    return [
        profiling_root / "experiments" / "plaid_main_topk_local" / "k10" / "msmarco" / "plaid" / "search_py_output.csv",
        profiling_root / "experiments" / "plaid_main_topk_remaining" / "k10" / "msmarco" / "plaid" / "search_py_output.csv",
    ]


def plaid_plus_path(profiling_root: Path, dataset: str, topk: int) -> Path | None:
    path = (
        profiling_root
        / "experiments"
        / "plaid_main_topk_r5"
        / f"k{topk}"
        / dataset
        / "plaid_gpu_resident"
        / "benchmark_results.csv"
    )
    return path if len(read_csv_rows(path)) > 0 else None


def cpu_search_v3_path(profiling_root: Path) -> Path:
    return profiling_root / "cpu_search_v3" / "summary_t32.csv"


def load_igp_points(
    profiling_root: Path,
    dataset: str,
    topk: int,
) -> list[Point]:
    candidates = [
        profiling_root / "igp" / "summary.csv",
        profiling_root / "igp" / "summary_t32.csv",
        profiling_root / "igp" / "summary_t36_2bit.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    points: list[Point] = []
    for idx, row in enumerate(read_csv_rows(path)):
        if row.get("dataset", "").strip() != dataset:
            continue
        if int(row.get("topk", "0")) != topk:
            continue
        recall = float_value(row, "recall_mean", "recall")
        qps = float_value(row, "qps", "throughput_qps")
        if recall is None or qps is None:
            continue
        points.append(
            Point(
                system="igp",
                dataset=dataset,
                topk=topk,
                recall=recall,
                qps=qps,
                label=f"phi{row.get('phi_pb', '')}_{row.get('phi_ref', idx)}",
                source=path,
            )
        )
    return points


def pareto_frontier(points: list[Point]) -> list[Point]:
    frontier: list[Point] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if (
                other.qps >= point.qps
                and other.recall >= point.recall
                and (other.qps > point.qps or other.recall > point.recall)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: (item.recall, item.qps))


def load_points(
    profiling_root: Path,
    dataset: str,
    topk: int,
    min_recall: float,
    use_frontier: bool,
    include_cpu: bool,
) -> dict[str, list[Point]]:
    points_by_system = {
        "v6_lite": load_benchmark_points_from_paths(v6_lite_paths(profiling_root, dataset, topk), "v6_lite", dataset, topk),
        "plaid": load_benchmark_points(plaid_path(profiling_root, dataset, topk), "plaid", dataset, topk),
        "plaid_plus": load_benchmark_points(plaid_plus_path(profiling_root, dataset, topk), "plaid_plus", dataset, topk),
        "igp": load_igp_points(profiling_root, dataset, topk),
    }
    if include_cpu:
        points_by_system["cpu_search_v3"] = load_benchmark_points(
            cpu_search_v3_path(profiling_root),
            "cpu_search_v3",
            dataset,
            topk,
        )

    if dataset == "msmarco" and topk == 10 and not points_by_system["plaid"]:
        points_by_system["plaid"] = load_search_py_points(
            plaid_search_paths(profiling_root, dataset, topk),
            "plaid",
            dataset,
            topk,
        )

    filtered: dict[str, list[Point]] = {}
    for system, points in points_by_system.items():
        kept = [point for point in points if point.recall > min_recall]
        filtered[system] = pareto_frontier(kept) if use_frontier else sorted(kept, key=lambda item: (item.recall, item.qps))
    return filtered


def write_source_points(path: Path, all_points: list[Point]) -> None:
    fieldnames = ["system", "dataset", "topk", "recall", "qps", "label", "source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in all_points:
            writer.writerow(
                {
                    "system": SYSTEM_STYLES[point.system]["label"],
                    "dataset": point.dataset,
                    "topk": point.topk,
                    "recall": f"{point.recall:.9g}",
                    "qps": f"{point.qps:.9g}",
                    "label": point.label,
                    "source": str(point.source),
                }
            )


def render_figure(
    profiling_root: Path,
    output_dir: Path,
    output_stem: str,
    min_recall: float,
    use_frontier: bool,
    include_cpu: bool,
) -> tuple[Path, Path, Path]:
    setup_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(TOPKS),
        ncols=len(DATASETS),
        figsize=(17.5, 7.0),
        sharex=False,
        sharey=False,
    )

    plotted_points: list[Point] = []
    missing: list[str] = []
    panel_points: dict[tuple[str, int], dict[str, list[Point]]] = {}
    y_max_by_topk: dict[int, float] = {}
    x_limits_by_topk: dict[int, tuple[float, float]] = {}

    for topk in TOPKS:
        row_qps_values: list[float] = []
        row_recall_values: list[float] = []
        for dataset in DATASETS:
            points_by_system = load_points(
                profiling_root,
                dataset,
                topk,
                min_recall,
                use_frontier,
                include_cpu,
            )
            panel_points[(dataset, topk)] = points_by_system
            row_qps_values.extend(
                point.qps
                for points in points_by_system.values()
                for point in points
            )
            row_recall_values.extend(
                point.recall
                for points in points_by_system.values()
                for point in points
            )
        y_max_by_topk[topk] = max(row_qps_values) if row_qps_values else 1.0
        if row_recall_values:
            x_limits_by_topk[topk] = (
                max(0.0, min_recall, min(row_recall_values) - 0.02),
                min(1.0, max(row_recall_values) + 0.01),
            )
        else:
            x_limits_by_topk[topk] = (max(0.0, min_recall), 1.0)

    for row_idx, topk in enumerate(TOPKS):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            points_by_system = panel_points[(dataset, topk)]

            subplot_points = [point for points in points_by_system.values() for point in points]
            plotted_points.extend(subplot_points)

            for system, style in SYSTEM_STYLES.items():
                if system not in points_by_system:
                    continue
                points = points_by_system[system]
                if not points:
                    missing.append(f"{style['label']} {dataset} k={topk}")
                    continue
                ax.plot(
                    [point.recall for point in points],
                    [point.qps for point in points],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.2,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    label=style["label"],
                )

            ax.set_title(f"{DATASET_LABELS[dataset]} (TopK = {topk})", pad=10, fontsize=22)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if col_idx != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda value, _: "" if abs(value) < 1e-9 else f"{value:g}")
                )
            ax.grid(True, color="#BFBFBF", linestyle="--", alpha=0.45, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

            if subplot_points:
                left, right = x_limits_by_topk[topk]
                ax.set_xlim(left, right)
                ax.set_ylim(0.0, y_max_by_topk[topk] * 1.10)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#555555",
                )

    fig.tight_layout(rect=(0.065, 0.095, 0.99, 0.86), w_pad=0.9, h_pad=2.0)

    all_axes_left = min(ax.get_position().x0 for row in axes for ax in row)
    all_axes_bottom = min(ax.get_position().y0 for row in axes for ax in row)
    all_axes_top = max(ax.get_position().y1 for row in axes for ax in row)
    middle_col_center = (
        axes[0][1].get_position().x0 + axes[0][1].get_position().x1
    ) / 2

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=1.8,
        columnspacing=1.15,
        labelspacing=0.8,
        markerscale=1.6,
    )

    fig.text(
        max(0.018, all_axes_left - 0.055),
        (all_axes_bottom + all_axes_top) / 2,
        "QPS",
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
    )
    fig.text(middle_col_center, 0.055, "Recall", ha="center", va="center", fontsize=22)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    csv_path = output_dir / f"{output_stem}_points.csv"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_source_points(csv_path, plotted_points)

    if missing:
        print("missing_series=" + "; ".join(missing))
    return png_path, pdf_path, csv_path


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path, pdf_path, csv_path = render_figure(
        profiling_root=profiling_root,
        output_dir=output_dir,
        output_stem=args.output_stem,
        min_recall=args.min_recall,
        use_frontier=not args.all_points,
        include_cpu=args.include_cpu,
    )
    print(f"png={png_path.resolve()}")
    print(f"pdf={pdf_path.resolve()}")
    print(f"points={csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
