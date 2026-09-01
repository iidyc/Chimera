#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


DATASETS = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "hotpot": "HotpotQA",
    "lotte": "LoTTE",
    "msmarco": "MSMARCO",
}
TOPK = 100

SYSTEM_STYLES = {
    "cpu": {"label": "CPU", "color": "#F58518", "marker": "X"},
    "v6_lite": {"label": "Chimera", "color": "#2B78A6", "marker": "o"},
    "v7": {"label": "Chimera+", "color": "#59A14F", "marker": "^"},
}


@dataclass(frozen=True)
class Point:
    system: str
    dataset: str
    recall: float
    qps: float
    label: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TopK=100 Chimera-CPU vs Chimera-v6_lite vs Chimera-v7."
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
        default="chimera_topk100_comparison",
        help="Output filename stem. Default: chimera_topk100_comparison.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.775,
        help="Drop plotted points below this recall. Default: 0.775.",
    )
    parser.add_argument(
        "--all-points",
        action="store_true",
        help="Plot all rows instead of each system's Pareto frontier.",
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
) -> list[Point]:
    if path is None:
        return []

    points: list[Point] = []
    for idx, row in enumerate(read_csv_rows(path)):
        row_dataset = row.get("dataset", dataset).strip() or dataset
        row_topk = row.get("k", row.get("topk", str(TOPK))).strip() or str(TOPK)
        if row_dataset != dataset or int(row_topk) != TOPK:
            continue

        recall = float_value(row, "recall", "recall_mean")
        qps = float_value(row, "qps", "throughput_qps")
        if recall is None or qps is None:
            continue

        points.append(
            Point(
                system=system,
                dataset=dataset,
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
) -> list[Point]:
    points: list[Point] = []
    seen: set[tuple[str, float, float]] = set()
    for path in paths:
        for point in load_benchmark_points(path, system, dataset):
            key = (point.label, point.recall, point.qps)
            if key in seen:
                continue
            seen.add(key)
            points.append(point)
    return points


def v6_lite_paths(profiling_root: Path, dataset: str) -> list[Path]:
    paths: list[Path] = []
    base_dir = (
        profiling_root
        / "experiments"
        / "gpu_main_topk_vram16_lite_c4"
        / f"k{TOPK}"
        / dataset
        / "gpu_search_v6_lite_vram16_c4"
    )
    base_path = latest_csv(base_dir)
    if base_path is not None:
        paths.append(base_path)

    high_recall_dir = (
        profiling_root
        / "experiments"
        / "gpu_main_topk_vram16_lite_hotpot_highrecall"
        / f"k{TOPK}"
        / dataset
        / "gpu_search_v6_lite_vram16_hotpot_highrecall"
    )
    high_recall_path = latest_csv(high_recall_dir)
    if high_recall_path is not None:
        paths.append(high_recall_path)

    return paths


def v7_path(profiling_root: Path, dataset: str) -> Path | None:
    return latest_csv(profiling_root / dataset / "gpu_search_v7")


def cpu_path(profiling_root: Path) -> Path:
    return profiling_root / "cpu_search_v3" / "summary_t32.csv"


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
    min_recall: float,
    use_frontier: bool,
) -> dict[str, list[Point]]:
    points_by_system = {
        "cpu": load_benchmark_points(cpu_path(profiling_root), "cpu", dataset),
        "v6_lite": load_benchmark_points_from_paths(v6_lite_paths(profiling_root, dataset), "v6_lite", dataset),
        "v7": load_benchmark_points(v7_path(profiling_root, dataset), "v7", dataset),
    }

    filtered: dict[str, list[Point]] = {}
    for system, points in points_by_system.items():
        kept = [point for point in points if point.recall >= min_recall]
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
                    "topk": TOPK,
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
) -> tuple[Path, Path, Path]:
    setup_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        }
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(DATASETS),
        figsize=(20.0, 5.0),
        sharex=False,
        sharey=False,
    )

    points_by_panel: dict[str, dict[str, list[Point]]] = {}
    x_limits_by_dataset: dict[str, tuple[float, float]] = {}
    all_points_for_scale: list[Point] = []
    missing: list[str] = []

    for dataset in DATASETS:
        points_by_system = load_points(profiling_root, dataset, min_recall, use_frontier)
        v6_points = points_by_system["v6_lite"]
        if v6_points:
            max_v6_recall = max(point.recall for point in v6_points)
        else:
            max_v6_recall = min_recall
        points_by_system = {
            system: [point for point in points if point.recall <= max_v6_recall + 1e-12]
            for system, points in points_by_system.items()
        }
        points_by_panel[dataset] = points_by_system
        panel_points = [point for points in points_by_system.values() for point in points]
        all_points_for_scale.extend(panel_points)
        if panel_points:
            x_limits_by_dataset[dataset] = (
                max(0.0, min_recall, min(point.recall for point in panel_points) - 0.02),
                min(1.0, max_v6_recall + 0.005),
            )
        else:
            x_limits_by_dataset[dataset] = (max(0.0, min_recall), min(1.0, max_v6_recall + 0.005))

    if all_points_for_scale:
        ylim = (0.0, max(point.qps for point in all_points_for_scale) * 1.10)
    else:
        ylim = (0.0, 1.0)

    plotted_points: list[Point] = []
    for col_idx, dataset in enumerate(DATASETS):
        ax = axes[col_idx]
        points_by_system = points_by_panel[dataset]
        subplot_points = [point for points in points_by_system.values() for point in points]
        plotted_points.extend(subplot_points)

        for system, style in SYSTEM_STYLES.items():
            points = points_by_system[system]
            if not points:
                missing.append(f"{style['label']} {dataset} k={TOPK}")
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

        ax.set_title(f"{DATASET_LABELS[dataset]} (TopK = {TOPK})", pad=10, fontsize=22)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if col_idx != 0:
            ax.tick_params(labelleft=False)
        else:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _: "" if abs(value) < 1e-9 else f"{value:g}")
            )
        ax.set_xlim(*x_limits_by_dataset[dataset])
        ax.set_ylim(*ylim)
        ax.grid(True, color="#BFBFBF", linestyle="--", alpha=0.45, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.tight_layout(rect=(0.055, 0.17, 0.80, 0.90), w_pad=0.9)

    all_axes_left = min(ax.get_position().x0 for ax in axes)
    all_axes_bottom = min(ax.get_position().y0 for ax in axes)
    all_axes_top = max(ax.get_position().y1 for ax in axes)
    middle_col_center = (axes[1].get_position().x0 + axes[1].get_position().x1) / 2

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.82, 0.54),
        handlelength=1.8,
        columnspacing=1.15,
        labelspacing=1.0,
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
    fig.text(middle_col_center, 0.075, "Recall", ha="center", va="center", fontsize=22)

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
    )
    print(f"png={png_path.resolve()}")
    print(f"pdf={pdf_path.resolve()}")
    print(f"points={csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
