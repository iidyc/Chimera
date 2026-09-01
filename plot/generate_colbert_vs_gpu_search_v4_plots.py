#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

baseline = "PLAID"
oursys = "GPU-MVR"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate small scatter plots for ColBERT vs GPU-MVR v4 using each "
            "implementation's Pareto frontier independently."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling root containing comparison CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot"),
        help="Directory to write plots and helper cache directories.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to plot. Repeatable. Default: lotte, hotpot, msmarco.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="Filter out plotted points with recall below this threshold. Default: 0.8.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_latest_gpu_v4_frontier(dataset_dir: Path) -> Path:
    candidates = sorted((dataset_dir / "gpu_search_v4").glob("pareto_frontier_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no gpu_search_v4 pareto_frontier_*.csv found under {dataset_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def load_frontier_rows(colbert_frontier: Path, gpu_v4_frontier: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    colbert_rows = sorted(read_rows(colbert_frontier), key=lambda row: float(row["recall"]))
    gpu_v4_rows = sorted(read_rows(gpu_v4_frontier), key=lambda row: float(row["recall"]))
    if not colbert_rows:
        raise RuntimeError(f"no ColBERT rows found in {colbert_frontier}")
    if not gpu_v4_rows:
        raise RuntimeError(f"no GPU-MVR v4 rows found in {gpu_v4_frontier}")
    return colbert_rows, gpu_v4_rows


def filter_rows_by_min_recall(rows: list[dict[str, str]], min_recall: float) -> list[dict[str, str]]:
    return [row for row in rows if float(row["recall"]) >= min_recall]


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


def render_dataset(
    profiling_root: Path,
    output_dir: Path,
    dataset: str,
    min_recall: float,
) -> tuple[Path, Path]:
    setup_matplotlib_cache(output_dir)

    import matplotlib.pyplot as plt

    dataset_dir = profiling_root / dataset
    colbert_frontier = dataset_dir / "colbert" / "pareto_frontier.csv"
    gpu_v4_frontier = find_latest_gpu_v4_frontier(dataset_dir)
    colbert_rows, gpu_v4_rows = load_frontier_rows(colbert_frontier, gpu_v4_frontier)
    colbert_rows = filter_rows_by_min_recall(colbert_rows, min_recall)
    gpu_v4_rows = filter_rows_by_min_recall(gpu_v4_rows, min_recall)
    if not colbert_rows:
        raise RuntimeError(
            f"no ColBERT rows remain for {dataset} after applying min recall {min_recall:.3f}"
        )
    if not gpu_v4_rows:
        raise RuntimeError(
            f"no GPU-MVR v4 rows remain for {dataset} after applying min recall {min_recall:.3f}"
        )

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.plot(
        [float(row["recall"]) for row in colbert_rows],
        [float(row["qps"]) for row in colbert_rows],
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label=baseline,
    )
    ax.plot(
        [float(row["recall"]) for row in gpu_v4_rows],
        [float(row["qps"]) for row in gpu_v4_rows],
        marker="s",
        linewidth=1.2,
        markersize=3.5,
        label=oursys,
    )

    all_rows = colbert_rows + gpu_v4_rows
    recall_values = [float(row["recall"]) for row in all_rows]
    qps_values = [float(row["qps"]) for row in all_rows]

    ax.set_xlim(max(0.0, min_recall), min(1.0, max(recall_values) + 0.005))
    ax.set_ylim(max(0.0, min(qps_values) * 0.95), max(qps_values) * 1.05)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS")
    ax.set_title(dataset.upper())
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout(pad=0.3)

    png_path = output_dir / f"colbert_vs_gpu_search_v4_{dataset}.png"
    pdf_path = output_dir / f"colbert_vs_gpu_search_v4_{dataset}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or ["lotte", "hotpot", "msmarco"]

    for dataset in datasets:
        png_path, pdf_path = render_dataset(
            profiling_root,
            output_dir,
            dataset,
            args.min_recall,
        )
        print(f"dataset={dataset} png={png_path.resolve()} pdf={pdf_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
