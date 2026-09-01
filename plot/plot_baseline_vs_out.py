#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


baseline = "PLAID"
baseline_dir = "colbert"
oursys = "GPU-MVR"
default_gpu_version = "v6"
default_datasets = ["lotte", "hotpot", "msmarco"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate small scatter plots for a baseline system versus a GPU-MVR "
            "version using each implementation's Pareto frontier independently."
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
        "--gpu-version",
        default=default_gpu_version,
        help="GPU-MVR gpu_search version to compare, e.g. v4 or gpu_search_v4.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.835,
        help="Filter out plotted points with recall below this threshold. Default: 0.835.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_gpu_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("gpu_search_"):
        normalized = normalized.removeprefix("gpu_search_")
    if not normalized.startswith("v") or not normalized[1:].isdigit():
        raise ValueError(f"invalid GPU-MVR version {version!r}; expected vN or gpu_search_vN")
    return normalized


def find_latest_gpu_frontier(dataset_dir: Path, gpu_version: str) -> Path:
    candidates = sorted((dataset_dir / f"gpu_search_{gpu_version}").glob("pareto_frontier_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"no gpu_search_{gpu_version} pareto_frontier_*.csv found under {dataset_dir}"
        )
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def load_frontier_rows(
    baseline_frontier: Path,
    gpu_frontier: Path,
    gpu_version: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    baseline_rows = sorted(read_rows(baseline_frontier), key=lambda row: float(row["recall"]))
    gpu_rows = sorted(read_rows(gpu_frontier), key=lambda row: float(row["recall"]))
    if not baseline_rows:
        raise RuntimeError(f"no baseline rows found in {baseline_frontier}")
    if not gpu_rows:
        raise RuntimeError(f"no GPU-MVR {gpu_version} rows found in {gpu_frontier}")
    return baseline_rows, gpu_rows


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
    gpu_version: str,
    min_recall: float,
) -> tuple[Path, Path]:
    setup_matplotlib_cache(output_dir)

    import matplotlib.pyplot as plt

    dataset_dir = profiling_root / dataset
    baseline_frontier = dataset_dir / baseline_dir / "pareto_frontier.csv"
    gpu_frontier = find_latest_gpu_frontier(dataset_dir, gpu_version)
    baseline_rows, gpu_rows = load_frontier_rows(baseline_frontier, gpu_frontier, gpu_version)
    baseline_rows = filter_rows_by_min_recall(baseline_rows, min_recall)
    gpu_rows = filter_rows_by_min_recall(gpu_rows, min_recall)
    if not baseline_rows:
        raise RuntimeError(
            f"no {baseline} rows remain for {dataset} after applying min recall {min_recall:.3f}"
        )
    if not gpu_rows:
        raise RuntimeError(
            f"no {oursys} {gpu_version} rows remain for {dataset} after applying min recall {min_recall:.3f}"
        )

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.plot(
        [float(row["recall"]) for row in baseline_rows],
        [float(row["qps"]) for row in baseline_rows],
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label=baseline,
    )
    ax.plot(
        [float(row["recall"]) for row in gpu_rows],
        [float(row["qps"]) for row in gpu_rows],
        marker="s",
        linewidth=1.2,
        markersize=3.5,
        label=oursys,
    )

    all_rows = baseline_rows + gpu_rows
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

    png_path = output_dir / f"baseline_vs_gpu_search_{gpu_version}_{dataset}.png"
    pdf_path = output_dir / f"baseline_vs_gpu_search_{gpu_version}_{dataset}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or default_datasets
    gpu_version = normalize_gpu_version(args.gpu_version)

    for dataset in datasets:
        png_path, pdf_path = render_dataset(
            profiling_root,
            output_dir,
            dataset,
            gpu_version,
            args.min_recall,
        )
        print(
            f"dataset={dataset} gpu_version={gpu_version} "
            f"png={png_path.resolve()} pdf={pdf_path.resolve()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
