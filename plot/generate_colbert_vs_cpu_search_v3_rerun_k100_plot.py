#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


DEFAULT_DATASETS = ["lotte", "hotpot", "msmarco"]
DEFAULT_OUTPUT_NAME = "plaid_vs_cpu_search_v3_rerun_k100_side_by_side"
DATASET_TITLES = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side Pareto-frontier comparisons for PLAID and "
            "cpu_search_v3 rerun results at top-k=100."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling root containing per-dataset frontier CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot"),
        help="Directory where plot files will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output file stem. Default: {DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to plot. Repeatable. Default: lotte, hotpot, msmarco.",
    )
    parser.add_argument(
        "--cpu-implementation",
        default="cpu_search_v3",
        help="CPU implementation directory under profiling/<dataset>/.",
    )
    parser.add_argument(
        "--cpu-batch-substring",
        default="rerun",
        help="Only consider CPU frontier files whose filename contains this substring.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="top-k value to plot. Default: 100.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Optional recall filter applied to both systems before plotting.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_colbert_frontier(dataset_dir: Path) -> Path:
    path = dataset_dir / "colbert" / "pareto_frontier.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing ColBERT frontier: {path}")
    return path


def find_latest_cpu_frontier(
    dataset_dir: Path,
    implementation: str,
    k: int,
    batch_substring: str,
) -> Path:
    impl_dir = dataset_dir / implementation
    candidates = sorted(impl_dir.glob(f"pareto_frontier_k{k}_*.csv"))
    if batch_substring:
        candidates = [path for path in candidates if batch_substring in path.name]
    if not candidates:
        raise FileNotFoundError(
            f"no pareto_frontier_k{k}_*.csv matching {batch_substring!r} found under {impl_dir}"
        )

    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def load_frontier_rows(path: Path, min_recall: float) -> list[dict[str, str]]:
    rows = [row for row in read_rows(path) if float(row["recall"]) >= min_recall]
    if not rows:
        raise RuntimeError(f"no rows remain in {path} after applying min_recall={min_recall:.3f}")
    rows.sort(key=lambda row: float(row["qps"]))
    return rows


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


def render_plot(
    profiling_root: Path,
    output_dir: Path,
    output_name: str,
    datasets: list[str],
    cpu_implementation: str,
    cpu_batch_substring: str,
    k: int,
    min_recall: float,
) -> tuple[Path, Path, list[tuple[str, Path, Path]]]:
    setup_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_rows: list[tuple[str, list[dict[str, str]], list[dict[str, str]]]] = []
    selected_sources: list[tuple[str, Path, Path]] = []
    global_min_recall = 1.0
    global_max_recall = 0.0

    for dataset in datasets:
        dataset_dir = profiling_root / dataset
        colbert_path = find_colbert_frontier(dataset_dir)
        cpu_path = find_latest_cpu_frontier(dataset_dir, cpu_implementation, k, cpu_batch_substring)
        colbert_rows = load_frontier_rows(colbert_path, min_recall)
        cpu_rows = load_frontier_rows(cpu_path, min_recall)
        dataset_rows.append((dataset, colbert_rows, cpu_rows))
        selected_sources.append((dataset, colbert_path, cpu_path))

        all_rows = colbert_rows + cpu_rows
        global_min_recall = min(global_min_recall, *(float(row["recall"]) for row in all_rows))
        global_max_recall = max(global_max_recall, *(float(row["recall"]) for row in all_rows))

    fig, axes = plt.subplots(1, len(dataset_rows), figsize=(4.0 * len(dataset_rows), 3.3), sharey=True)
    if len(dataset_rows) == 1:
        axes = [axes]

    for ax, (dataset, colbert_rows, cpu_rows) in zip(axes, dataset_rows):
        colbert_qps = [float(row["qps"]) for row in colbert_rows]
        colbert_recall = [float(row["recall"]) for row in colbert_rows]
        cpu_qps = [float(row["qps"]) for row in cpu_rows]
        cpu_recall = [float(row["recall"]) for row in cpu_rows]

        ax.plot(
            colbert_qps,
            colbert_recall,
            marker="o",
            linewidth=1.3,
            markersize=3.8,
            color="#1f77b4",
            label="PLAID",
        )
        ax.plot(
            cpu_qps,
            cpu_recall,
            marker="s",
            linewidth=1.3,
            markersize=3.8,
            color="#ff7f0e",
            label="cpu_search_v3 rerun",
        )

        qps_values = colbert_qps + cpu_qps
        ax.set_xlim(max(0.0, min(qps_values) * 0.85), max(qps_values) * 1.05)
        ax.set_xlabel("QPS")
        ax.set_title(DATASET_TITLES.get(dataset, dataset.upper()))
        ax.grid(True, alpha=0.3, linewidth=0.5)

    axes[0].set_ylabel("Recall@100")
    axes[0].set_ylim(max(0.0, global_min_recall - 0.01), min(1.0, global_max_recall + 0.005))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Top-100 Pareto Frontier: PLAID vs cpu_search_v3 rerun", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, selected_sources


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DEFAULT_DATASETS

    png_path, pdf_path, selected_sources = render_plot(
        profiling_root=profiling_root,
        output_dir=output_dir,
        output_name=args.output_name,
        datasets=datasets,
        cpu_implementation=args.cpu_implementation,
        cpu_batch_substring=args.cpu_batch_substring,
        k=args.k,
        min_recall=args.min_recall,
    )

    for dataset, colbert_path, cpu_path in selected_sources:
        print(
            f"dataset={dataset} colbert_frontier={colbert_path} cpu_frontier={cpu_path}"
        )
    print(f"png={png_path.resolve()}")
    print(f"pdf={pdf_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
