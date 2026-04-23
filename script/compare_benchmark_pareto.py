#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RESULTS_HEADER = [
    "implementation",
    "dataset",
    "label",
    "setting",
    "queries",
    "k",
    "qps",
    "recall",
    "source_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge benchmark result tables for ColBERT and GPU-MVR v4 and write "
            "a joint recall-vs-qps Pareto-style comparison for each dataset."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling root containing per-dataset benchmark outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to compare. Repeatable. Default: lotte, hotpot, msmarco.",
    )
    return parser.parse_args()


def find_latest_gpu_v4_results(dataset_dir: Path) -> Path:
    candidates = sorted((dataset_dir / "gpu_search_v4").glob("benchmark_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no gpu_search_v4 benchmark_results_*.csv found under {dataset_dir}")

    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_colbert_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {
                "implementation": row["implementation"],
                "dataset": row["dataset"],
                "label": row["label"],
                "setting": f"ncells={row['ncells']} ndocs={row['ndocs']}",
                "queries": row["queries"],
                "k": row["k"],
                "qps": row["qps"],
                "recall": row["recall"],
                "source_csv": str(path),
            }
        )
    return normalized


def normalize_gpu_v4_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {
                "implementation": row["implementation"],
                "dataset": row["dataset"],
                "label": row["label"],
                "setting": (
                    f"nprobe={row['nprobe']} "
                    f"k_rank_cluster={row['k_rank_cluster']} "
                    f"k_rank_all_tokens={row['k_rank_all_tokens']} "
                    f"itopk_size={row['itopk_size']} "
                    f"overlap_chunks={row['overlap_chunks']}"
                ),
                "queries": row["queries"],
                "k": row["k"],
                "qps": row["qps"],
                "recall": row["recall"],
                "source_csv": str(path),
            }
        )
    return normalized


def compute_pareto_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    frontier: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        qps = float(row["qps"])
        recall = float(row["recall"])
        dominated = False
        for other_idx, other in enumerate(rows):
            if idx == other_idx:
                continue
            other_qps = float(other["qps"])
            other_recall = float(other["recall"])
            if (
                other_qps >= qps
                and other_recall >= recall
                and (other_qps > qps or other_recall > recall)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)

    frontier.sort(key=lambda row: (float(row["qps"]), float(row["recall"])))
    return frontier


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def compare_dataset(profiling_root: Path, dataset: str) -> tuple[Path, Path, list[dict[str, str]]]:
    dataset_dir = profiling_root / dataset
    colbert_results = dataset_dir / "colbert" / "benchmark_results.csv"
    if not colbert_results.is_file():
        raise FileNotFoundError(f"missing ColBERT results: {colbert_results}")

    gpu_v4_results = find_latest_gpu_v4_results(dataset_dir)
    all_rows = normalize_colbert_rows(colbert_results) + normalize_gpu_v4_rows(gpu_v4_results)
    frontier_rows = compute_pareto_rows(all_rows)

    output_dir = dataset_dir / "comparison_colbert_vs_gpu_search_v4"
    combined_csv = output_dir / "combined_points.csv"
    pareto_csv = output_dir / "pareto_frontier.csv"
    write_csv(combined_csv, all_rows)
    write_csv(pareto_csv, frontier_rows)
    return combined_csv, pareto_csv, frontier_rows


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    datasets = args.datasets or ["lotte", "hotpot", "msmarco"]

    for dataset in datasets:
        combined_csv, pareto_csv, frontier_rows = compare_dataset(profiling_root, dataset)
        print(
            f"dataset={dataset} combined={combined_csv} pareto={pareto_csv} "
            f"frontier_points={len(frontier_rows)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
