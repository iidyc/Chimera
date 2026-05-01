#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DATASETS = ("lotte", "hotpot", "msmarco")
CONFIG_HEADER = [
    "label",
    "nprobe",
    "k_rank_cluster",
    "k_rank_all_tokens",
    "itopk_size",
    "overlap_chunks",
]
SUMMARY_HEADER = [
    "dataset",
    "k",
    "target_recall",
    "label",
    "recall",
    "qps",
    "avg_latency_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "stddev_ms",
    "queries",
    "warmup",
    "nprobe",
    "k_rank_cluster",
    "k_rank_all_tokens",
    "itopk_size",
    "overlap_chunks",
    "results_csv",
    "config_csv",
]


class SelectionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the highest-QPS cpu_search configuration that meets the "
            "target recall for each dataset and write one-row config CSV files "
            "for follow-on profiling jobs."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("profiling"),
        help="Root containing profiling/<dataset>/cpu_search benchmark CSVs.",
    )
    parser.add_argument(
        "--implementation-label",
        default="cpu_search",
        help="Implementation subdirectory under profiling/<dataset>/.",
    )
    parser.add_argument(
        "--k10-batch-id",
        required=True,
        help="Batch id used for the top-k=10 benchmark run.",
    )
    parser.add_argument(
        "--k100-batch-id",
        required=True,
        help="Batch id used for the top-k=100 benchmark run.",
    )
    parser.add_argument(
        "--topk10-recall",
        type=float,
        default=0.9,
        help="Target recall threshold for the top-k=10 run.",
    )
    parser.add_argument(
        "--topk100-recall",
        type=float,
        default=0.95,
        help="Target recall threshold for the top-k=100 run.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Summary CSV describing the selected configs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory where one-row config CSV files will be written.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_results_csv(results_root: Path, implementation: str, dataset: str, k: int, batch_id: str) -> Path:
    path = results_root / dataset / implementation / f"benchmark_results_k{k}_{batch_id}.csv"
    if not path.is_file():
        raise SelectionError(f"missing results CSV: {path}")
    return path


def choose_best_row(rows: list[dict[str, str]], dataset: str, k: int, target_recall: float) -> dict[str, str]:
    eligible = [row for row in rows if float(row["recall"]) >= target_recall]
    if not eligible:
        raise SelectionError(
            f"no config reaches recall>={target_recall} for dataset={dataset} k={k}"
        )
    eligible.sort(
        key=lambda row: (
            float(row["qps"]),
            float(row["recall"]),
            -float(row["avg_latency_ms"]),
        ),
        reverse=True,
    )
    return eligible[0]


def write_config_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFIG_HEADER)
        writer.writeheader()
        writer.writerow({key: row[key] for key in CONFIG_HEADER})


def main() -> int:
    args = parse_args()

    try:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.config_dir.mkdir(parents=True, exist_ok=True)

        selections: list[dict[str, str]] = []
        cases = (
            (10, args.k10_batch_id, args.topk10_recall),
            (100, args.k100_batch_id, args.topk100_recall),
        )

        for k, batch_id, target_recall in cases:
            for dataset in DATASETS:
                results_csv = find_results_csv(
                    args.results_root,
                    args.implementation_label,
                    dataset,
                    k,
                    batch_id,
                )
                rows = load_csv_rows(results_csv)
                best = choose_best_row(rows, dataset, k, target_recall)
                config_csv = (
                    args.config_dir
                    / f"{dataset}_k{k}_target{target_recall:.2f}_{best['label']}.csv"
                )
                write_config_csv(config_csv, best)
                selection = {key: best[key] for key in SUMMARY_HEADER if key in best}
                selection.update(
                    {
                        "dataset": dataset,
                        "k": str(k),
                        "target_recall": f"{target_recall:.2f}",
                        "results_csv": str(results_csv),
                        "config_csv": str(config_csv),
                    }
                )
                selections.append(selection)
                print(
                    f"[select] dataset={dataset} k={k} target_recall={target_recall:.2f} "
                    f"label={best['label']} recall={best['recall']} qps={best['qps']} "
                    f"config_csv={config_csv}"
                )

        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER)
            writer.writeheader()
            writer.writerows(selections)

    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
