#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


BENCHMARK_HEADER = [
    "implementation",
    "dataset",
    "label",
    "ncells",
    "ndocs",
    "index_source",
    "index_root",
    "index_path",
    "query_file",
    "gt_file",
    "queries",
    "k",
    "num_runs",
    "avg_time_s",
    "qps",
    "recall",
    "gpu_mem_current_mib",
    "gpu_mem_peak_mib",
    "gpu_mem_total_mib",
    "torch_peak_allocated_mib",
    "torch_peak_reserved_mib",
]


SEARCH_HEADER = [
    "ncells",
    "ndocs",
    "avg_time_s",
    "qps",
    "recall",
    "gpu_mem_current_mib",
    "gpu_mem_peak_mib",
    "gpu_mem_total_mib",
    "torch_peak_allocated_mib",
    "torch_peak_reserved_mib",
]


def read_config(path):
    configs = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip()
            if not label or label.startswith("#"):
                continue
            configs.append(
                {
                    "label": label,
                    "ncells": row["ncells"].strip(),
                    "ndocs": row["ndocs"].strip(),
                }
            )
    return configs


def read_search_outputs(paths, allow_missing):
    rows_by_pair = {}
    for path in paths:
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(path)

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                pair = (row["ncells"].strip(), row["ndocs"].strip())
                rows_by_pair[pair] = {key: row.get(key, "").strip() for key in SEARCH_HEADER}
    return rows_by_pair


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pareto_rows(rows):
    frontier = []
    for row in rows:
        qps = float(row["qps"])
        recall = float(row["recall"])
        dominated = False
        for other in rows:
            if other is row:
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
    return sorted(frontier, key=lambda item: (float(item["qps"]), float(item["recall"])))


def main():
    parser = argparse.ArgumentParser(
        description="Combine ColBERT search.py CSV shards into benchmark_results.csv."
    )
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--search-output", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--implementation-label", default="plaid")
    parser.add_argument("--dataset", default="msmarco")
    parser.add_argument("--k", required=True)
    parser.add_argument("--queries", default="6980")
    parser.add_argument("--num-runs", default="3")
    parser.add_argument("--index-source", default="dataset")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--allow-missing-inputs", action="store_true")
    args = parser.parse_args()

    configs = read_config(args.config_file)
    rows_by_pair = read_search_outputs(args.search_output, args.allow_missing_inputs)

    benchmark_rows = []
    combined_search_rows = []
    missing = []
    for config in configs:
        pair = (config["ncells"], config["ndocs"])
        search_row = rows_by_pair.get(pair)
        if search_row is None:
            missing.append(config["label"])
            continue

        combined_search_rows.append(search_row)
        benchmark_rows.append(
            {
                "implementation": args.implementation_label,
                "dataset": args.dataset,
                "label": config["label"],
                "ncells": config["ncells"],
                "ndocs": config["ndocs"],
                "index_source": args.index_source,
                "index_root": args.index_root,
                "index_path": args.index_path,
                "query_file": args.query_file,
                "gt_file": args.gt_file,
                "queries": args.queries,
                "k": args.k,
                "num_runs": args.num_runs,
                "avg_time_s": search_row["avg_time_s"],
                "qps": search_row["qps"],
                "recall": search_row["recall"],
                "gpu_mem_current_mib": search_row["gpu_mem_current_mib"],
                "gpu_mem_peak_mib": search_row["gpu_mem_peak_mib"],
                "gpu_mem_total_mib": search_row["gpu_mem_total_mib"],
                "torch_peak_allocated_mib": search_row["torch_peak_allocated_mib"],
                "torch_peak_reserved_mib": search_row["torch_peak_reserved_mib"],
            }
        )

    write_csv(args.output_dir / "search_py_output.csv", SEARCH_HEADER, combined_search_rows)
    write_csv(args.output_dir / "benchmark_results.csv", BENCHMARK_HEADER, benchmark_rows)
    write_csv(args.output_dir / "pareto_frontier.csv", BENCHMARK_HEADER, pareto_rows(benchmark_rows))

    print(f"wrote_rows={len(benchmark_rows)}")
    if missing:
        print(f"missing_labels={','.join(missing)}")


if __name__ == "__main__":
    main()
