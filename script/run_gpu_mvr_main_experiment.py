#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_mvr_experiment_lib import (
    DATASETS,
    collect_result_rows,
    common_env,
    dataset_config_path,
    ensure_binaries,
    normalize_datasets,
    normalize_systems,
    parse_logs,
    repo_root_from_script,
    reset_run_dir,
    run_bench,
    utc_run_id,
    write_best_at_or_above_recall,
    write_rows,
    write_summary,
    build_systems,
)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Run the main GPU-MVR experiment for v6_lite, v7_lite, and v8. "
            "Each dataset uses its profiling/<dataset>_config.csv file, with "
            "profiling/latte_config.csv accepted for the lotte dataset."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "gpu_mvr_main_experiment")
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--systems", nargs="+", default=["v6_lite", "v7_lite", "v8"])
    parser.add_argument("-N", "--num-runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=None, help="Single top-k value to run. Overrides --ks.")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 100], help="Top-k values to run.")
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove this run-id directory before running.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    output_root = args.output_root.resolve()
    output_run_dir = output_root / "runs" / args.run_id
    log_run_dir = output_root / "logs" / args.run_id
    datasets = normalize_datasets(args.datasets)
    systems = normalize_systems(args.systems)

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    ks = [args.k] if args.k is not None else args.ks
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("all k values must be positive")
    if args.build:
        build_systems(repo_root, build_dir, systems, args.dry_run)
    ensure_binaries(build_dir, systems, args.dry_run)
    reset_run_dir(output_run_dir, args.force)
    reset_run_dir(log_run_dir, args.force)

    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    for run_index in range(1, args.num_runs + 1):
        print(f"[repeat] {run_index}/{args.num_runs}", flush=True)
        for k in ks:
            print(f"[topk] k={k}", flush=True)
            for dataset in datasets:
                config_file = dataset_config_path(repo_root, dataset)
                for system in systems:
                    run_bench(
                        repo_root,
                        build_dir,
                        output_run_dir,
                        log_run_dir,
                        dataset,
                        system,
                        config_file,
                        k,
                        args.nq,
                        args.warmup,
                        args.dry_run,
                        env,
                    )

    if args.dry_run:
        return 0

    parse_logs(repo_root, log_run_dir, args.dry_run)
    rows = collect_result_rows(output_run_dir)
    results_csv = output_root / f"main_results_{args.run_id}.csv"
    summary_csv = output_root / f"main_summary_{args.run_id}.csv"
    best_csv = output_root / f"main_best_recall{str(args.target_recall).replace('.', '')}_{args.run_id}.csv"

    preferred = [
        "system",
        "dataset",
        "k",
        "label",
        "run_index",
        "recall",
        "qps",
        "gpu_mem_peak_mib",
        "avg_latency_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "stddev_ms",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
        "source_csv",
    ]
    write_rows(results_csv, rows, preferred)
    write_summary(summary_csv, rows, ["system", "dataset", "k", "label"])
    if len(ks) == 1:
        write_best_at_or_above_recall(best_csv, rows, args.target_recall)
    else:
        for k in ks:
            k_rows = [row for row in rows if row.get("k") == str(k)]
            k_best_csv = output_root / (
                f"main_k{k}_best_recall{str(args.target_recall).replace('.', '')}_{args.run_id}.csv"
            )
            write_best_at_or_above_recall(k_best_csv, k_rows, args.target_recall)
            print(f"[wrote] {k_best_csv}")
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    if len(ks) == 1:
        print(f"[wrote] {best_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
