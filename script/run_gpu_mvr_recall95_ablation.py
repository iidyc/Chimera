#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_mvr_experiment_lib import (
    DATASETS,
    R095_LABELS,
    build_systems,
    collect_result_rows,
    common_env,
    dataset_config_path,
    ensure_binaries,
    find_config_row,
    normalize_datasets,
    normalize_systems,
    parse_logs,
    repo_root_from_script,
    reset_run_dir,
    run_bench,
    utc_run_id,
    write_rows,
    write_single_config,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Run Recall@100 ~= 0.95 ablations for GPU-MVR variants. "
            "Defaults use lotte:g2, msmarco:d1, and hotpot:k1 from profiling configs."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "gpu_mvr_recall95_ablation")
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--systems",
        nargs="+",
        default=[
            "v6_lite",
            "v6_turbo",
            "v6_nolut",
            "v6_nosum",
            "v6_nolut_nosum",
            "v7_lite",
            "v7_noalign",
            "v8",
        ],
    )
    parser.add_argument("--lotte-label", default=R095_LABELS["lotte"])
    parser.add_argument("--msmarco-label", default=R095_LABELS["msmarco"])
    parser.add_argument("--hotpot-label", default=R095_LABELS["hotpot"])
    parser.add_argument("-N", "--num-runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove this run-id directory before running.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def label_for_dataset(args: argparse.Namespace, dataset: str) -> str:
    return {
        "lotte": args.lotte_label,
        "msmarco": args.msmarco_label,
        "hotpot": args.hotpot_label,
    }[dataset]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    output_root = args.output_root.resolve()
    output_run_dir = output_root / "runs" / args.run_id
    log_run_dir = output_root / "logs" / args.run_id
    config_run_dir = output_root / "configs" / args.run_id
    datasets = normalize_datasets(args.datasets)
    systems = normalize_systems(args.systems)

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.build:
        build_systems(repo_root, build_dir, systems, args.dry_run)
    ensure_binaries(build_dir, systems, args.dry_run)
    reset_run_dir(output_run_dir, args.force)
    reset_run_dir(log_run_dir, args.force)
    reset_run_dir(config_run_dir, args.force)

    config_by_dataset = {}
    for dataset in datasets:
        config_file = dataset_config_path(repo_root, dataset)
        label = label_for_dataset(args, dataset)
        row = find_config_row(config_file, label)
        target_config = config_run_dir / f"{dataset}_recall95_{label}.csv"
        write_single_config(target_config, row, label=f"r095_{label}")
        config_by_dataset[dataset] = target_config

    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    for run_index in range(1, args.num_runs + 1):
        print(f"[repeat] {run_index}/{args.num_runs}", flush=True)
        for dataset in datasets:
            for system in systems:
                run_bench(
                    repo_root,
                    build_dir,
                    output_run_dir,
                    log_run_dir,
                    dataset,
                    system,
                    config_by_dataset[dataset],
                    args.k,
                    args.nq,
                    args.warmup,
                    args.dry_run,
                    env,
                )

    if args.dry_run:
        return 0

    parse_logs(repo_root, log_run_dir, args.dry_run)
    rows = collect_result_rows(output_run_dir)
    for row in rows:
        row["target_recall"] = "0.95"
    results_csv = output_root / f"recall95_ablation_results_{args.run_id}.csv"
    summary_csv = output_root / f"recall95_ablation_summary_{args.run_id}.csv"
    preferred = [
        "system",
        "dataset",
        "target_recall",
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
    write_summary(summary_csv, rows, ["system", "dataset", "target_recall", "label"])
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
