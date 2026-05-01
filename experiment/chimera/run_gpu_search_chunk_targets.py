#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import argparse
from gpu_search_experiment_lib import (
    DATASETS,
    SYSTEMS,
    build_systems,
    common_env,
    ensure_binaries,
    normalize_datasets,
    normalize_targets,
    repo_root_from_script,
    reset_run_dir,
    run_gpu_search_direct,
    selected_recall_target_configs,
    utc_run_id,
    write_rows,
    write_summary,
)
def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Benchmark gpu_search with overlap_chunks 1..12 at Recall@100 target points."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "Chimera" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "chimera_chunk_targets")
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--targets", nargs="+", default=["0.90", "0.95"])
    parser.add_argument("--chunks", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("-N", "--num-runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stage3-threads", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=32)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    output_root = args.output_root.resolve()
    log_dir = output_root / "logs" / args.run_id
    datasets = normalize_datasets(args.datasets)
    targets = normalize_targets(args.targets)
    configs = selected_recall_target_configs(datasets, targets)
    system = SYSTEMS["gpu_search"]
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.slots <= 0:
        raise ValueError("--slots must be > 0")
    if any(chunk <= 0 for chunk in args.chunks):
        raise ValueError("--chunks must all be > 0")
    if args.build:
        build_systems(repo_root, build_dir, [system], args.dry_run)
    ensure_binaries(build_dir, [system], args.dry_run)
    reset_run_dir(log_dir, args.force)
    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    rows = []
    for run_index in range(1, args.num_runs + 1):
        for cfg in configs:
            for chunk in args.chunks:
                config = cfg.as_row(
                    label=f"r{cfg.target_recall.replace('.', '')}_{cfg.label}_c{chunk}",
                    overlap_chunks=chunk,
                )
                log_path = (
                    log_dir / cfg.dataset / cfg.target_recall /
                    f"chunk{chunk}_slots{args.slots}_run{run_index}.log"
                )
                rows.append(run_gpu_search_direct(
                    repo_root,
                    build_dir,
                    log_path,
                    system,
                    cfg.dataset,
                    config,
                    args.k,
                    args.nq,
                    args.warmup,
                    run_index,
                    args.dry_run,
                    env,
                    concurrent_queries=args.slots,
                    stage3_threads=args.stage3_threads,
                    extra_row={
                        "experiment": "chimera_chunk_targets",
                        "target_recall": cfg.target_recall,
                        "base_label": cfg.label,
                        "chunks": str(chunk),
                    },
                ))
    if args.dry_run:
        return 0
    results_csv = output_root / f"chimera_chunk_targets_results_{args.run_id}.csv"
    summary_csv = output_root / f"chimera_chunk_targets_summary_{args.run_id}.csv"
    preferred = [
        "experiment", "system", "dataset", "target_recall", "base_label",
        "chunks", "slots", "k", "run_index", "recall_at_10", "recall_at_100",
        "recall", "qps", "avg_latency_ms", "p50_ms", "p90_ms", "p95_ms",
        "p99_ms", "max_ms", "stddev_ms", "gpu_mem_peak_mib", "nprobe",
        "k_rank_cluster", "k_rank_all_tokens", "itopk_size", "overlap_chunks",
        "binary", "log", "returncode", "command", "error_tail",
    ]
    write_rows(results_csv, rows, preferred)
    write_summary(summary_csv, rows, ["system", "dataset", "target_recall", "chunks", "slots"])
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
