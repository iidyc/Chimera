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
    normalize_datasets,
    parse_logs,
    quote_cmd,
    repo_root_from_script,
    reset_run_dir,
    write_rows,
    write_summary,
    ensure_binaries,
    SYSTEMS,
)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Run full gpu_search_v9 config sweeps for slots 1/8 and top-k 10/100."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "gpu_mvr_v9_main_benchmark")
    parser.add_argument("--run-id", default="20260429")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--slots", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 100])
    parser.add_argument("-N", "--num-runs", type=int, default=1)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stage3-threads", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_bench(
    args: argparse.Namespace,
    dataset: str,
    slot: int,
    k: int,
    run_index: int,
    env: dict[str, str],
) -> None:
    output_run_dir = args.output_root / "runs" / args.run_id
    log_run_dir = args.output_root / "logs" / args.run_id
    config_file = dataset_config_path(args.repo_root, dataset)
    impl_label = f"v9_slots{slot}"
    cmd = [
        str(args.repo_root / "script" / "bench_gpu_mvr.sh"),
        "--dataset", dataset,
        "--version", "v9",
        "--implementation-label", impl_label,
        "--binary", str(args.build_dir / "gpu_search_v9"),
        "--config-file", str(config_file),
        "--output-dir", str(output_run_dir),
        "--log-dir", str(log_run_dir),
        "--build-dir", str(args.build_dir),
        "--k", str(k),
        "--nq", str(args.nq),
        "--warmup", str(args.warmup),
        "--concurrent-queries", str(slot),
    ]
    if args.stage3_threads > 0:
        cmd += ["--stage3-threads", str(args.stage3_threads)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"[run] run={run_index} dataset={dataset} k={k} slots={slot}", flush=True)
    print(f"[cmd] {quote_cmd(cmd)}", flush=True)
    import subprocess

    subprocess.run(cmd, cwd=args.repo_root, env=env, check=True)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output_root = args.output_root.resolve()
    datasets = normalize_datasets(args.datasets)
    ensure_binaries(args.build_dir, [SYSTEMS["v9"]], args.dry_run)
    output_run_dir = args.output_root / "runs" / args.run_id
    log_run_dir = args.output_root / "logs" / args.run_id
    reset_run_dir(output_run_dir, args.force)
    reset_run_dir(log_run_dir, args.force)

    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    for run_index in range(1, args.num_runs + 1):
        for k in args.ks:
            for dataset in datasets:
                for slot in args.slots:
                    run_bench(args, dataset, slot, k, run_index, env)

    if args.dry_run:
        return 0

    parse_logs(args.repo_root, log_run_dir, args.dry_run)
    rows = collect_result_rows(output_run_dir)
    for row in rows:
        system = row.get("system", "")
        if system.startswith("v9_slots"):
            row["slots"] = system.removeprefix("v9_slots")
            row["system"] = "v9"
        else:
            row["slots"] = ""

    results_csv = args.output_root / f"v9_main_results_{args.run_id}.csv"
    summary_csv = args.output_root / f"v9_main_summary_{args.run_id}.csv"
    preferred = [
        "system", "slots", "dataset", "k", "label", "run_index", "recall", "qps",
        "gpu_mem_peak_mib", "avg_latency_ms", "p50_ms", "p90_ms", "p95_ms",
        "p99_ms", "max_ms", "stddev_ms", "nprobe", "k_rank_cluster",
        "k_rank_all_tokens", "itopk_size", "overlap_chunks", "source_csv",
    ]
    write_rows(results_csv, rows, preferred)
    write_summary(summary_csv, rows, ["system", "slots", "dataset", "k", "label"])
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
