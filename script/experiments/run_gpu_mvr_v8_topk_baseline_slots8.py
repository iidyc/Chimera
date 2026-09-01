#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gpu_mvr_experiment_lib import (
    DATASETS,
    SYSTEMS,
    build_systems,
    common_env,
    dataset_config_path,
    ensure_binaries,
    normalize_datasets,
    parse_gpu_search_log_by_config,
    quote_cmd,
    read_config_rows,
    repo_root_from_script,
    reset_run_dir,
    utc_run_id,
    validate_dataset_inputs,
    write_config_rows,
    write_rows,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark gpu_search_v8 top-k baseline with 8 query slots. "
            "This is run_gpu_mvr_v8_topk_baseline.py with slots=8."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "profiling" / "gpu_mvr_v8_topk_baseline_slots8",
    )
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--config-labels", nargs="+", default=None)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--topks", nargs="+", type=int, default=[100])
    parser.add_argument("-N", "--num-runs", type=int, default=3)
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
    system = SYSTEMS["v8"]

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.slots <= 0:
        raise ValueError("--slots must be > 0")
    if not args.topks or any(k <= 0 for k in args.topks):
        raise ValueError("--topks must contain positive integers")
    if args.build:
        build_systems(repo_root, build_dir, [system], args.dry_run)
    ensure_binaries(build_dir, [system], args.dry_run)
    reset_run_dir(log_dir, args.force)

    wanted_labels = set(args.config_labels or [])
    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    config_root = repo_root / "config" / "tmp" / "gpu_mvr_v8_topk_baseline_slots8" / args.run_id
    rows = []
    for dataset in datasets:
        validate_dataset_inputs(repo_root, dataset)
        configs = read_config_rows(dataset_config_path(repo_root, dataset))
        if wanted_labels:
            configs = [row for row in configs if row["label"] in wanted_labels]

        for topk in args.topks:
            run_configs = []
            config_meta = {}
            for config in configs:
                base_label = config["label"]
                for run_index in range(1, args.num_runs + 1):
                    run_config = dict(config)
                    run_config["label"] = f"{base_label}_k{topk}_s{args.slots}_run{run_index}"
                    run_configs.append(run_config)
                    config_meta[run_config["label"]] = (base_label, run_index)

            label_part = "selected" if wanted_labels else "all"
            config_path = (
                config_root / dataset / system.label /
                f"{label_part}_k{topk}_slots{args.slots}_runs{args.num_runs}.csv"
            )
            log_path = (
                log_dir / dataset / system.label /
                f"{label_part}_k{topk}_slots{args.slots}_runs{args.num_runs}.log"
            )
            write_config_rows(config_path, run_configs)
            cmd = [
                str(build_dir / system.binary),
                "--query", str(repo_root / "dataset" / dataset / "raw" / "query.bin"),
                "--gt", str(repo_root / "dataset" / dataset / "raw" / "gt.tsv"),
                "--index", str(repo_root / "dataset" / dataset / "gpu_mvr_2m"),
                "--k", str(topk),
                "--nq", str(args.nq),
                "--warmup", str(args.warmup),
                "--concurrent-queries", str(args.slots),
                "--config-file", str(config_path),
            ]
            if args.stage3_threads > 0:
                cmd.extend(["--stage3-threads", str(args.stage3_threads)])

            print(
                f"[run] dataset={dataset} system={system.label} labels={label_part} "
                f"rows={len(run_configs)} k={topk} slots={args.slots}",
                flush=True,
            )
            if args.dry_run:
                print(f"[dry-run] {quote_cmd(cmd)}", flush=True)
                parsed = {}
                returncode = ""
                error_tail = ""
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as handle:
                    handle.write(f"[driver] command={quote_cmd(cmd)}\n")
                    handle.write(f"[driver] dataset={dataset}\n")
                    handle.write(f"[driver] system={system.label}\n")
                    handle.write(f"[driver] config_file={config_path}\n")
                    handle.write(f"[driver] topk={topk}\n")
                    handle.write(f"[driver] slots={args.slots}\n")
                    handle.flush()
                    proc = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                        check=False,
                    )
                parsed = parse_gpu_search_log_by_config(log_path) if proc.returncode == 0 else {}
                returncode = str(proc.returncode)
                error_tail = (
                    ""
                    if proc.returncode == 0
                    else "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
                )

            for config in run_configs:
                base_label, run_index = config_meta[config["label"]]
                row = {
                    "experiment": "v8_topk_baseline_slots8",
                    "system": system.label,
                    "dataset": dataset,
                    "label": base_label,
                    "config_label": config["label"],
                    "slots": str(args.slots),
                    "k": str(topk),
                    "nq": str(args.nq),
                    "warmup": str(args.warmup),
                    "run_index": str(run_index),
                    "nprobe": config["nprobe"],
                    "k_rank_cluster": config["k_rank_cluster"],
                    "k_rank_all_tokens": config["k_rank_all_tokens"],
                    "itopk_size": config["itopk_size"],
                    "overlap_chunks": config["overlap_chunks"],
                    "binary": system.binary,
                    "log": str(log_path),
                    "returncode": returncode,
                    "command": quote_cmd(cmd),
                    "error_tail": error_tail,
                }
                row.update(parsed.get(config["label"], {}))
                rows.append(row)

    if args.dry_run:
        return 0

    results_csv = output_root / f"v8_topk_baseline_slots8_results_{args.run_id}.csv"
    summary_csv = output_root / f"v8_topk_baseline_slots8_summary_{args.run_id}.csv"
    preferred = [
        "experiment", "system", "dataset", "label", "config_label", "slots",
        "k", "run_index", "recall_at_10", "recall_at_100", "recall", "qps",
        "avg_latency_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms",
        "stddev_ms", "gpu_mem_peak_mib", "effective_concurrent_queries",
        "stage3_threads", "nprobe", "k_rank_cluster", "k_rank_all_tokens",
        "itopk_size", "overlap_chunks", "binary", "log", "returncode",
        "command", "error_tail",
    ]
    write_rows(results_csv, rows, preferred)
    write_summary(summary_csv, rows, ["system", "dataset", "label", "slots", "k"])
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
