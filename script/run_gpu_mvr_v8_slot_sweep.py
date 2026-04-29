#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gpu_mvr_experiment_lib import (
    DATASETS,
    R095_LABELS,
    SYSTEMS,
    SystemSpec,
    build_systems,
    common_env,
    dataset_config_path,
    ensure_binaries,
    find_config_row,
    normalize_datasets,
    quote_cmd,
    repo_root_from_script,
    reset_run_dir,
    utc_run_id,
    write_rows,
    write_summary,
)


METRIC_PATTERNS = {
    "end_to_end_ms": re.compile(r"^\[SEARCH\] End-to-end measured time: ([0-9.]+) ms$", re.MULTILINE),
    "avg_latency_ms": re.compile(r"^\[SEARCH\] Average latency per query: ([0-9.]+) ms$", re.MULTILINE),
    "qps": re.compile(r"^\[SEARCH\] Throughput: ([0-9.]+) qps$", re.MULTILINE),
    "recall": re.compile(r"^Recall@\d+: ([0-9.]+)$", re.MULTILINE),
    "gpu_mem_current_mib": re.compile(r"^\[GPU_MEM\] current=([0-9.]+) MiB,", re.MULTILINE),
    "gpu_mem_peak_mib": re.compile(r"^\[GPU_MEM\].* peak=([0-9.]+) MiB,", re.MULTILINE),
    "effective_concurrent_queries": re.compile(r"effective_concurrent_queries=(\d+)"),
    "stage3_threads": re.compile(r"stage3_threads=(\d+)"),
}

LATENCY_RE = re.compile(r"^\[SEARCH\] Query latency distribution \(ms\): (.+)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Run v8 at Recall@100 ~= 0.95 while sweeping concurrent query slots. "
            "This varies --concurrent-queries, not overlap_chunks."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "gpu_mvr_v8_slot_sweep")
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=[
            "v8",
            "v8_nosum",
            "v8_nolut",
            "v8_nolut_nosum",
            "v8s8",
            "v8s8_nosum",
            "v8s8_nolut",
            "v8s8_nolut_nosum",
        ],
        default=["v8", "v8_nolut"],
    )
    parser.add_argument("--slots", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--lotte-label", default=R095_LABELS["lotte"])
    parser.add_argument("--msmarco-label", default=R095_LABELS["msmarco"])
    parser.add_argument("--hotpot-label", default=R095_LABELS["hotpot"])
    parser.add_argument("-N", "--num-runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stage3-threads", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=32)
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


def parse_key_value_payload(payload: str) -> dict[str, str]:
    fields = {}
    for item in payload.split(","):
        key, value = item.strip().split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        metrics[name] = match.group(1) if match else ""

    latency_match = LATENCY_RE.search(text)
    if latency_match:
        latency = parse_key_value_payload(latency_match.group(1))
        metrics.update({
            "min_ms": latency.get("min", ""),
            "p50_ms": latency.get("p50", ""),
            "p90_ms": latency.get("p90", ""),
            "p95_ms": latency.get("p95", ""),
            "p99_ms": latency.get("p99", ""),
            "max_ms": latency.get("max", ""),
            "stddev_ms": latency.get("stddev", ""),
        })
    return metrics


def command_for(
    repo_root: Path,
    build_dir: Path,
    system: SystemSpec,
    dataset: str,
    cfg: dict[str, str],
    slots: int,
    k: int,
    nq: int,
    warmup: int,
    stage3_threads: int,
) -> list[str]:
    cmd = [
        str(build_dir / system.binary),
        "--query", str(repo_root / "dataset" / dataset / "raw" / "query.bin"),
        "--gt", str(repo_root / "dataset" / dataset / "raw" / "gt.tsv"),
        "--index", str(repo_root / "dataset" / dataset / "gpu_mvr_2m"),
        "--k", str(k),
        "--nq", str(nq),
        "--warmup", str(warmup),
        "--nprobe", cfg["nprobe"],
        "--k-rank-cluster", cfg["k_rank_cluster"],
        "--k-rank-all-tokens", cfg["k_rank_all_tokens"],
        "--itopk-size", cfg["itopk_size"],
        "--overlap-chunks", cfg["overlap_chunks"],
        "--concurrent-queries", str(slots),
    ]
    if stage3_threads > 0:
        cmd += ["--stage3-threads", str(stage3_threads)]
    return cmd


def run_one(
    args: argparse.Namespace,
    repo_root: Path,
    build_dir: Path,
    log_dir: Path,
    system: SystemSpec,
    dataset: str,
    cfg: dict[str, str],
    slots: int,
    run_index: int,
    env: dict[str, str],
) -> dict[str, str]:
    log_path = log_dir / dataset / system.label / f"slots{slots}_run{run_index}.log"
    cmd = command_for(
        repo_root,
        build_dir,
        system,
        dataset,
        cfg,
        slots,
        args.k,
        args.nq,
        args.warmup,
        args.stage3_threads,
    )
    row = {
        "system": system.label,
        "dataset": dataset,
        "target_recall": "0.95",
        "slots": str(slots),
        "config_label": cfg["label"],
        "run_index": str(run_index),
        "nprobe": cfg["nprobe"],
        "k_rank_cluster": cfg["k_rank_cluster"],
        "k_rank_all_tokens": cfg["k_rank_all_tokens"],
        "itopk_size": cfg["itopk_size"],
        "overlap_chunks": cfg["overlap_chunks"],
        "k": str(args.k),
        "nq": str(args.nq),
        "warmup": str(args.warmup),
        "binary": system.binary,
        "log": str(log_path),
        "command": quote_cmd(cmd),
    }
    if args.dry_run:
        print(f"[dry-run] {quote_cmd(cmd)}", flush=True)
        return row

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] dataset={dataset} system={system.label} slots={slots} run={run_index}", flush=True)
    start = time.time()
    with log_path.open("w") as handle:
        handle.write(f"[driver] command={quote_cmd(cmd)}\n")
        handle.write(f"[driver] dataset={dataset}\n")
        handle.write(f"[driver] system={system.label}\n")
        handle.write(f"[driver] config_label={cfg['label']}\n")
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
    row["returncode"] = str(proc.returncode)
    row["elapsed_s"] = f"{time.time() - start:.3f}"
    if proc.returncode == 0:
        row.update(parse_log(log_path))
    else:
        row["error_tail"] = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
    return row


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    output_root = args.output_root.resolve()
    log_dir = output_root / "logs" / args.run_id
    datasets = normalize_datasets(args.datasets)
    systems = [SYSTEMS[name] for name in args.systems]

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if any(slot <= 0 for slot in args.slots):
        raise ValueError("--slots must all be positive")
    if args.stage3_threads > 0 and args.stage3_threads == args.omp_num_threads:
        print(
            "[driver] stage3_threads equals omp_num_threads; using v8 auto-split "
            "so the experiment stays within the requested total thread pool.",
            flush=True,
        )
        args.stage3_threads = 0
    if args.build:
        build_systems(repo_root, build_dir, systems, args.dry_run)
    ensure_binaries(build_dir, systems, args.dry_run)
    reset_run_dir(log_dir, args.force)

    configs = {}
    for dataset in datasets:
        config_file = dataset_config_path(repo_root, dataset)
        configs[dataset] = find_config_row(config_file, label_for_dataset(args, dataset))

    rows = []
    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    for run_index in range(1, args.num_runs + 1):
        for dataset in datasets:
            for system in systems:
                for slots in args.slots:
                    row = run_one(
                        args,
                        repo_root,
                        build_dir,
                        log_dir,
                        system,
                        dataset,
                        configs[dataset],
                        slots,
                        run_index,
                        env,
                    )
                    rows.append(row)

    if args.dry_run:
        return 0

    results_csv = output_root / f"v8_slot_sweep_results_{args.run_id}.csv"
    summary_csv = output_root / f"v8_slot_sweep_summary_{args.run_id}.csv"
    preferred = [
        "system",
        "dataset",
        "target_recall",
        "slots",
        "config_label",
        "run_index",
        "recall",
        "qps",
        "gpu_mem_peak_mib",
        "avg_latency_ms",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "stddev_ms",
        "effective_concurrent_queries",
        "stage3_threads",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
        "binary",
        "log",
        "returncode",
        "elapsed_s",
        "command",
        "error_tail",
    ]
    write_rows(results_csv, rows, preferred)
    write_summary(summary_csv, rows, ["system", "dataset", "slots", "target_recall"])
    print(f"[wrote] {results_csv}")
    print(f"[wrote] {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
