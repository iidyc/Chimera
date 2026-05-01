#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from gpu_mvr_experiment_lib import (
    DATASETS,
    common_env,
    ensure_binaries,
    normalize_datasets,
    quote_cmd,
    repo_root_from_script,
    reset_run_dir,
    write_rows,
    write_summary,
    SYSTEMS,
)


@dataclass(frozen=True)
class TargetConfig:
    dataset: str
    target_recall: str
    label: str
    nprobe: int
    k_rank_cluster: int
    k_rank_all_tokens: int
    itopk_size: int
    overlap_chunks: int


TARGET_CONFIGS = [
    TargetConfig("lotte", "0.90", "c1", 64, 2400, 240, 128, 5),
    TargetConfig("lotte", "0.95", "f1", 128, 4000, 400, 192, 5),
    TargetConfig("msmarco", "0.90", "b2", 64, 1800, 180, 96, 4),
    TargetConfig("msmarco", "0.95", "q2", 96, 3200, 320, 160, 6),
    TargetConfig("hotpot", "0.90", "q4", 160, 5000, 500, 224, 6),
    TargetConfig("hotpot", "0.95", "q9", 384, 28000, 1800, 448, 5),
]

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
        description="Run gpu_search_v9 slot ablation at fixed Recall@100 ~= 0.90 and 0.95 points."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-root", type=Path, default=repo_root / "profiling" / "gpu_mvr_v9_slot_recall_ablation")
    parser.add_argument("--run-id", default="20260429")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--targets", nargs="+", choices=["0.90", "0.95"], default=["0.90", "0.95"])
    parser.add_argument("--slots", nargs="+", type=int, default=[1, 8])
    parser.add_argument("-N", "--num-runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stage3-threads", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--omp-num-threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def selected_configs(args: argparse.Namespace) -> list[TargetConfig]:
    datasets = set(normalize_datasets(args.datasets))
    targets = set(args.targets)
    return [cfg for cfg in TARGET_CONFIGS if cfg.dataset in datasets and cfg.target_recall in targets]


def command_for(args: argparse.Namespace, cfg: TargetConfig, slots: int) -> list[str]:
    cmd = [
        str(args.build_dir / "gpu_search_v9"),
        "--query", str(args.repo_root / "dataset" / cfg.dataset / "raw" / "query.bin"),
        "--gt", str(args.repo_root / "dataset" / cfg.dataset / "raw" / "gt.tsv"),
        "--index", str(args.repo_root / "dataset" / cfg.dataset / "gpu_mvr_2m"),
        "--k", str(args.k),
        "--nq", str(args.nq),
        "--warmup", str(args.warmup),
        "--nprobe", str(cfg.nprobe),
        "--k-rank-cluster", str(cfg.k_rank_cluster),
        "--k-rank-all-tokens", str(cfg.k_rank_all_tokens),
        "--itopk-size", str(cfg.itopk_size),
        "--overlap-chunks", str(cfg.overlap_chunks),
        "--concurrent-queries", str(slots),
    ]
    if args.stage3_threads > 0:
        cmd += ["--stage3-threads", str(args.stage3_threads)]
    return cmd


def run_one(args: argparse.Namespace, cfg: TargetConfig, slots: int, run_index: int, env: dict[str, str]) -> dict[str, str]:
    log_path = (
        args.output_root / "logs" / args.run_id / cfg.dataset / f"r{cfg.target_recall.replace('.', '')}"
        / f"v9_slots{slots}_run{run_index}.log"
    )
    cmd = command_for(args, cfg, slots)
    row = {
        "system": "v9",
        "dataset": cfg.dataset,
        "target_recall": cfg.target_recall,
        "slots": str(slots),
        "config_label": cfg.label,
        "run_index": str(run_index),
        "k": str(args.k),
        "nq": str(args.nq),
        "warmup": str(args.warmup),
        "nprobe": str(cfg.nprobe),
        "k_rank_cluster": str(cfg.k_rank_cluster),
        "k_rank_all_tokens": str(cfg.k_rank_all_tokens),
        "itopk_size": str(cfg.itopk_size),
        "overlap_chunks": str(cfg.overlap_chunks),
        "binary": "gpu_search_v9",
        "log": str(log_path),
        "command": quote_cmd(cmd),
    }
    if args.dry_run:
        print(f"[dry-run] {quote_cmd(cmd)}", flush=True)
        return row

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[run] dataset={cfg.dataset} target={cfg.target_recall} slots={slots} run={run_index}",
        flush=True,
    )
    start = time.time()
    with log_path.open("w") as handle:
        handle.write(f"[driver] command={quote_cmd(cmd)}\n")
        handle.write(f"[driver] dataset={cfg.dataset}\n")
        handle.write(f"[driver] system=v9\n")
        handle.write(f"[driver] target_recall={cfg.target_recall}\n")
        handle.write(f"[driver] config_label={cfg.label}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.repo_root,
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
        row["meets_target"] = str(float(row.get("recall") or "0") >= float(cfg.target_recall))
        print(
            f"[done] dataset={cfg.dataset} target={cfg.target_recall} slots={slots} "
            f"qps={row.get('qps')} recall={row.get('recall')}",
            flush=True,
        )
    else:
        row["error_tail"] = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
        print(f"[fail] {log_path} rc={proc.returncode}", flush=True)
    return row


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output_root = args.output_root.resolve()
    ensure_binaries(args.build_dir, [SYSTEMS["v9"]], args.dry_run)
    if args.force:
        reset_run_dir(args.output_root / "logs" / args.run_id, True)

    env = common_env(args.cuda_visible_devices, args.omp_num_threads)
    rows: list[dict[str, str]] = []
    results_csv = args.output_root / f"v9_slot_recall_results_{args.run_id}.csv"
    summary_csv = args.output_root / f"v9_slot_recall_summary_{args.run_id}.csv"
    preferred = [
        "system", "dataset", "target_recall", "slots", "config_label", "run_index",
        "recall", "meets_target", "qps", "gpu_mem_peak_mib", "avg_latency_ms",
        "min_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "stddev_ms",
        "effective_concurrent_queries", "stage3_threads", "nprobe", "k_rank_cluster",
        "k_rank_all_tokens", "itopk_size", "overlap_chunks", "binary", "log",
        "returncode", "elapsed_s", "command", "error_tail",
    ]
    for run_index in range(1, args.num_runs + 1):
        for cfg in selected_configs(args):
            for slots in args.slots:
                rows.append(run_one(args, cfg, slots, run_index, env))
                if not args.dry_run:
                    write_rows(results_csv, rows, preferred)
                    write_summary(summary_csv, rows, ["system", "dataset", "target_recall", "slots"])

    if not args.dry_run:
        print(f"[wrote] {results_csv}")
        print(f"[wrote] {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
