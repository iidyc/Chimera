#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


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


SYSTEMS = {
    "v6_lite": {
        "binary": "gpu_search_v6_lite",
        "extra_args": [],
    },
    "v6_turbo": {
        "binary": "gpu_search_v6_turbo",
        "extra_args": [],
    },
    "v7_lite": {
        "binary": "gpu_search_v7_lite",
        "extra_args": [],
    },
    "v7_unalign": {
        "binary": "gpu_search_v7_unalign",
        "extra_args": [],
    },
    "v8": {
        "binary": "gpu_search_v8",
        "extra_args": ["--concurrent-queries", "{v8_slots}"],
    },
}


# Configs are fixed k=100 operating points selected from prior sweeps.
# They are intentionally a little above the requested target recall.
TARGET_CONFIGS = [
    TargetConfig("lotte", "0.90", "c1", 64, 2400, 240, 128, 5),
    TargetConfig("lotte", "0.95", "f1", 128, 4000, 400, 192, 5),
    TargetConfig("msmarco", "0.90", "b2", 64, 1800, 180, 96, 4),
    TargetConfig("msmarco", "0.95", "q2", 96, 3200, 320, 160, 6),
    TargetConfig("hotpot", "0.90", "q4", 160, 5000, 500, 224, 6),
    TargetConfig("hotpot", "0.95", "q9", 384, 28000, 1800, 448, 5),
]


METRIC_PATTERNS = {
    "qps": re.compile(r"Throughput:\s+([0-9.]+)\s+qps"),
    "avg_latency_ms": re.compile(r"Average latency per query:\s+([0-9.]+)\s+ms"),
    "recall": re.compile(r"Recall@\d+:\s+([0-9.]+)"),
    "peak_gpu_mib": re.compile(r"\[GPU_MEM\].*peak=([0-9.]+)\s+MiB"),
    "current_gpu_mib": re.compile(r"\[GPU_MEM\].*current=([0-9.]+)\s+MiB"),
    "stage3_threads": re.compile(r"stage3_threads=(\d+)"),
    "effective_concurrent_queries": re.compile(r"effective_concurrent_queries=(\d+)"),
    "end_to_end_ms": re.compile(r"End-to-end measured time:\s+([0-9.]+)\s+ms"),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare gpu_search_v6_lite, gpu_search_v7_lite, and gpu_search_v8 "
            "at fixed k=100 recall~0.90 and recall~0.95 operating points. "
            "v8 is run with 8 concurrent query slots by default."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "profiling" / "v6lite_v7lite_v8_slots_recall_targets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted({cfg.dataset for cfg in TARGET_CONFIGS}),
        default=sorted({cfg.dataset for cfg in TARGET_CONFIGS}),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted({cfg.target_recall for cfg in TARGET_CONFIGS}),
        default=sorted({cfg.target_recall for cfg in TARGET_CONFIGS}),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(SYSTEMS.keys()),
        default=list(SYSTEMS.keys()),
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--omp-threads", type=int, default=24)
    parser.add_argument("--v8-slots", type=int, default=8)
    parser.add_argument(
        "--stage3-threads",
        type=int,
        default=0,
        help="Optional per-slot Stage 3 OpenMP threads. 0 lets the binary auto-split OMP threads.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the requested binaries before running benchmarks.",
    )
    return parser.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def selected_configs(args: argparse.Namespace) -> list[TargetConfig]:
    dataset_set = set(args.datasets)
    target_set = set(args.targets)
    return [
        cfg for cfg in TARGET_CONFIGS
        if cfg.dataset in dataset_set and cfg.target_recall in target_set
    ]


def ensure_inputs(repo_root: Path, cfg: TargetConfig) -> None:
    required = [
        repo_root / "dataset" / cfg.dataset / "raw" / "query.bin",
        repo_root / "dataset" / cfg.dataset / "raw" / "gt.tsv",
        repo_root / "dataset" / cfg.dataset / "gpu_mvr_2m",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def build_binaries(args: argparse.Namespace) -> None:
    targets = [SYSTEMS[system]["binary"] for system in args.systems]
    cmd = ["cmake", "--build", str(args.build_dir), "--target", *targets, "-j", "8"]
    print(f"[build] {quote_cmd(cmd)}", flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=args.repo_root, check=True)


def command_for(
    args: argparse.Namespace,
    cfg: TargetConfig,
    system: str,
) -> list[str]:
    binary = args.build_dir / SYSTEMS[system]["binary"]
    cmd = [
        str(binary),
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
    ]
    for item in SYSTEMS[system]["extra_args"]:
        cmd.append(item.format(v8_slots=args.v8_slots))
    if system == "v8" and args.stage3_threads > 0:
        cmd += ["--stage3-threads", str(args.stage3_threads)]
    return cmd


def parse_metric(name: str, text: str) -> str:
    match = METRIC_PATTERNS[name].search(text)
    return match.group(1) if match else ""


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    return {name: parse_metric(name, text) for name in METRIC_PATTERNS}


def run_one(args: argparse.Namespace, cfg: TargetConfig, system: str) -> dict[str, object]:
    ensure_inputs(args.repo_root, cfg)
    binary = args.build_dir / SYSTEMS[system]["binary"]
    if not binary.exists():
        raise FileNotFoundError(f"missing binary: {binary}")

    log_dir = args.output_dir / "logs" / cfg.dataset / f"r{cfg.target_recall.replace('.', '')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{system}_{cfg.label}_slots{args.v8_slots if system == 'v8' else 1}.log"

    cmd = command_for(args, cfg, system)
    row: dict[str, object] = {
        "dataset": cfg.dataset,
        "target_recall": cfg.target_recall,
        "config_label": cfg.label,
        "system": system,
        "v8_slots": args.v8_slots if system == "v8" else "",
        "k": args.k,
        "nq": args.nq,
        "warmup": args.warmup,
        "nprobe": cfg.nprobe,
        "k_rank_cluster": cfg.k_rank_cluster,
        "k_rank_all_tokens": cfg.k_rank_all_tokens,
        "itopk_size": cfg.itopk_size,
        "overlap_chunks": cfg.overlap_chunks,
        "log": str(log_path),
        "command": quote_cmd(cmd),
    }

    if args.dry_run:
        print(f"[dry-run] {quote_cmd(cmd)}")
        row.update(returncode="", elapsed_s="")
        return row

    if args.reuse_existing and log_path.exists():
        print(f"[reuse] {log_path}", flush=True)
        row.update(returncode=0, elapsed_s="")
    else:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"[run] {cfg.dataset} r{cfg.target_recall} {cfg.label} {system}", flush=True)
        t0 = time.time()
        with log_path.open("w") as handle:
            handle.write(f"[driver] command={quote_cmd(cmd)}\n")
            handle.write(f"[driver] omp_num_threads={env.get('OMP_NUM_THREADS', '')}\n")
            handle.write(f"[driver] cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=args.repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
        row.update(returncode=proc.returncode, elapsed_s=f"{time.time() - t0:.3f}")
        if proc.returncode != 0:
            text = log_path.read_text(errors="replace")
            row["error_tail"] = "\n".join(text.splitlines()[-12:])
            print(f"[fail] {log_path} rc={proc.returncode}", flush=True)
            return row

    metrics = parse_log(log_path)
    row.update(metrics)
    try:
        row["meets_target"] = float(row.get("recall") or 0.0) >= float(cfg.target_recall)
    except ValueError:
        row["meets_target"] = False
    print(
        "[done] "
        f"{cfg.dataset} r{cfg.target_recall} {system} "
        f"qps={row.get('qps')} recall={row.get('recall')} mem={row.get('peak_gpu_mib')}",
        flush=True,
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "target_recall", "config_label", "system", "v8_slots",
        "qps", "avg_latency_ms", "recall", "meets_target",
        "peak_gpu_mib", "current_gpu_mib", "end_to_end_ms",
        "effective_concurrent_queries", "stage3_threads",
        "k", "nq", "warmup", "nprobe", "k_rank_cluster",
        "k_rank_all_tokens", "itopk_size", "overlap_chunks",
        "returncode", "elapsed_s", "log", "command", "error_tail",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_best_summary(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("returncode")) not in {"0", ""}:
            continue
        if not row.get("qps"):
            continue
        groups.setdefault((str(row["dataset"]), str(row["target_recall"])), []).append(row)

    summary_rows = []
    for (dataset, target), group in sorted(groups.items()):
        valid = [row for row in group if row.get("meets_target") is True]
        candidates = valid if valid else group
        best = max(candidates, key=lambda row: float(row["qps"]))
        baseline = next((row for row in group if row["system"] == "v7_lite"), None)
        speedup_vs_v7 = ""
        if baseline and baseline.get("qps"):
            speedup_vs_v7 = f"{float(best['qps']) / float(baseline['qps']):.4f}"
        summary_rows.append({
            "dataset": dataset,
            "target_recall": target,
            "best_system": best["system"],
            "best_qps": best["qps"],
            "best_avg_latency_ms": best.get("avg_latency_ms", ""),
            "best_recall": best.get("recall", ""),
            "best_peak_gpu_mib": best.get("peak_gpu_mib", ""),
            "best_v8_slots": best.get("v8_slots", ""),
            "speedup_vs_v7_lite": speedup_vs_v7,
            "all_met_target": all(row.get("meets_target") is True for row in group),
        })

    with path.open("w", newline="") as handle:
        fieldnames = [
            "dataset", "target_recall", "best_system", "best_qps",
            "best_avg_latency_ms", "best_recall", "best_peak_gpu_mib",
            "best_v8_slots", "speedup_vs_v7_lite", "all_met_target",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.build:
        build_binaries(args)

    rows: list[dict[str, object]] = []
    result_csv = args.output_dir / "results.csv"
    best_csv = args.output_dir / "best_summary.csv"

    for cfg in selected_configs(args):
        for system in args.systems:
            rows.append(run_one(args, cfg, system))
            if not args.dry_run:
                write_csv(result_csv, rows)
                write_best_summary(best_csv, rows)

    if args.dry_run:
        return 0

    write_csv(result_csv, rows)
    write_best_summary(best_csv, rows)
    print(f"[wrote] {result_csv}")
    print(f"[wrote] {best_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
