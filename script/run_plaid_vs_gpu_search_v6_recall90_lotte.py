#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SystemSummary:
    system: str
    config: str
    recall: float
    qps: float
    avg_latency_ms: float
    raw_total_ms: float
    centroid_search_raw_ms: float
    ivf_data_transfer_raw_ms: float
    ivf_scanning_raw_ms: float
    doc_score_aggregate_raw_ms: float
    reranking_raw_ms: float

    @property
    def scale_to_latency(self) -> float:
        if self.raw_total_ms <= 0.0:
            return 0.0
        return self.avg_latency_ms / self.raw_total_ms

    def scaled_row(self) -> dict:
        scale = self.scale_to_latency
        return {
            "system": self.system,
            "config": self.config,
            "recall": self.recall,
            "qps": self.qps,
            "avg_latency_ms": self.avg_latency_ms,
            "raw_total_ms": self.raw_total_ms,
            "scale_to_latency": scale,
            "centroid_search_ms": self.centroid_search_raw_ms * scale,
            "ivf_data_transfer_ms": self.ivf_data_transfer_raw_ms * scale,
            "ivf_scanning_ms": self.ivf_scanning_raw_ms * scale,
            "doc_score_aggregate_ms": self.doc_score_aggregate_raw_ms * scale,
            "reranking_ms": self.reranking_raw_ms * scale,
            "centroid_search_raw_ms": self.centroid_search_raw_ms,
            "ivf_data_transfer_raw_ms": self.ivf_data_transfer_raw_ms,
            "ivf_scanning_raw_ms": self.ivf_scanning_raw_ms,
            "doc_score_aggregate_raw_ms": self.doc_score_aggregate_raw_ms,
            "reranking_raw_ms": self.reranking_raw_ms,
        }


GPU_CONFIG = {
    "label": "f4",
    "nprobe": 96,
    "k_rank_cluster": 1800,
    "k_rank_all_tokens": 200,
    "itopk_size": 128,
    "overlap_chunks": 4,
}

PLAID_CONFIG = {
    "label": "c4m",
    "ncells": 2,
    "ndocs": 3000,
}

PHASE_ORDER = [
    ("Centroid Search", "centroid_search_ms", "#4C78A8"),
    ("IVF Data Transfer", "ivf_data_transfer_ms", "#F58518"),
    ("IVF Scanning", "ivf_scanning_ms", "#54A24B"),
    ("Doc Score Aggregate", "doc_score_aggregate_ms", "#E45756"),
    ("Reranking", "reranking_ms", "#B279A2"),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description=(
            "Reproduce the LoTTE recall~0.9 latency breakdown comparison between "
            "PLAID (ColBERT) and gpu_search_v6/gpu_search_v8 using fixed configs."
        )
    )
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--plot-root", type=Path)
    p.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    p.add_argument(
        "--colbert-python",
        type=Path,
        default=Path(os.environ.get("COLBERT_PYTHON", "python")),
    )
    p.add_argument(
        "--plaid-variant",
        choices=["cpu_hosted", "gpu_resident"],
        default="cpu_hosted",
        help="PLAID storage mode to benchmark.",
    )
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    p.add_argument(
        "--systems",
        nargs="+",
        choices=["plaid", "gpu_search_v6", "gpu_search_v8"],
        default=["plaid", "gpu_search_v6", "gpu_search_v8"],
        help="Systems to include in the comparison.",
    )
    return p.parse_args()


def default_output_dir(repo_root: Path, plaid_variant: str, k: int) -> Path:
    run_name = (
        f"plaid_vs_gpu_search_recall90_k{k}"
        if plaid_variant == "cpu_hosted"
        else f"plaid_gpu_only_vs_gpu_search_recall90_k{k}"
    )
    return repo_root / "profiling" / "lotte" / run_name


def default_plot_root(repo_root: Path, plaid_variant: str, k: int) -> Path:
    plot_name = (
        f"lotte_plaid_vs_gpu_search_recall90_k{k}_latency_breakdown"
        if plaid_variant == "cpu_hosted"
        else f"lotte_plaid_gpu_only_vs_gpu_search_recall90_k{k}_latency_breakdown"
    )
    return repo_root / "plot" / plot_name


def plaid_labels(plaid_variant: str) -> tuple[str, str]:
    if plaid_variant == "gpu_resident":
        return "plaid_gpu_resident", "PLAID GPU-only"
    return "PLAID", "PLAID"


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None, reuse: bool = False) -> None:
    if reuse and log_path.exists():
        print(f"[reuse] {log_path}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(shlex_quote(x) for x in cmd)}")
    with log_path.open("w") as handle:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def shlex_quote(text: str) -> str:
    return subprocess.list2cmdline([text])


def read_text(path: Path) -> str:
    return path.read_text()


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"failed to parse {label}")
    return float(match.group(1))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_gpu_timed(args: argparse.Namespace, system: str) -> tuple[Path, float, float, float]:
    binary = args.build_dir / system
    if not binary.exists():
        raise FileNotFoundError(f"missing {system} binary: {binary}")

    log_path = args.output_dir / f"{system}_{GPU_CONFIG['label']}_timed.log"
    cmd = [
        str(binary),
        "--query", str(args.repo_root / "dataset" / "lotte" / "raw" / "query.bin"),
        "--gt", str(args.repo_root / "dataset" / "lotte" / "raw" / "gt.tsv"),
        "--index", str(args.repo_root / "dataset" / "lotte" / "gpu_mvr_2m"),
        "--k", str(args.k),
        "--nprobe", str(GPU_CONFIG["nprobe"]),
        "--k-rank-cluster", str(GPU_CONFIG["k_rank_cluster"]),
        "--k-rank-all-tokens", str(GPU_CONFIG["k_rank_all_tokens"]),
        "--itopk-size", str(GPU_CONFIG["itopk_size"]),
        "--overlap-chunks", str(GPU_CONFIG["overlap_chunks"]),
        "--nq", "-1",
        "--warmup", str(args.warmup),
    ]
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_logged(cmd, log_path, env=env, reuse=args.reuse_existing)

    text = read_text(log_path)
    qps = parse_float(r"Throughput:\s+([0-9.]+)\s+qps", text, f"{system} qps")
    latency = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, f"{system} latency")
    recall = parse_float(r"Recall@100:\s+([0-9.]+)", text, f"{system} recall")
    return log_path, qps, latency, recall


def run_gpu_profile(args: argparse.Namespace, system: str) -> tuple[Path, dict[str, float]]:
    binary = args.build_dir / system
    use_avg_profile = system in {"gpu_search_v6", "gpu_search_v8"}
    log_path = args.output_dir / (
        f"{system}_{GPU_CONFIG['label']}_profile_avg.log"
        if use_avg_profile else
        f"{system}_{GPU_CONFIG['label']}_profile.log"
    )
    cmd = [
        str(binary),
        "--query", str(args.repo_root / "dataset" / "lotte" / "raw" / "query.bin"),
        "--gt", str(args.repo_root / "dataset" / "lotte" / "raw" / "gt.tsv"),
        "--index", str(args.repo_root / "dataset" / "lotte" / "gpu_mvr_2m"),
        "--k", str(args.k),
        "--nprobe", str(GPU_CONFIG["nprobe"]),
        "--k-rank-cluster", str(GPU_CONFIG["k_rank_cluster"]),
        "--k-rank-all-tokens", str(GPU_CONFIG["k_rank_all_tokens"]),
        "--itopk-size", str(GPU_CONFIG["itopk_size"]),
        "--overlap-chunks", str(GPU_CONFIG["overlap_chunks"]),
        "--nq", "-1",
        "--warmup", str(args.warmup),
    ]
    if use_avg_profile:
        cmd.append("--profile-eval-all-queries")
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_logged(cmd, log_path, env=env, reuse=args.reuse_existing)

    text = read_text(log_path)
    prefix = r"\[PROFILE_AVG\]" if use_avg_profile else r"\[PROFILE\]"
    stage1 = parse_float(prefix + r" Stage 1 time:\s+([0-9.]+)\s+ms", text, f"{system} stage1")
    cagra = parse_float(prefix + r"\s+1\. CAGRA search\s+:\s+([0-9.]+)\s+ms", text, f"{system} cagra")
    expansion = parse_float(prefix + r"\s+2\. GPU IVF expansion\s+:\s+([0-9.]+)\s+ms", text, f"{system} expansion")
    binary_ip = parse_float(prefix + r"\s+3\. Binary IP kernel\s+:\s+([0-9.]+)\s+ms", text, f"{system} binary ip")
    total = parse_float(prefix + r" Total search time\s+:\s+([0-9.]+)\s+ms", text, f"{system} total search")

    centroid = cagra
    ivf_scanning = expansion + binary_ip
    doc_score_aggregate = max(stage1 - centroid - ivf_scanning, 0.0)
    reranking = max(total - stage1, 0.0)
    return log_path, {
        "raw_total_ms": total,
        "centroid_search_raw_ms": centroid,
        "ivf_data_transfer_raw_ms": 0.0,
        "ivf_scanning_raw_ms": ivf_scanning,
        "doc_score_aggregate_raw_ms": doc_score_aggregate,
        "reranking_raw_ms": reranking,
    }


def run_plaid_search(args: argparse.Namespace) -> tuple[Path, Path, Path, float, float, float, dict[str, float]]:
    plaid_impl_label, _ = plaid_labels(args.plaid_variant)
    plaid_prefix = f"{plaid_impl_label}_{PLAID_CONFIG['label']}"
    timed_csv = args.output_dir / f"{plaid_prefix}_timed.csv"
    profile_csv = args.output_dir / f"{plaid_prefix}_profile.csv"
    log_path = args.output_dir / f"{plaid_prefix}.log"

    if not (args.reuse_existing and timed_csv.exists() and profile_csv.exists() and log_path.exists()):
        ensure_parent(log_path)
        if timed_csv.exists():
            timed_csv.unlink()
        if profile_csv.exists():
            profile_csv.unlink()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.repo_root / "ColBERT")
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        cmd = [
            str(args.colbert_python),
            str(args.repo_root / "ColBERT" / "experiment" / "search.py"),
            "--root-path", str(args.repo_root / "dataset" / "lotte"),
            "--experiment-name", "colbert",
            "--index-name", "4bit",
            "--query-path", str(args.repo_root / "dataset" / "lotte" / "raw" / "query.bin"),
            "--ground-truth-path", str(args.repo_root / "dataset" / "lotte" / "raw" / "gt.tsv"),
            "--output-csv", str(timed_csv),
            "--pairs", f"{PLAID_CONFIG['ncells']},{PLAID_CONFIG['ndocs']}",
            "--k", str(args.k),
            "--warmup", str(args.warmup),
            "--dataset-name", "lotte",
            "--implementation-label", plaid_impl_label,
            "--profile-breakdown-csv", str(profile_csv),
        ]
        if args.plaid_variant == "gpu_resident":
            cmd.extend([
                "--compressed-embeddings-storage", "gpu",
                "--gpu-index-resident",
            ])
        run_logged(cmd, log_path, env=env, reuse=False)
    else:
        print(f"[reuse] {log_path}")

    with timed_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one timed PLAID row in {timed_csv}, got {len(rows)}")
    row = rows[0]
    avg_time_s = float(row["avg_time_s"])
    qps = float(row["qps"])
    recall = float(row["recall"])

    profile_rows = []
    with profile_csv.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["ncells"]) != PLAID_CONFIG["ncells"] or int(row["ndocs"]) != PLAID_CONFIG["ndocs"]:
                continue
            profile_rows.append(row)
    if not profile_rows:
        raise ValueError(f"no PLAID profile rows found in {profile_csv}")

    stage_metrics = {
        row["metric_name"]: float(row["avg_ms"])
        for row in profile_rows
        if row["metric_type"] == "stage"
    }
    transfer_metrics = {
        row["metric_name"]: float(row["avg_ms"])
        for row in profile_rows
        if row["metric_type"] == "transfer_h2d"
    }

    raw_total = stage_metrics["search.total"]
    centroid = stage_metrics["candidate.centroid_score"] + stage_metrics["candidate.cell_select"]
    ivf_transfer = sum(transfer_metrics.values())
    ivf_scanning = stage_metrics["candidate.ivf_lookup"] + stage_metrics["candidate.pid_dedup"]
    doc_score_aggregate = max(raw_total - centroid - ivf_transfer - ivf_scanning, 0.0)

    return (
        log_path,
        timed_csv,
        profile_csv,
        qps,
        avg_time_s * 1000.0,
        recall,
        {
            "raw_total_ms": raw_total,
            "centroid_search_raw_ms": centroid,
            "ivf_data_transfer_raw_ms": ivf_transfer,
            "ivf_scanning_raw_ms": ivf_scanning,
            "doc_score_aggregate_raw_ms": doc_score_aggregate,
            "reranking_raw_ms": 0.0,
        },
    )


def write_comparison_csv(output_csv: Path, rows: list[dict]) -> None:
    ensure_parent(output_csv)
    fieldnames = [
        "system",
        "config",
        "recall",
        "qps",
        "avg_latency_ms",
        "raw_total_ms",
        "scale_to_latency",
        "centroid_search_ms",
        "ivf_data_transfer_ms",
        "ivf_scanning_ms",
        "doc_score_aggregate_ms",
        "reranking_ms",
        "centroid_search_raw_ms",
        "ivf_data_transfer_raw_ms",
        "ivf_scanning_raw_ms",
        "doc_score_aggregate_raw_ms",
        "reranking_raw_ms",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_latency_breakdown(plot_root: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    systems = [row["system"] for row in rows]
    x = list(range(len(rows)))
    bottoms = [0.0 for _ in rows]

    for phase_label, key, color in PHASE_ORDER:
        values = [float(row[key]) for row in rows]
        ax.bar(x, values, bottom=bottoms, label=phase_label, color=color, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_ylabel("Average Query Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{row['system']}\n{row['config']}\nRecall={float(row['recall']):.3f}\nQPS={float(row['qps']):.1f}"
        for row in rows
    ])
    compared = " vs ".join(row["system"] for row in rows)
    ax.set_title(f"LoTTE: {compared} at Recall@100 ≈ 0.9", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for idx, row in enumerate(rows):
        total = float(row["avg_latency_ms"])
        ax.text(idx, total + max(0.05, total * 0.02), f"{total:.2f} ms", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plot_root.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(plot_root.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.repo_root, args.plaid_variant, args.k)
    if args.plot_root is None:
        args.plot_root = default_plot_root(args.repo_root, args.plaid_variant, args.k)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SystemSummary] = []
    extra_paths: list[str] = []

    if "plaid" in args.systems:
        _, plaid_system_label = plaid_labels(args.plaid_variant)
        plaid_log, plaid_timed_csv, plaid_profile_csv, plaid_qps, plaid_latency, plaid_recall, plaid_breakdown = run_plaid_search(args)
        summaries.append(
            SystemSummary(
                system=plaid_system_label,
                config=f"{PLAID_CONFIG['label']} (ncells={PLAID_CONFIG['ncells']}, ndocs={PLAID_CONFIG['ndocs']})",
                recall=plaid_recall,
                qps=plaid_qps,
                avg_latency_ms=plaid_latency,
                **plaid_breakdown,
            )
        )
        extra_paths.extend([
            f"[logs] plaid={plaid_log}",
            f"[csv] plaid_timed={plaid_timed_csv}",
            f"[csv] plaid_profile={plaid_profile_csv}",
        ])

    for system in ("gpu_search_v6", "gpu_search_v8"):
        if system not in args.systems:
            continue
        gpu_timed_log, gpu_qps, gpu_latency, gpu_recall = run_gpu_timed(args, system)
        gpu_profile_log, gpu_breakdown = run_gpu_profile(args, system)
        summaries.append(
            SystemSummary(
                system=system,
                config=GPU_CONFIG["label"],
                recall=gpu_recall,
                qps=gpu_qps,
                avg_latency_ms=gpu_latency,
                **gpu_breakdown,
            )
        )
        extra_paths.extend([
            f"[logs] {system}_timed={gpu_timed_log}",
            f"[logs] {system}_profile={gpu_profile_log}",
        ])

    gpu_recalls = {
        summary.system: summary.recall
        for summary in summaries
        if summary.system in {"gpu_search_v6", "gpu_search_v8"}
    }
    if len(gpu_recalls) == 2:
        v6_recall = gpu_recalls["gpu_search_v6"]
        v8_recall = gpu_recalls["gpu_search_v8"]
        recall_delta = abs(v6_recall - v8_recall)
        if recall_delta > 1e-3:
            raise RuntimeError(
                "gpu_search_v6 and gpu_search_v8 diverged more than expected for this comparison, "
                f"got {v6_recall:.12f} vs {v8_recall:.12f} (delta={recall_delta:.6f})"
            )
        if recall_delta > 0.0:
            print(
                f"[warn] gpu_search recall mismatch is small but non-zero: "
                f"v6={v6_recall:.6f} v8={v8_recall:.6f} delta={recall_delta:.6f}"
            )

    rows = [summary.scaled_row() for summary in summaries]
    comparison_csv = args.output_dir / "comparison.csv"
    write_comparison_csv(comparison_csv, rows)
    plot_latency_breakdown(args.plot_root, rows)

    print(f"[done] comparison_csv={comparison_csv}")
    print(f"[done] plot_png={args.plot_root.with_suffix('.png')}")
    print(f"[done] plot_pdf={args.plot_root.with_suffix('.pdf')}")
    for line in extra_paths:
        print(line)
    for summary in summaries:
        print(
            f"[summary] system={summary.system} config={summary.config} "
            f"recall={summary.recall:.6f} qps={summary.qps:.3f} latency_ms={summary.avg_latency_ms:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
