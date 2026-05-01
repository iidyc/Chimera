#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_PATHS = {
    "lotte": {
        "query": "dataset/lotte/raw/query.bin",
        "gt": "dataset/lotte/raw/gt.tsv",
    },
    "msmarco": {
        "query": "dataset/msmarco/raw/query.bin",
        "gt": "dataset/msmarco/raw/gt.tsv",
    },
    "hotpot": {
        "query": "dataset/hotpot/raw/query.bin",
        "gt": "dataset/hotpot/raw/gt.tsv",
    },
}
LEGACY_CHIMERA_INDEX_PREFIX = "gpu_" "mvr"


def chimera_index_dir(repo_root: Path, dataset: str) -> Path:
    dataset_root = repo_root / "dataset" / dataset
    candidates = [
        dataset_root / "gpu_search_2m",
        dataset_root / "gpu_search_1m",
        dataset_root / "gpu_search",
        dataset_root / "gpu_search_index",
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_2m",
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_1m",
        dataset_root / LEGACY_CHIMERA_INDEX_PREFIX,
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_index",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "doclens.bin").is_file():
            return candidate
    return candidates[0]


# Fixed operating points chosen from existing v6 sweeps so the LUT ablation
# uses stable, reproducible settings at approximately the requested recalls.
OPERATING_POINTS = {
    0.9: {
        "lotte": {
            "nprobe": 64,
            "k_rank_cluster": 2400,
            "k_rank_all_tokens": 240,
            "itopk_size": 128,
            "overlap_chunks": 5,
        },
        "msmarco": {
            "nprobe": 64,
            "k_rank_cluster": 1200,
            "k_rank_all_tokens": 150,
            "itopk_size": 96,
            "overlap_chunks": 5,
        },
        "hotpot": {
            "nprobe": 160,
            "k_rank_cluster": 3800,
            "k_rank_all_tokens": 360,
            "itopk_size": 192,
            "overlap_chunks": 5,
        },
    },
    0.95: {
        "lotte": {
            "nprobe": 128,
            "k_rank_cluster": 4000,
            "k_rank_all_tokens": 400,
            "itopk_size": 192,
            "overlap_chunks": 5,
        },
        "msmarco": {
            "nprobe": 96,
            "k_rank_cluster": 3200,
            "k_rank_all_tokens": 320,
            "itopk_size": 160,
            "overlap_chunks": 5,
        },
        "hotpot": {
            "nprobe": 448,
            "k_rank_cluster": 28000,
            "k_rank_all_tokens": 2200,
            "itopk_size": 512,
            "overlap_chunks": 5,
        },
    },
}


PROFILE_FIELDS = {
    "profile_total_search_time_ms": r"Total search time\s+:\s+([0-9.]+)\s+ms",
    "profile_total_wall_ms": r"Total wall-clock time:\s+([0-9.]+)\s+ms",
    "profile_stage1_time_ms": r"Stage 1 time:\s+([0-9.]+)\s+ms",
    "profile_cagra_ms": r"CAGRA search\s+:\s+([0-9.]+)\s+ms",
    "profile_ivf_expansion_ms": r"GPU IVF expansion\s+:\s+([0-9.]+)\s+ms",
    "profile_binary_ip_ms": r"Binary IP kernel\s+:\s+([0-9.]+)\s+ms",
    "profile_atomic_agg_ms": r"Aggregation \+ tracking\s+:\s+([0-9.]+)\s+ms",
    "profile_sum_scores_ms": r"Sum doc scores \(sparse\)\s+:\s+([0-9.]+)\s+ms",
    "profile_topk_sort_ms": r"Top-k sort \(sparse\)\s+:\s+([0-9.]+)\s+ms",
    "profile_d2d_ms": r"D2D copy top-k doc IDs\s+:\s+([0-9.]+)\s+ms",
    "profile_memset_ms": r"Memset .*:\s+([0-9.]+)\s+ms",
    "profile_sum_accounted_ms": r"Sum accounted\s+:\s+([0-9.]+)\s+ms",
    "profile_stage23_time_ms": r"Stage 2\+3 time\s+:\s+([0-9.]+)\s+ms",
    "profile_phase_b_binary_ip_total_ms": r"Phase B binary_ip total\s+:\s+([0-9.]+)\s+ms",
    "profile_phase_b_doc_score_total_ms": r"Phase B doc_score total\s+:\s+([0-9.]+)\s+ms",
    "profile_phase_b_total_kernel_ms": r"Phase B total kernel time\s+:\s+([0-9.]+)\s+ms",
    "profile_transfer_total_ms": r"Total transfer:\s+([0-9.]+)\s+ms",
    "profile_h2d_ms": r"H2D:\s+([0-9.]+)\s+ms",
    "profile_d2h_ms": r"D2H:\s+([0-9.]+)\s+ms",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Compare gpu_search with and without LUT at fixed operating "
            "points for Recall@100 ≈ 0.9 and ≈ 0.95 across LoTTE/MSMARCO/HotpotQA."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "Chimera" / "build")
    parser.add_argument(
        "--binary-lut",
        type=Path,
        default=None,
        help="Path to the LUT-enabled gpu_search binary. Defaults to <build-dir>/gpu_search.",
    )
    parser.add_argument(
        "--binary-nolut",
        type=Path,
        default=None,
        help=(
            "Path to the no-LUT gpu_search binary. Defaults to <build-dir>/gpu_search_nolut. "
            "Current tree does not build this target by default."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["lotte", "msmarco", "hotpot"],
        default=["lotte", "msmarco", "hotpot"],
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        choices=[0.9, 0.95],
        default=[0.9, 0.95],
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "profiling" / "gpu_search_lut_ablation")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return parser.parse_args()


def gpu_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def quote_arg(text: str) -> str:
    return subprocess.list2cmdline([text])


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], reuse: bool) -> None:
    if reuse and log_path.exists():
        print(f"[reuse] {log_path}", flush=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(quote_arg(x) for x in cmd)}", flush=True)
    with log_path.open("w") as handle:
        result = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def read_text(path: Path) -> str:
    return path.read_text()


def timed_log_complete(path: Path) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    return (
        re.search(r"Throughput:\s+([0-9.]+)\s+qps", text, re.MULTILINE) is not None
        and re.search(r"Average latency per query:\s+([0-9.]+)\s+ms", text, re.MULTILINE) is not None
        and re.search(r"Recall@\d+:\s+([0-9.]+)", text, re.MULTILINE) is not None
    )


def profile_log_complete(path: Path) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    return (
        "[PROFILE_AVG]" in text
        and re.search(r"Total wall-clock time:\s+([0-9.]+)\s+ms", text, re.MULTILINE) is not None
        and re.search(r"Recall@\d+:\s+([0-9.]+)", text, re.MULTILINE) is not None
    )


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"failed to parse {label}")
    return float(match.group(1))


def maybe_parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def run_timed(
    args: argparse.Namespace,
    binary: Path,
    variant: str,
    dataset: str,
    recall_target: float,
) -> tuple[Path, float, float, float]:
    point = OPERATING_POINTS[recall_target][dataset]
    paths = DATASET_PATHS[dataset]
    log_path = args.output_dir / f"{variant}_{dataset}_r{str(recall_target).replace('.', '')}_timed.log"
    cmd = [
        str(binary),
        "--query", str(args.repo_root / paths["query"]),
        "--gt", str(args.repo_root / paths["gt"]),
        "--index", str(chimera_index_dir(args.repo_root, dataset)),
        "--k", str(args.k),
        "--nprobe", str(point["nprobe"]),
        "--k-rank-cluster", str(point["k_rank_cluster"]),
        "--k-rank-all-tokens", str(point["k_rank_all_tokens"]),
        "--itopk-size", str(point["itopk_size"]),
        "--overlap-chunks", str(point["overlap_chunks"]),
        "--warmup", str(args.warmup),
    ]
    reuse_existing = args.reuse_existing and timed_log_complete(log_path)
    run_logged(cmd, log_path, gpu_env(args.cuda_visible_devices), reuse_existing)

    text = read_text(log_path)
    qps = parse_float(r"Throughput:\s+([0-9.]+)\s+qps", text, f"{variant} qps")
    latency = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, f"{variant} latency")
    recall = parse_float(r"Recall@\d+:\s+([0-9.]+)", text, f"{variant} recall")
    return log_path, qps, latency, recall


def run_profile(
    args: argparse.Namespace,
    binary: Path,
    variant: str,
    dataset: str,
    recall_target: float,
) -> tuple[Path, dict[str, float | None]]:
    point = OPERATING_POINTS[recall_target][dataset]
    paths = DATASET_PATHS[dataset]
    log_path = args.output_dir / f"{variant}_{dataset}_r{str(recall_target).replace('.', '')}_profile.log"
    cmd = [
        str(binary),
        "--query", str(args.repo_root / paths["query"]),
        "--gt", str(args.repo_root / paths["gt"]),
        "--index", str(chimera_index_dir(args.repo_root, dataset)),
        "--k", str(args.k),
        "--nprobe", str(point["nprobe"]),
        "--k-rank-cluster", str(point["k_rank_cluster"]),
        "--k-rank-all-tokens", str(point["k_rank_all_tokens"]),
        "--itopk-size", str(point["itopk_size"]),
        "--overlap-chunks", str(point["overlap_chunks"]),
        "--warmup", str(args.warmup),
        "--profile-eval-all-queries",
    ]
    reuse_existing = args.reuse_existing and profile_log_complete(log_path)
    run_logged(cmd, log_path, gpu_env(args.cuda_visible_devices), reuse_existing)

    text = read_text(log_path)
    metrics = {name: maybe_parse_float(pattern, text) for name, pattern in PROFILE_FIELDS.items()}
    return log_path, metrics


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_qps_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    datasets = [dataset for dataset in ["lotte", "msmarco", "hotpot"] if any(r["dataset"] == dataset for r in rows)]
    targets = [target for target in [0.9, 0.95] if any(float(r["recall_target"]) == target for r in rows)]
    if not datasets or not targets:
        return

    titles = {"lotte": "LoTTE", "msmarco": "MSMARCO", "hotpot": "HotpotQA"}
    labels = {"lut": "with LUT", "nolut": "w/o LUT"}
    colors = {"lut": "#d62728", "nolut": "#1f77b4"}

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
        }
    )
    fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 3.4), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        target_rows = [r for r in rows if float(r["recall_target"]) == target]
        x = list(range(len(datasets)))
        width = 0.36
        grouped = {}
        for offset, variant in [(-width / 2, "lut"), (width / 2, "nolut")]:
            ys = []
            for dataset in datasets:
                match = next(r for r in target_rows if r["dataset"] == dataset and r["variant"] == variant)
                qps = float(match["timed_qps"])
                ys.append(qps)
                grouped.setdefault(dataset, {})[variant] = qps
            ax.bar(
                [v + offset for v in x],
                ys,
                width=width,
                color=colors[variant],
                label=labels[variant],
            )
        ax.set_xticks(x, [titles[d] for d in datasets])
        ax.set_title(f"Recall@100 = {target:.2f}")
        ax.grid(axis="y", alpha=0.3)
        ymax = 0.0
        for xi, dataset in zip(x, datasets):
            lut_qps = grouped[dataset]["lut"]
            nolut_qps = grouped[dataset]["nolut"]
            ymax = max(ymax, lut_qps, nolut_qps)
            gain = (lut_qps / nolut_qps - 1.0) * 100.0
            ax.text(
                xi,
                max(lut_qps, nolut_qps) * 1.03,
                f"+{gain:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylim(0, ymax * 1.42)
    axes[0].set_ylabel("QPS")
    axes[-1].legend(frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"qps_compare.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    binary_lut = args.binary_lut or (args.build_dir / "gpu_search")
    binary_nolut = args.binary_nolut or (args.build_dir / "gpu_search_nolut")

    if not binary_lut.exists():
        raise FileNotFoundError(f"missing LUT-enabled binary: {binary_lut}")
    if not binary_nolut.exists():
        raise FileNotFoundError(
            f"missing no-LUT binary: {binary_nolut}. "
            "The current tree does not build gpu_search_nolut by default."
        )

    rows: list[dict[str, str]] = []
    for recall_target in args.recall_targets:
        for dataset in args.datasets:
            point = OPERATING_POINTS[recall_target][dataset]
            for variant, binary in [("lut", binary_lut), ("nolut", binary_nolut)]:
                timed_log, qps, latency, recall = run_timed(args, binary, variant, dataset, recall_target)
                if args.timed_only:
                    profile_log = Path()
                    metrics = {name: None for name in PROFILE_FIELDS}
                else:
                    profile_log, metrics = run_profile(args, binary, variant, dataset, recall_target)
                row = {
                    "variant": variant,
                    "dataset": dataset,
                    "recall_target": str(recall_target),
                    "timed_qps": str(qps),
                    "timed_avg_latency_ms": str(latency),
                    "timed_recall": str(recall),
                    "timed_log": str(timed_log),
                    "profile_log": str(profile_log),
                    "nprobe": str(point["nprobe"]),
                    "k_rank_cluster": str(point["k_rank_cluster"]),
                    "k_rank_all_tokens": str(point["k_rank_all_tokens"]),
                    "itopk_size": str(point["itopk_size"]),
                    "overlap_chunks": str(point["overlap_chunks"]),
                }
                row.update({k: "" if v is None else str(v) for k, v in metrics.items()})
                rows.append(row)
                print(
                    f"[done] dataset={dataset} target={recall_target:.2f} variant={variant} "
                    f"qps={qps:.3f} latency_ms={latency:.3f} recall={recall:.6f}"
                , flush=True)

    fieldnames = [
        "variant",
        "dataset",
        "recall_target",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
        "timed_qps",
        "timed_avg_latency_ms",
        "timed_recall",
        "timed_log",
        "profile_log",
        *PROFILE_FIELDS.keys(),
    ]
    write_csv(args.output_dir / "comparison.csv", rows, fieldnames)
    render_qps_plot(rows, args.output_dir)
    script_copy = args.output_dir / Path(__file__).name
    script_copy.write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
