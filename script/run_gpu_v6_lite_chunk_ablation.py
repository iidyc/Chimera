#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


DATASET_PATHS = {
    "lotte": {
        "query": "dataset/lotte/raw/query.bin",
        "gt": "dataset/lotte/raw/gt.tsv",
        "index": "dataset/lotte/gpu_mvr_2m",
    },
    "msmarco": {
        "query": "dataset/msmarco/raw/query.bin",
        "gt": "dataset/msmarco/raw/gt.tsv",
        "index": "dataset/msmarco/gpu_mvr_2m",
    },
    "hotpot": {
        "query": "dataset/hotpot/raw/query.bin",
        "gt": "dataset/hotpot/raw/gt.tsv",
        "index": "dataset/hotpot/gpu_mvr_2m",
    },
}


# Fixed operating points chosen from the v6/v6_lite sweeps. The chunk ablation
# defaults to the Recall@100=0.95 target and changes only overlap_chunks.
OPERATING_POINTS = {
    0.90: {
        "lotte": {
            "label": "c2",
            "nprobe": 96,
            "k_rank_cluster": 1800,
            "k_rank_all_tokens": 200,
            "itopk_size": 128,
        },
        "msmarco": {
            "label": "b4",
            "nprobe": 64,
            "k_rank_cluster": 1800,
            "k_rank_all_tokens": 180,
            "itopk_size": 96,
        },
        "hotpot": {
            "label": "g2",
            "nprobe": 160,
            "k_rank_cluster": 6000,
            "k_rank_all_tokens": 500,
            "itopk_size": 224,
        },
    },
    0.95: {
        "lotte": {
            "label": "g2",
            "nprobe": 160,
            "k_rank_cluster": 6000,
            "k_rank_all_tokens": 500,
            "itopk_size": 224,
        },
        "msmarco": {
            "label": "d1",
            "nprobe": 128,
            "k_rank_cluster": 3000,
            "k_rank_all_tokens": 300,
            "itopk_size": 150,
        },
        "hotpot": {
            "label": "k1",
            "nprobe": 384,
            "k_rank_cluster": 28000,
            "k_rank_all_tokens": 1800,
            "itopk_size": 448,
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
    "profile_phase_a_wall_ms": r"Phase A \(Data Preparation\) wall:\s+([0-9.]+)\s+ms",
    "profile_phase_b_wall_ms": r"Phase B wall time:\s+([0-9.]+)\s+ms",
    "profile_phase_c_total_ms": r"Phase C: .*\(total=([0-9.]+)\s+ms,",
    "profile_phase_b_binary_ip_total_ms": r"Phase B binary_ip total\s+:\s+([0-9.]+)\s+ms",
    "profile_phase_b_doc_score_total_ms": r"Phase B doc_score total\s+:\s+([0-9.]+)\s+ms",
    "profile_phase_b_total_kernel_ms": r"Phase B total kernel time\s+:\s+([0-9.]+)\s+ms",
    "profile_transfer_total_ms": r"Total transfer:\s+([0-9.]+)\s+ms",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep v6_lite overlap_chunks=1..8 at fixed Recall@100 ≈ 0.95 "
            "operating points for LoTTE/HotpotQA by default."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Path to gpu_search_v6_lite. Defaults to <build-dir>/gpu_search_v6_lite.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["lotte", "msmarco", "hotpot"],
        default=["lotte", "hotpot"],
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        choices=[0.90, 0.95],
        default=[0.95],
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="overlap_chunks values to sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "profiling" / "gpu_v6_lite_chunk_ablation")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1, help="Queries to evaluate. -1 means all queries.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Also run profile-eval-all-queries logs.")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return parser.parse_args()


def gpu_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.pop("GPU_MVR_V6_USE_DOCPTRS", None)
    return env


def quote_arg(text: str) -> str:
    return subprocess.list2cmdline([text])


def run_logged(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    reuse: bool,
    expected_labels: list[str] | None = None,
) -> None:
    if reuse and log_complete(log_path, expected_labels):
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


def log_complete(path: Path, expected_labels: list[str] | None = None) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    labels = expected_labels or ["single"]
    return all(
        re.search(rf"\[RUN\] Starting config label={re.escape(label)}\b", text, re.MULTILINE) is not None
        and re.search(rf"\[CONFIG\] label={re.escape(label)}\b", text, re.MULTILINE) is not None
        for label in labels
    ) and len(re.findall(r"Throughput:\s+([0-9.]+)\s+qps", text, re.MULTILINE)) >= len(labels) \
        and len(re.findall(r"Average latency per query:\s+([0-9.]+)\s+ms", text, re.MULTILINE)) >= len(labels) \
        and len(re.findall(r"Recall@\d+:\s+([0-9.]+)", text, re.MULTILINE)) >= len(labels)


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"failed to parse {label}")
    return float(match.group(1))


def maybe_parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return None if not match else float(match.group(1))


def log_prefix(target: float, dataset: str, chunk: int) -> str:
    target_text = str(target).replace(".", "")
    return f"v6_lite_{dataset}_r{target_text}_c{chunk}"


def multi_config_log_prefix(target: float, dataset: str) -> str:
    target_text = str(target).replace(".", "")
    return f"v6_lite_{dataset}_r{target_text}_chunks"


def chunk_label(chunk: int) -> str:
    return f"c{chunk}"


def write_runtime_config_csv(path: Path, point: dict[str, int | str], chunks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label",
            "nprobe",
            "k_rank_cluster",
            "k_rank_all_tokens",
            "itopk_size",
            "overlap_chunks",
        ])
        for chunk in chunks:
            writer.writerow([
                chunk_label(chunk),
                point["nprobe"],
                point["k_rank_cluster"],
                point["k_rank_all_tokens"],
                point["itopk_size"],
                chunk,
            ])


def command_for_config(
    args: argparse.Namespace,
    binary: Path,
    dataset: str,
    config_file: Path,
    profile: bool,
) -> list[str]:
    paths = DATASET_PATHS[dataset]
    cmd = [
        str(binary),
        "--query", str(args.repo_root / paths["query"]),
        "--gt", str(args.repo_root / paths["gt"]),
        "--index", str(args.repo_root / paths["index"]),
        "--config-file", str(config_file),
        "--k", str(args.k),
        "--warmup", str(args.warmup),
    ]
    if args.nq >= 0:
        cmd.extend(["--nq", str(args.nq)])
    if profile:
        cmd.append("--profile-eval-all-queries")
    return cmd


def parse_log_text(text: str) -> tuple[float, float, float, dict[str, float | None]]:
    qps = parse_float(r"Throughput:\s+([0-9.]+)\s+qps", text, "qps")
    latency = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, "latency")
    recall = parse_float(r"Recall@\d+:\s+([0-9.]+)", text, "recall")
    metrics = {name: maybe_parse_float(pattern, text) for name, pattern in PROFILE_FIELDS.items()}
    return qps, latency, recall, metrics


def parse_multi_config_log(log_path: Path) -> dict[str, tuple[float, float, float, dict[str, float | None]]]:
    text = read_text(log_path)
    matches = list(re.finditer(r"\[RUN\] Starting config label=([^\s]+)\b", text))
    if not matches:
        return {"single": parse_log_text(text)}

    parsed = {}
    for index, match in enumerate(matches):
        label = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed[label] = parse_log_text(text[match.start():end])
    return parsed


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_qps_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    if plt is None:
        print("[skip] matplotlib is not installed; skipping qps plot rendering", flush=True)
        return
    if not rows:
        return
    titles = {"lotte": "LoTTE", "msmarco": "MSMARCO", "hotpot": "HotpotQA"}
    targets = sorted({float(row["recall_target"]) for row in rows})
    datasets = [d for d in ["lotte", "msmarco", "hotpot"] if any(row["dataset"] == d for row in rows)]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(len(targets), len(datasets), figsize=(3.4 * len(datasets), 2.8 * len(targets)), sharex=True)
    if len(targets) == 1 and len(datasets) == 1:
        axes = [[axes]]
    elif len(targets) == 1:
        axes = [axes]
    elif len(datasets) == 1:
        axes = [[ax] for ax in axes]

    for r, target in enumerate(targets):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            series = sorted(
                [row for row in rows if row["dataset"] == dataset and float(row["recall_target"]) == target],
                key=lambda row: int(row["overlap_chunks"]),
            )
            xs = [int(row["overlap_chunks"]) for row in series]
            ys = [float(row["timed_qps"]) for row in series]
            ax.plot(xs, ys, marker="o", linewidth=1.8, color="#2f6f6d")
            best = max(series, key=lambda row: float(row["timed_qps"]))
            ax.scatter([int(best["overlap_chunks"])], [float(best["timed_qps"])], color="#d65f2e", zorder=3)
            ax.set_title(f"{titles[dataset]}, R@100≈{target:.2f}")
            ax.set_xlabel("overlap_chunks")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.set_ylabel("QPS")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"v6_lite_qps_vs_chunks.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    binary = args.binary or (args.build_dir / "gpu_search_v6_lite")
    if not binary.exists():
        raise FileNotFoundError(f"missing binary: {binary}")
    if any(chunk < 1 for chunk in args.chunks):
        raise ValueError("all chunks must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    env = gpu_env(args.cuda_visible_devices)

    for target in args.recall_targets:
        for dataset in args.datasets:
            point = OPERATING_POINTS[target][dataset]
            prefix = multi_config_log_prefix(target, dataset)
            config_file = args.output_dir / f"{prefix}_config.csv"
            write_runtime_config_csv(config_file, point, args.chunks)
            labels = [chunk_label(chunk) for chunk in args.chunks]

            timed_log = args.output_dir / f"{prefix}_timed.log"
            run_logged(
                command_for_config(args, binary, dataset, config_file, False),
                timed_log,
                env,
                args.reuse_existing,
                labels,
            )
            timed_by_label = parse_multi_config_log(timed_log)

            profile_log = Path()
            profile_by_label: dict[str, tuple[float, float, float, dict[str, float | None]]] = {}
            if args.profile:
                profile_log = args.output_dir / f"{prefix}_profile.log"
                run_logged(
                    command_for_config(args, binary, dataset, config_file, True),
                    profile_log,
                    env,
                    args.reuse_existing,
                    labels,
                )
                profile_by_label = parse_multi_config_log(profile_log)

            for chunk in args.chunks:
                label = chunk_label(chunk)
                qps, latency, recall, timed_metrics = timed_by_label[label]
                profile_metrics = {name: None for name in PROFILE_FIELDS}
                if args.profile and label in profile_by_label:
                    _, _, _, profile_metrics = profile_by_label[label]

                metrics = profile_metrics if args.profile else timed_metrics
                row = {
                    "system": "gpu_search_v6_lite",
                    "dataset": dataset,
                    "recall_target": f"{target:.2f}",
                    "operating_point": point["label"],
                    "k": str(args.k),
                    "nq": str(args.nq),
                    "warmup": str(args.warmup),
                    "nprobe": str(point["nprobe"]),
                    "k_rank_cluster": str(point["k_rank_cluster"]),
                    "k_rank_all_tokens": str(point["k_rank_all_tokens"]),
                    "itopk_size": str(point["itopk_size"]),
                    "overlap_chunks": str(chunk),
                    "timed_qps": str(qps),
                    "timed_avg_latency_ms": str(latency),
                    "timed_recall_at_100": str(recall),
                    "timed_log": str(timed_log),
                    "profile_log": str(profile_log),
                }
                row.update({name: "" if value is None else str(value) for name, value in metrics.items()})
                rows.append(row)
                print(
                    f"[done] dataset={dataset} target={target:.2f} chunk={chunk} "
                    f"qps={qps:.3f} latency_ms={latency:.3f} recall={recall:.5f}",
                    flush=True,
                )

    fieldnames = [
        "system",
        "dataset",
        "recall_target",
        "operating_point",
        "k",
        "nq",
        "warmup",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
        "timed_qps",
        "timed_avg_latency_ms",
        "timed_recall_at_100",
        *PROFILE_FIELDS.keys(),
        "timed_log",
        "profile_log",
    ]
    write_csv(args.output_dir / "v6_lite_chunk_sweep.csv", rows, fieldnames)

    summary_rows = []
    for target in sorted({row["recall_target"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows}):
            group = [row for row in rows if row["dataset"] == dataset and row["recall_target"] == target]
            if not group:
                continue
            best = max(group, key=lambda row: float(row["timed_qps"]))
            chunk1 = next((row for row in group if row["overlap_chunks"] == "1"), None)
            chunk8 = next((row for row in group if row["overlap_chunks"] == "8"), None)
            summary_rows.append({
                "dataset": dataset,
                "recall_target": target,
                "operating_point": best["operating_point"],
                "best_overlap_chunks": best["overlap_chunks"],
                "best_qps": best["timed_qps"],
                "best_latency_ms": best["timed_avg_latency_ms"],
                "best_recall_at_100": best["timed_recall_at_100"],
                "chunk1_qps": "" if chunk1 is None else chunk1["timed_qps"],
                "chunk8_qps": "" if chunk8 is None else chunk8["timed_qps"],
            })
    write_csv(
        args.output_dir / "v6_lite_chunk_best_summary.csv",
        summary_rows,
        [
            "dataset",
            "recall_target",
            "operating_point",
            "best_overlap_chunks",
            "best_qps",
            "best_latency_ms",
            "best_recall_at_100",
            "chunk1_qps",
            "chunk8_qps",
        ],
    )
    render_qps_plot(rows, args.output_dir)
    (args.output_dir / Path(__file__).name).write_text(Path(__file__).read_text())
    print(f"[out] {args.output_dir / 'v6_lite_chunk_sweep.csv'}", flush=True)
    print(f"[out] {args.output_dir / 'v6_lite_chunk_best_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
