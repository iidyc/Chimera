#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_CONFIGS = {
    "lotte": {
        "query": "dataset/lotte/raw/query.bin",
        "gt": "dataset/lotte/raw/gt.tsv",
        "index": "dataset/lotte/gpu_mvr_2m",
        "nprobe": 128,
        "k_rank_cluster": 4000,
        "k_rank_all_tokens": 400,
        "itopk_size": 192,
    },
    "msmarco": {
        "query": "dataset/msmarco/raw/query.bin",
        "gt": "dataset/msmarco/raw/gt.tsv",
        "index": "dataset/msmarco/gpu_mvr_2m",
        "nprobe": 96,
        "k_rank_cluster": 3200,
        "k_rank_all_tokens": 320,
        "itopk_size": 160,
    },
    "hotpot": {
        "query": "dataset/hotpot/raw/query.bin",
        "gt": "dataset/hotpot/raw/gt.tsv",
        "index": "dataset/hotpot/gpu_mvr_2m",
        "nprobe": 448,
        "k_rank_cluster": 28000,
        "k_rank_all_tokens": 2200,
        "itopk_size": 512,
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "profiling" / "gpu_overlap_chunks_recall95_k100"
    parser = argparse.ArgumentParser(
        description=(
            "Sweep overlap_chunks for gpu_search_v6/gpu_search_v8 at fixed "
            "≈0.95-recall k=100 operating points."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["gpu_search_v6", "gpu_search_v8"],
        default=["gpu_search_v6"],
        help="GPU-MVR binaries to benchmark.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["lotte", "msmarco", "hotpot"],
        default=["lotte"],
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="overlap_chunks values to sweep.",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return parser.parse_args()


def quote_arg(text: str) -> str:
    return subprocess.list2cmdline([text])


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], reuse: bool) -> None:
    if reuse and log_path.exists():
        print(f"[reuse] {log_path}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(quote_arg(x) for x in cmd)}")
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


def gpu_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def run_timed(
    args: argparse.Namespace,
    system: str,
    dataset: str,
    chunk: int,
) -> tuple[Path, float, float, float]:
    cfg = DATASET_CONFIGS[dataset]
    binary = args.build_dir / system
    if not binary.exists():
        raise FileNotFoundError(f"missing binary: {binary}")

    log_path = args.output_dir / f"{system}_{dataset}_c{chunk}_timed.log"
    cmd = [
        str(binary),
        "--query", str(args.repo_root / cfg["query"]),
        "--gt", str(args.repo_root / cfg["gt"]),
        "--index", str(args.repo_root / cfg["index"]),
        "--k", str(args.k),
        "--nprobe", str(cfg["nprobe"]),
        "--k-rank-cluster", str(cfg["k_rank_cluster"]),
        "--k-rank-all-tokens", str(cfg["k_rank_all_tokens"]),
        "--itopk-size", str(cfg["itopk_size"]),
        "--overlap-chunks", str(chunk),
        "--nq", "0",
        "--warmup", str(args.warmup),
    ]
    run_logged(cmd, log_path, gpu_env(args.cuda_visible_devices), args.reuse_existing)

    text = read_text(log_path)
    qps = parse_float(r"Throughput:\s+([0-9.]+)\s+qps", text, f"{system} {dataset} qps")
    latency_ms = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, f"{system} {dataset} latency")
    recall = parse_float(r"Recall@\d+:\s+([0-9.]+)", text, f"{system} {dataset} recall")
    return log_path, qps, latency_ms, recall


def run_profile(
    args: argparse.Namespace,
    system: str,
    dataset: str,
    chunk: int,
) -> tuple[Path, dict[str, float | None]]:
    cfg = DATASET_CONFIGS[dataset]
    binary = args.build_dir / system
    log_path = args.output_dir / f"{system}_{dataset}_c{chunk}_profile.log"
    cmd = [
        str(binary),
        "--query", str(args.repo_root / cfg["query"]),
        "--gt", str(args.repo_root / cfg["gt"]),
        "--index", str(args.repo_root / cfg["index"]),
        "--k", str(args.k),
        "--nprobe", str(cfg["nprobe"]),
        "--k-rank-cluster", str(cfg["k_rank_cluster"]),
        "--k-rank-all-tokens", str(cfg["k_rank_all_tokens"]),
        "--itopk-size", str(cfg["itopk_size"]),
        "--overlap-chunks", str(chunk),
        "--nq", "0",
        "--warmup", str(args.warmup),
        "--profile-eval-all-queries",
    ]
    run_logged(cmd, log_path, gpu_env(args.cuda_visible_devices), args.reuse_existing)

    text = read_text(log_path)
    metrics = {
        "profile_total_search_time_ms": maybe_parse_float(r"Total search time\s+:\s+([0-9.]+)\s+ms", text),
        "profile_total_wall_ms": maybe_parse_float(r"Total wall-clock time:\s+([0-9.]+)\s+ms", text),
        "profile_stage1_time_ms": maybe_parse_float(r"Stage 1 time:\s+([0-9.]+)\s+ms", text),
        "profile_cagra_ms": maybe_parse_float(r"CAGRA search\s+:\s+([0-9.]+)\s+ms", text),
        "profile_ivf_expansion_ms": maybe_parse_float(r"GPU IVF expansion\s+:\s+([0-9.]+)\s+ms", text),
        "profile_binary_ip_ms": maybe_parse_float(r"Binary IP kernel\s+:\s+([0-9.]+)\s+ms", text),
        "profile_atomic_agg_ms": maybe_parse_float(r"Aggregation \+ tracking\s+:\s+([0-9.]+)\s+ms", text),
        "profile_sum_scores_ms": maybe_parse_float(r"Sum doc scores \(sparse\)\s+:\s+([0-9.]+)\s+ms", text),
        "profile_topk_sort_ms": maybe_parse_float(r"Top-k sort \(sparse\)\s+:\s+([0-9.]+)\s+ms", text),
        "profile_d2d_ms": maybe_parse_float(r"D2D copy top-k doc IDs\s+:\s+([0-9.]+)\s+ms", text),
        "profile_memset_ms": maybe_parse_float(r"Memset .*:\s+([0-9.]+)\s+ms", text),
        "profile_sum_accounted_ms": maybe_parse_float(r"Sum accounted\s+:\s+([0-9.]+)\s+ms", text),
        "profile_stage23_time_ms": maybe_parse_float(r"Stage 2\+3 time\s+:\s+([0-9.]+)\s+ms", text),
        "profile_phase_a_wall_ms": maybe_parse_float(r"Phase A \(Data Preparation\) wall:\s+([0-9.]+)\s+ms", text),
        "profile_phase_b_wall_ms": maybe_parse_float(r"Phase B wall time:\s+([0-9.]+)\s+ms", text),
        "profile_phase_c_total_ms": maybe_parse_float(r"Phase C: .*\(total=([0-9.]+)\s+ms,", text),
        "profile_phase_c_cpu_ip_ex_ms": maybe_parse_float(r"Phase C: .*cpu_ip_ex=([0-9.]+)\s+ms", text),
        "profile_phase_b_binary_ip_total_ms": maybe_parse_float(r"Phase B binary_ip total\s+:\s+([0-9.]+)\s+ms", text),
        "profile_phase_b_doc_score_total_ms": maybe_parse_float(r"Phase B doc_score total\s+:\s+([0-9.]+)\s+ms", text),
        "profile_phase_b_total_kernel_ms": maybe_parse_float(r"Phase B total kernel time\s+:\s+([0-9.]+)\s+ms", text),
        "profile_transfer_total_ms": maybe_parse_float(r"Total transfer:\s+([0-9.]+)\s+ms", text),
        "profile_h2d_ms": maybe_parse_float(r"H2D:\s+([0-9.]+)\s+ms", text),
        "profile_d2h_ms": maybe_parse_float(r"D2H:\s+([0-9.]+)\s+ms", text),
    }
    return log_path, metrics


def float_or_blank(value: float | None) -> str:
    return "" if value is None else str(value)


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["timed_qps"]))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_plots(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9})

    datasets = [dataset for dataset in ["lotte", "msmarco", "hotpot"] if any(r["dataset"] == dataset for r in rows)]
    systems = [system for system in ["gpu_search_v6", "gpu_search_v8"] if any(r["system"] == system for r in rows)]
    colors = {"gpu_search_v6": "#1f77b4", "gpu_search_v8": "#d62728"}
    labels = {"gpu_search_v6": "v6", "gpu_search_v8": "v8"}
    titles = {"lotte": "LoTTE", "msmarco": "MSMARCO", "hotpot": "HotpotQA"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.8), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        for system in systems:
            series = sorted(
                [row for row in rows if row["dataset"] == dataset and row["system"] == system],
                key=lambda row: int(row["overlap_chunks"]),
            )
            if not series:
                continue
            xs = [int(row["overlap_chunks"]) for row in series]
            ys = [float(row["timed_qps"]) for row in series]
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8, color=colors[system], label=labels[system])
        ax.set_title(titles[dataset])
        ax.set_xlabel("Overlap Chunks")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("QPS")
    if systems:
        axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"qps_vs_overlap_chunks.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.0), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        x_positions = []
        names = []
        bottoms = []
        stage1_vals = []
        phase_a_vals = []
        phase_b_vals = []
        phase_c_vals = []
        stage23_vals = []
        bests = []
        for idx, system in enumerate(systems):
            rr = [row for row in rows if row["dataset"] == dataset and row["system"] == system]
            if not rr:
                continue
            best = best_row(rr)
            bests.append(best)
            x_positions.append(len(x_positions))
            names.append(labels[system])
            stage1_vals.append(float(best["profile_stage1_time_ms"]))
            if system == "gpu_search_v6":
                phase_a_vals.append(float(best["profile_phase_a_wall_ms"] or 0.0))
                phase_b_vals.append(float(best["profile_phase_b_wall_ms"] or 0.0))
                phase_c_vals.append(float(best["profile_phase_c_total_ms"] or 0.0))
                stage23_vals.append(0.0)
            else:
                phase_a_vals.append(0.0)
                phase_b_vals.append(0.0)
                phase_c_vals.append(0.0)
                stage23_vals.append(float(best["profile_stage23_time_ms"] or 0.0))

        if not x_positions:
            continue

        components = [
            ("Stage 1", stage1_vals, "#4e79a7"),
            ("Phase A / Prep", phase_a_vals, "#59a14f"),
            ("Phase B", phase_b_vals, "#f28e2b"),
            ("Phase C / Rerank", phase_c_vals, "#e15759"),
            ("Stage 2+3", stage23_vals, "#b07aa1"),
        ]
        bottoms = [0.0] * len(x_positions)
        for label, values, color in components:
            ax.bar(x_positions, values, bottom=bottoms, color=color, width=0.65, label=label)
            bottoms = [bottoms[i] + values[i] for i in range(len(values))]
        ax.set_xticks(x_positions, names)
        ax.set_title(titles[dataset])
        ax.grid(True, axis="y", alpha=0.25)
        for idx, best in enumerate(bests):
            ax.text(
                x_positions[idx],
                bottoms[idx] + 0.05,
                f"c{best['overlap_chunks']}\n{float(best['timed_qps']):.0f} qps",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[0].set_ylabel("Average ms")
    handles, handle_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, handle_labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"best_chunk_breakdown.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for system in args.systems:
        for dataset in args.datasets:
            cfg = DATASET_CONFIGS[dataset]
            for chunk in args.chunks:
                print(f"[RUN] {system}_{dataset}_c{chunk} timed")
                timed_log, qps, latency_ms, recall = run_timed(args, system, dataset, chunk)
                print(
                    f"[DONE] {system}_{dataset}_c{chunk} timed "
                    f"recall={recall:.6f} qps={qps:.3f} latency_ms={latency_ms:.3f}"
                )

                print(f"[RUN] {system}_{dataset}_c{chunk} profile")
                profile_log, metrics = run_profile(args, system, dataset, chunk)
                print(
                    f"[DONE] {system}_{dataset}_c{chunk} profile "
                    f"stage1={metrics['profile_stage1_time_ms']} "
                    f"stage23={metrics['profile_stage23_time_ms']} "
                    f"phaseA={metrics['profile_phase_a_wall_ms']} "
                    f"phaseC={metrics['profile_phase_c_total_ms']}"
                )

                row = {
                    "system": system,
                    "dataset": dataset,
                    "k": str(args.k),
                    "nprobe": str(cfg["nprobe"]),
                    "k_rank_cluster": str(cfg["k_rank_cluster"]),
                    "k_rank_all_tokens": str(cfg["k_rank_all_tokens"]),
                    "itopk_size": str(cfg["itopk_size"]),
                    "overlap_chunks": str(chunk),
                    "timed_recall_at_100": str(recall),
                    "timed_qps": str(qps),
                    "timed_avg_latency_ms": str(latency_ms),
                    "profile_log": str(profile_log),
                    "timed_log": str(timed_log),
                }
                for key, value in metrics.items():
                    row[key] = float_or_blank(value)
                if system == "gpu_search_v6":
                    phase_a = float(row["profile_phase_a_wall_ms"] or 0.0)
                    phase_b = float(row["profile_phase_b_wall_ms"] or 0.0)
                    phase_c = float(row["profile_phase_c_total_ms"] or 0.0)
                    row["profile_stage23_like_ms"] = str(phase_a + phase_b + phase_c)
                else:
                    row["profile_stage23_like_ms"] = row["profile_stage23_time_ms"]
                rows.append(row)

    fieldnames = [
        "system",
        "dataset",
        "k",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
        "timed_recall_at_100",
        "timed_qps",
        "timed_avg_latency_ms",
        "profile_total_search_time_ms",
        "profile_total_wall_ms",
        "profile_stage1_time_ms",
        "profile_cagra_ms",
        "profile_ivf_expansion_ms",
        "profile_binary_ip_ms",
        "profile_atomic_agg_ms",
        "profile_sum_scores_ms",
        "profile_topk_sort_ms",
        "profile_d2d_ms",
        "profile_memset_ms",
        "profile_sum_accounted_ms",
        "profile_stage23_time_ms",
        "profile_stage23_like_ms",
        "profile_phase_a_wall_ms",
        "profile_phase_b_wall_ms",
        "profile_phase_c_total_ms",
        "profile_phase_c_cpu_ip_ex_ms",
        "profile_phase_b_binary_ip_total_ms",
        "profile_phase_b_doc_score_total_ms",
        "profile_phase_b_total_kernel_ms",
        "profile_transfer_total_ms",
        "profile_h2d_ms",
        "profile_d2h_ms",
        "profile_log",
        "timed_log",
    ]
    write_csv(args.output_dir / "overlap_chunks_sweep.csv", rows, fieldnames)

    summary_rows = []
    for dataset in sorted(set(row["dataset"] for row in rows)):
        for system in sorted(set(row["system"] for row in rows)):
            filtered = [row for row in rows if row["dataset"] == dataset and row["system"] == system]
            if not filtered:
                continue
            filtered.sort(key=lambda row: int(row["overlap_chunks"]))
            best = best_row(filtered)
            first = filtered[0]
            last = filtered[-1]
            summary_rows.append({
                "dataset": dataset,
                "system": system,
                "best_overlap_chunks": best["overlap_chunks"],
                "best_qps": best["timed_qps"],
                "best_latency_ms": best["timed_avg_latency_ms"],
                "best_recall_at_100": best["timed_recall_at_100"],
                "chunk1_qps": first["timed_qps"],
                "chunk8_qps": last["timed_qps"],
                "best_vs_chunk1_pct": str(100.0 * (float(best["timed_qps"]) / float(first["timed_qps"]) - 1.0)),
                "best_vs_chunk8_pct": str(100.0 * (float(best["timed_qps"]) / float(last["timed_qps"]) - 1.0)),
                "stage1_ms_at_best": best["profile_stage1_time_ms"],
                "stage23_like_ms_at_best": best["profile_stage23_like_ms"],
                "transfer_ms_at_best": best["profile_transfer_total_ms"],
                "phase_c_ms_at_best": best["profile_phase_c_total_ms"],
                "phase_b_wall_ms_at_best": best["profile_phase_b_wall_ms"],
            })
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fields:
        write_csv(args.output_dir / "overlap_chunks_best_summary.csv", summary_rows, summary_fields)

    render_plots(rows, args.output_dir)

    print("[DONE] all runs complete")
    print(args.output_dir / "overlap_chunks_sweep.csv")


if __name__ == "__main__":
    main()
