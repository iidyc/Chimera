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


# Same v6_lite operating points as the chunk ablation, with a fixed chunk count
# for the LUT/doc-score comparison. Variants change only LUT and Stage-2
# doc-score implementation.
DEFAULT_OVERLAP_CHUNKS = 4


OPERATING_POINTS = {
    0.90: {
        "lotte": {
            "label": "c2",
            "nprobe": 96,
            "k_rank_cluster": 1800,
            "k_rank_all_tokens": 200,
            "itopk_size": 128,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
        "msmarco": {
            "label": "b4",
            "nprobe": 64,
            "k_rank_cluster": 1800,
            "k_rank_all_tokens": 180,
            "itopk_size": 96,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
        "hotpot": {
            "label": "g2",
            "nprobe": 160,
            "k_rank_cluster": 6000,
            "k_rank_all_tokens": 500,
            "itopk_size": 224,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
    },
    0.95: {
        "lotte": {
            "label": "g2",
            "nprobe": 160,
            "k_rank_cluster": 6000,
            "k_rank_all_tokens": 500,
            "itopk_size": 224,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
        "msmarco": {
            "label": "d1",
            "nprobe": 128,
            "k_rank_cluster": 3000,
            "k_rank_all_tokens": 300,
            "itopk_size": 150,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
        "hotpot": {
            "label": "k1",
            "nprobe": 384,
            "k_rank_cluster": 28000,
            "k_rank_all_tokens": 1800,
            "itopk_size": 448,
            "overlap_chunks": DEFAULT_OVERLAP_CHUNKS,
        },
    },
}


VARIANTS = {
    "v6_lite": {
        "binary": "gpu_search_v6_lite",
        "label": "Chimera",
    },
    "v6_nosum": {
        "binary": "gpu_search_v6_nosum",
        "label": "LUT + naive doc score",
    },
    "v6_nolut": {
        "binary": "gpu_search_v6_nolut",
        "label": "No LUT + optimized doc score",
    },
    "v6_nolut_nosum": {
        "binary": "gpu_search_v6_nolut_nosum",
        "label": "No LUT + naive doc score",
    },
}

VARIANT_ORDER = ["v6_lite", "v6_nosum", "v6_nolut", "v6_nolut_nosum"]


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
            "Ablate LUT and Stage-2 doc-score optimization for v6_lite by "
            "comparing v6_lite, v6_nosum, v6_nolut, and v6_nolut_nosum at fixed "
            "Recall@100 ≈ 0.90 and ≈ 0.95 operating points."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--build-dir", type=Path, default=repo_root / "gpu-mvr" / "build")
    parser.add_argument("--binary-v6-lite", type=Path, default=None)
    parser.add_argument("--binary-v6-nosum", type=Path, default=None)
    parser.add_argument("--binary-v6-nolut", type=Path, default=None)
    parser.add_argument("--binary-v6-nolut-nosum", type=Path, default=None)
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
        choices=[0.90, 0.95],
        default=[0.90, 0.95],
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "profiling" / "gpu_v6_lite_lut_docscore_ablation")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nq", type=int, default=-1, help="Queries to evaluate. -1 means all queries.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Also run profile-eval-all-queries logs.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from an existing comparison CSV without running search.",
    )
    parser.add_argument(
        "--plot-input",
        type=Path,
        default=None,
        help="Comparison CSV to use with --plot-only. Default: output-dir/v6_lite_lut_docscore_comparison.csv",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return parser.parse_args()


def binary_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "v6_lite": args.binary_v6_lite or (args.build_dir / "gpu_search_v6_lite"),
        "v6_nosum": args.binary_v6_nosum or (args.build_dir / "gpu_search_v6_nosum"),
        "v6_nolut": args.binary_v6_nolut or (args.build_dir / "gpu_search_v6_nolut"),
        "v6_nolut_nosum": args.binary_v6_nolut_nosum or (args.build_dir / "gpu_search_v6_nolut_nosum"),
    }
    for variant, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing binary for {variant}: {path}")
    return paths


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
    expected_point: dict[str, int | str] | None = None,
) -> None:
    if reuse and log_complete(log_path, expected_point):
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


def log_matches_point(text: str, point: dict[str, int | str]) -> bool:
    expected = {
        "nprobe": point["nprobe"],
        "k_rank_cluster": point["k_rank_cluster"],
        "k_rank_all_tokens": point["k_rank_all_tokens"],
        "itopk_size": point["itopk_size"],
        "overlap_chunks": point["overlap_chunks"],
    }
    return all(
        re.search(rf"\b{name}={value}\b", text, re.MULTILINE) is not None
        for name, value in expected.items()
    )


def log_complete(path: Path, expected_point: dict[str, int | str] | None = None) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    has_metrics = (
        re.search(r"Throughput:\s+([0-9.]+)\s+qps", text, re.MULTILINE) is not None
        and re.search(r"Average latency per query:\s+([0-9.]+)\s+ms", text, re.MULTILINE) is not None
        and re.search(r"Recall@\d+:\s+([0-9.]+)", text, re.MULTILINE) is not None
    )
    if not has_metrics:
        return False
    return expected_point is None or log_matches_point(text, expected_point)


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"failed to parse {label}")
    return float(match.group(1))


def maybe_parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return None if not match else float(match.group(1))


def log_prefix(variant: str, target: float, dataset: str) -> str:
    target_text = str(target).replace(".", "")
    return f"{variant}_{dataset}_r{target_text}"


def command_for(
    args: argparse.Namespace,
    binary: Path,
    dataset: str,
    target: float,
    profile: bool,
) -> list[str]:
    paths = DATASET_PATHS[dataset]
    point = OPERATING_POINTS[target][dataset]
    cmd = [
        str(binary),
        "--query", str(args.repo_root / paths["query"]),
        "--gt", str(args.repo_root / paths["gt"]),
        "--index", str(args.repo_root / paths["index"]),
        "--k", str(args.k),
        "--nprobe", str(point["nprobe"]),
        "--k-rank-cluster", str(point["k_rank_cluster"]),
        "--k-rank-all-tokens", str(point["k_rank_all_tokens"]),
        "--itopk-size", str(point["itopk_size"]),
        "--overlap-chunks", str(point["overlap_chunks"]),
        "--warmup", str(args.warmup),
    ]
    if args.nq >= 0:
        cmd.extend(["--nq", str(args.nq)])
    if profile:
        cmd.append("--profile-eval-all-queries")
    return cmd


def parse_log(log_path: Path) -> tuple[float, float, float, dict[str, float | None]]:
    text = read_text(log_path)
    qps = parse_float(r"Throughput:\s+([0-9.]+)\s+qps", text, "qps")
    latency = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, "latency")
    recall = parse_float(r"Recall@\d+:\s+([0-9.]+)", text, "recall")
    metrics = {name: maybe_parse_float(pattern, text) for name, pattern in PROFILE_FIELDS.items()}
    return qps, latency, recall, metrics


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_qps_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    datasets = [
        dataset
        for dataset in ["lotte", "hotpot", "msmarco"]
        if any(row["dataset"] == dataset for row in rows)
    ]
    targets = [
        target
        for target in [0.90, 0.95]
        if any(abs(float(row["recall_target"]) - target) < 1e-6 for row in rows)
    ]
    if not datasets or not targets:
        return

    titles = {"lotte": "LoTTE", "msmarco": "MSMARCO", "hotpot": "HotpotQA"}
    variants = VARIANT_ORDER
    labels = {
        "v6_lite": "Chimera",
        "v6_nosum": "LUT + naive sum",
        "v6_nolut": "w/o LUT",
        "v6_nolut_nosum": "w/o LUT + naive sum",
    }
    colors = {
        "v6_lite": "#d62728",
        "v6_nosum": "#ff9896",
        "v6_nolut": "#1f77b4",
        "v6_nolut_nosum": "#7f7f7f",
    }
    hatches = {"v6_lite": "", "v6_nosum": "..", "v6_nolut": "//", "v6_nolut_nosum": "\\\\"}

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    )

    fig, axes = plt.subplots(1, len(targets), figsize=(5.4 * len(targets), 4.2), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        target_rows = [row for row in rows if abs(float(row["recall_target"]) - target) < 1e-6]
        x = list(range(len(datasets)))
        width = 0.18
        offsets = [
            (idx - (len(variants) - 1) / 2.0) * width
            for idx in range(len(variants))
        ]
        grouped: dict[str, dict[str, float]] = {}
        for offset, variant in zip(offsets, variants):
            ys = []
            for dataset in datasets:
                match = next(
                    row
                    for row in target_rows
                    if row["dataset"] == dataset and row["variant"] == variant
                )
                qps = float(match["timed_qps"])
                ys.append(qps)
                grouped.setdefault(dataset, {})[variant] = qps
            ax.bar(
                [value + offset for value in x],
                ys,
                width=width,
                color=colors[variant],
                hatch=hatches[variant],
                edgecolor="black",
                linewidth=1.0,
                label=labels[variant],
            )

        ax.set_xticks(x, [titles[dataset] for dataset in datasets])
        ax.set_title(f"Recall@100 = {target:.2f}")
        ax.grid(axis="y", alpha=0.3)
        ymax = 0.0
        for xi, dataset in zip(x, datasets):
            lite_qps = grouped[dataset]["v6_lite"]
            lut_nosum_qps = grouped[dataset]["v6_nosum"]
            nolut_qps = grouped[dataset]["v6_nolut"]
            nosum_qps = grouped[dataset]["v6_nolut_nosum"]
            ymax = max(ymax, lite_qps, lut_nosum_qps, nolut_qps, nosum_qps)
            ax.text(
                xi + offsets[0],
                lite_qps * 1.03,
                f"{lite_qps / nosum_qps:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=16,
            )
            ax.text(
                xi + offsets[-1],
                nosum_qps * 1.03,
                "1x",
                ha="center",
                va="bottom",
                fontsize=16,
            )
        ax.set_ylim(0, ymax * 1.85)
    axes[0].set_ylabel("QPS")
    axes[-1].legend(frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"qps_lut_nolut_v6.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def comparison_fieldnames() -> list[str]:
    return [
        "variant",
        "variant_label",
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


def rows_from_existing_logs(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for target in args.recall_targets:
        for dataset in args.datasets:
            point = OPERATING_POINTS[target][dataset]
            for variant in VARIANT_ORDER:
                timed_log = args.output_dir / f"{log_prefix(variant, target, dataset)}_timed.log"
                if not log_complete(timed_log, point):
                    raise FileNotFoundError(
                        f"missing complete timed log for plot-only mode: {timed_log}"
                    )
                qps, latency, recall, metrics = parse_log(timed_log)
                row = {
                    "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
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
                    "overlap_chunks": str(point["overlap_chunks"]),
                    "timed_qps": str(qps),
                    "timed_avg_latency_ms": str(latency),
                    "timed_recall_at_100": str(recall),
                    "timed_log": str(timed_log),
                    "profile_log": "",
                }
                row.update({name: "" if value is None else str(value) for name, value in metrics.items()})
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        csv_path = args.plot_input or (args.output_dir / "v6_lite_lut_docscore_comparison.csv")
        if csv_path.exists():
            rows = read_csv(csv_path)
        else:
            rows = rows_from_existing_logs(args)
            write_csv(csv_path, rows, comparison_fieldnames())
        render_qps_plot(rows, args.output_dir)
        print(f"[out] {args.output_dir / 'qps_lut_nolut_v6.png'}", flush=True)
        print(f"[out] {args.output_dir / 'qps_lut_nolut_v6.pdf'}", flush=True)
        return

    binaries = binary_paths(args)
    env = gpu_env(args.cuda_visible_devices)

    rows: list[dict[str, str]] = []
    for target in args.recall_targets:
        for dataset in args.datasets:
            point = OPERATING_POINTS[target][dataset]
            for variant in VARIANT_ORDER:
                prefix = log_prefix(variant, target, dataset)
                timed_log = args.output_dir / f"{prefix}_timed.log"
                run_logged(
                    command_for(args, binaries[variant], dataset, target, False),
                    timed_log,
                    env,
                    args.reuse_existing,
                    point,
                )
                qps, latency, recall, timed_metrics = parse_log(timed_log)

                profile_log = Path()
                profile_metrics = {name: None for name in PROFILE_FIELDS}
                if args.profile:
                    profile_log = args.output_dir / f"{prefix}_profile.log"
                    run_logged(
                        command_for(args, binaries[variant], dataset, target, True),
                        profile_log,
                        env,
                        args.reuse_existing,
                        point,
                    )
                    _, _, _, profile_metrics = parse_log(profile_log)

                metrics = profile_metrics if args.profile else timed_metrics
                row = {
                    "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
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
                    "overlap_chunks": str(point["overlap_chunks"]),
                    "timed_qps": str(qps),
                    "timed_avg_latency_ms": str(latency),
                    "timed_recall_at_100": str(recall),
                    "timed_log": str(timed_log),
                    "profile_log": str(profile_log),
                }
                row.update({name: "" if value is None else str(value) for name, value in metrics.items()})
                rows.append(row)
                print(
                    f"[done] dataset={dataset} target={target:.2f} variant={variant} "
                    f"qps={qps:.3f} latency_ms={latency:.3f} recall={recall:.5f}",
                    flush=True,
                )

    write_csv(args.output_dir / "v6_lite_lut_docscore_comparison.csv", rows, comparison_fieldnames())

    summary_rows = []
    for target in sorted({row["recall_target"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows}):
            group = [row for row in rows if row["dataset"] == dataset and row["recall_target"] == target]
            if not group:
                continue
            by_variant = {row["variant"]: row for row in group}
            lite_qps = float(by_variant["v6_lite"]["timed_qps"])
            lut_nosum_qps = float(by_variant["v6_nosum"]["timed_qps"])
            nolut_qps = float(by_variant["v6_nolut"]["timed_qps"])
            nosum_qps = float(by_variant["v6_nolut_nosum"]["timed_qps"])
            summary_rows.append({
                "dataset": dataset,
                "recall_target": target,
                "operating_point": by_variant["v6_lite"]["operating_point"],
                "v6_lite_qps": by_variant["v6_lite"]["timed_qps"],
                "v6_nosum_qps": by_variant["v6_nosum"]["timed_qps"],
                "v6_nolut_qps": by_variant["v6_nolut"]["timed_qps"],
                "v6_nolut_nosum_qps": by_variant["v6_nolut_nosum"]["timed_qps"],
                "lut_gain_optimized_sum_pct": str(100.0 * (lite_qps / nolut_qps - 1.0)),
                "lut_gain_naive_sum_pct": str(100.0 * (lut_nosum_qps / nosum_qps - 1.0)),
                "docscore_opt_gain_lut_pct": str(100.0 * (lite_qps / lut_nosum_qps - 1.0)),
                "docscore_opt_gain_nolut_pct": str(100.0 * (nolut_qps / nosum_qps - 1.0)),
                "combined_gain_pct": str(100.0 * (lite_qps / nosum_qps - 1.0)),
                "v6_lite_recall_at_100": by_variant["v6_lite"]["timed_recall_at_100"],
                "v6_nosum_recall_at_100": by_variant["v6_nosum"]["timed_recall_at_100"],
                "v6_nolut_recall_at_100": by_variant["v6_nolut"]["timed_recall_at_100"],
                "v6_nolut_nosum_recall_at_100": by_variant["v6_nolut_nosum"]["timed_recall_at_100"],
            })
    write_csv(
        args.output_dir / "v6_lite_lut_docscore_summary.csv",
        summary_rows,
        [
            "dataset",
            "recall_target",
            "operating_point",
            "v6_lite_qps",
            "v6_nosum_qps",
            "v6_nolut_qps",
            "v6_nolut_nosum_qps",
            "lut_gain_optimized_sum_pct",
            "lut_gain_naive_sum_pct",
            "docscore_opt_gain_lut_pct",
            "docscore_opt_gain_nolut_pct",
            "combined_gain_pct",
            "v6_lite_recall_at_100",
            "v6_nosum_recall_at_100",
            "v6_nolut_recall_at_100",
            "v6_nolut_nosum_recall_at_100",
        ],
    )
    render_qps_plot(rows, args.output_dir)
    (args.output_dir / Path(__file__).name).write_text(Path(__file__).read_text())
    print(f"[out] {args.output_dir / 'v6_lite_lut_docscore_comparison.csv'}", flush=True)
    print(f"[out] {args.output_dir / 'v6_lite_lut_docscore_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
