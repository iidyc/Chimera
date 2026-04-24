#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_GPU_IMPLS = ["gpu_search_v6", "gpu_search_v8"]
DEFAULT_COLBERT_IMPLS = ["plaid_cpu_hosted", "plaid_gpu_resident"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge PLAID / ColBERT and GPU-MVR benchmark tables, compute a joint "
            "Pareto frontier, and generate per-dataset comparison plots."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling/experiments/plaid_vs_gpu_mvr/main/k100"),
        help="Profiling root containing per-dataset benchmark outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to compare. Repeatable. Default: lotte, hotpot, msmarco.",
    )
    parser.add_argument(
        "--gpu-impl",
        action="append",
        dest="gpu_impls",
        default=[],
        help=f"GPU-MVR implementation label. Repeatable. Default: {', '.join(DEFAULT_GPU_IMPLS)}",
    )
    parser.add_argument(
        "--colbert-impl",
        action="append",
        dest="colbert_impls",
        default=[],
        help=f"ColBERT implementation label. Repeatable. Default: {', '.join(DEFAULT_COLBERT_IMPLS)}",
    )
    parser.add_argument(
        "--output-subdir",
        default="comparison",
        help="Subdirectory name written under each dataset directory.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latest_gpu_results(dataset_dir: Path, implementation: str) -> Path:
    impl_dir = dataset_dir / implementation
    candidates = sorted(impl_dir.glob("benchmark_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no benchmark_results_*.csv found under {impl_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def colbert_results_path(dataset_dir: Path, implementation: str) -> Path:
    impl_dir = dataset_dir / implementation
    candidates = []
    benchmark_results = impl_dir / "benchmark_results.csv"
    if benchmark_results.is_file():
        candidates.append(benchmark_results)
    candidates.extend(sorted(impl_dir.glob("benchmark_results_*.csv")))
    if not candidates:
        raise FileNotFoundError(f"no benchmark_results*.csv found under {impl_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def normalize_gpu_rows(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {
                "implementation": row.get("implementation", path.parent.name),
                "dataset": row["dataset"],
                "label": row["label"],
                "setting": (
                    f"nprobe={row['nprobe']} "
                    f"k_rank_cluster={row['k_rank_cluster']} "
                    f"k_rank_all_tokens={row['k_rank_all_tokens']} "
                    f"itopk_size={row['itopk_size']} "
                    f"overlap_chunks={row['overlap_chunks']}"
                ),
                "queries": row["queries"],
                "k": row["k"],
                "qps": row["qps"],
                "recall": row["recall"],
                "gpu_mem_peak_mib": row.get("gpu_mem_peak_mib", ""),
                "source_csv": str(path),
            }
        )
    return normalized


def normalize_colbert_rows(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        if "implementation" in row:
            implementation = row["implementation"]
            dataset = row["dataset"]
            label = row["label"]
            queries = row["queries"]
            k = row["k"]
            qps = row["qps"]
            recall = row["recall"]
            gpu_mem_peak_mib = row.get("gpu_mem_peak_mib", "")
            setting = f"ncells={row['ncells']} ndocs={row['ndocs']}"
        else:
            implementation = path.parent.name
            dataset = path.parents[1].name
            label = row["label"]
            queries = ""
            k = ""
            qps = row["qps"]
            recall = row["recall"]
            gpu_mem_peak_mib = row.get("gpu_mem_peak_mib", "")
            setting = f"ncells={row['ncells']} ndocs={row['ndocs']}"

        normalized.append(
            {
                "implementation": implementation,
                "dataset": dataset,
                "label": label,
                "setting": setting,
                "queries": queries,
                "k": k,
                "qps": qps,
                "recall": recall,
                "gpu_mem_peak_mib": gpu_mem_peak_mib,
                "source_csv": str(path),
            }
        )
    return normalized


def compute_pareto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    frontier: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        qps = float(row["qps"])
        recall = float(row["recall"])
        dominated = False
        for other_idx, other in enumerate(rows):
            if idx == other_idx:
                continue
            other_qps = float(other["qps"])
            other_recall = float(other["recall"])
            if (
                other_qps >= qps
                and other_recall >= recall
                and (other_qps > qps or other_recall > recall)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (float(row["qps"]), float(row["recall"])))
    return frontier


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "implementation",
        "dataset",
        "label",
        "setting",
        "queries",
        "k",
        "qps",
        "recall",
        "gpu_mem_peak_mib",
        "source_csv",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset(output_dir: Path, dataset: str, rows: list[dict[str, str]]) -> None:
    implementations = []
    for row in rows:
        if row["implementation"] not in implementations:
            implementations.append(row["implementation"])

    any_memory = any(row.get("gpu_mem_peak_mib") for row in rows)
    if any_memory:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ax_perf, ax_mem = axes
    else:
        fig, ax_perf = plt.subplots(1, 1, figsize=(10, 6))
        ax_mem = None

    for implementation in implementations:
        impl_rows = [row for row in rows if row["implementation"] == implementation]
        recalls = [float(row["recall"]) for row in impl_rows]
        qps = [float(row["qps"]) for row in impl_rows]
        ax_perf.scatter(recalls, qps, label=implementation, s=36)
        ax_perf.plot(recalls, qps, linewidth=1, alpha=0.6)

        if ax_mem is not None:
            mem_rows = [row for row in impl_rows if row.get("gpu_mem_peak_mib")]
            if mem_rows:
                mem_recalls = [float(row["recall"]) for row in mem_rows]
                mem_values = [float(row["gpu_mem_peak_mib"]) for row in mem_rows]
                ax_mem.scatter(mem_recalls, mem_values, label=implementation, s=36)
                ax_mem.plot(mem_recalls, mem_values, linewidth=1, alpha=0.6)

    ax_perf.set_title(f"{dataset}: Recall vs QPS")
    ax_perf.set_ylabel("QPS")
    ax_perf.grid(alpha=0.3)
    ax_perf.legend()

    if ax_mem is not None:
        ax_mem.set_title(f"{dataset}: Recall vs Peak GPU Memory")
        ax_mem.set_xlabel("Recall@k")
        ax_mem.set_ylabel("Peak GPU Memory (MiB)")
        ax_mem.grid(alpha=0.3)
    else:
        ax_perf.set_xlabel("Recall@k")

    fig.tight_layout()
    fig.savefig(output_dir / "compare.png", dpi=200)
    fig.savefig(output_dir / "compare.pdf")
    plt.close(fig)


def compare_dataset(
    profiling_root: Path,
    dataset: str,
    gpu_impls: list[str],
    colbert_impls: list[str],
    output_subdir: str,
) -> tuple[Path, Path, Path]:
    dataset_dir = profiling_root / dataset
    rows: list[dict[str, str]] = []

    for implementation in gpu_impls:
        rows.extend(normalize_gpu_rows(latest_gpu_results(dataset_dir, implementation)))
    for implementation in colbert_impls:
        rows.extend(normalize_colbert_rows(colbert_results_path(dataset_dir, implementation)))

    frontier = compute_pareto(rows)
    output_dir = dataset_dir / output_subdir
    combined_csv = output_dir / "combined_points.csv"
    pareto_csv = output_dir / "pareto_frontier.csv"
    write_csv(combined_csv, rows)
    write_csv(pareto_csv, frontier)
    plot_dataset(output_dir, dataset, rows)
    return combined_csv, pareto_csv, output_dir / "compare.png"


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    datasets = args.datasets or ["lotte", "hotpot", "msmarco"]
    gpu_impls = args.gpu_impls or DEFAULT_GPU_IMPLS
    colbert_impls = args.colbert_impls or DEFAULT_COLBERT_IMPLS

    for dataset in datasets:
        combined_csv, pareto_csv, plot_path = compare_dataset(
            profiling_root=profiling_root,
            dataset=dataset,
            gpu_impls=gpu_impls,
            colbert_impls=colbert_impls,
            output_subdir=args.output_subdir,
        )
        print(
            f"dataset={dataset} combined={combined_csv} pareto={pareto_csv} "
            f"plot={plot_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
