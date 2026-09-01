#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from statistics import mean


DEFAULT_OUTPUT_DIR = Path("plot")
PROFILE_START_PREFIX = "[PROFILE] Phase A (Data Preparation) wall:"
CONFIG_PREFIX = "[CONFIG] label="
RUN_START_PREFIX = "[RUN] Starting config label="

PHASE_A_PATTERN = re.compile(
    r"wall: (?P<phase_a_ms>[0-9.]+) ms, GPU total: (?P<phase_a_gpu_ms>[0-9.]+) ms"
)
SIMPLE_VALUE_PATTERNS = {
    "phase_b_ms": re.compile(r"^\[PROFILE\] Phase B wall time: (?P<value>[0-9.]+) ms$"),
    "phase_abc_ms": re.compile(
        r"^\[PROFILE\] Total wall time for Phase A \+ B \+ C: (?P<value>[0-9.]+) ms$"
    ),
    "stage1_ms": re.compile(r"^\[PROFILE\] Stage 1 time: (?P<value>[0-9.]+) ms$"),
    "total_search_ms": re.compile(r"^\[PROFILE\] Total search time\s*: (?P<value>[0-9.]+) ms$"),
    "h2d_ms": re.compile(r"^\[PROFILE\]\s+H2D: (?P<value>[0-9.]+) ms,"),
    "d2h_ms": re.compile(r"^\[PROFILE\]\s+D2H: (?P<value>[0-9.]+) ms,"),
    "transfer_ms": re.compile(r"^\[PROFILE\]\s+Total transfer: (?P<value>[0-9.]+) ms,"),
}
STAGE1_COMPONENT_PATTERNS = {
    "stage1_cagra_ms": re.compile(r"^\[PROFILE\]\s+1\. CAGRA search\s*: (?P<value>[0-9.]+) ms$"),
    "stage1_ivf_ms": re.compile(r"^\[PROFILE\]\s+2\. GPU IVF expansion\s*: (?P<value>[0-9.]+) ms$"),
    "stage1_binary_ip_ms": re.compile(
        r"^\[PROFILE\]\s+3\. Binary IP kernel\s*: (?P<value>[0-9.]+) ms$"
    ),
    "stage1_aggregation_ms": re.compile(
        r"^\[PROFILE\]\s+4\. Aggregation \+ tracking\s*: (?P<value>[0-9.]+) ms$"
    ),
    "stage1_sum_scores_ms": re.compile(
        r"^\[PROFILE\]\s+5\. Sum doc scores \(sparse\) : (?P<value>[0-9.]+) ms$"
    ),
    "stage1_topk_sort_ms": re.compile(
        r"^\[PROFILE\]\s+6\. Top-k sort \(sparse\)\s*: (?P<value>[0-9.]+) ms$"
    ),
    "stage1_d2d_copy_ms": re.compile(
        r"^\[PROFILE\]\s+7\. D2D copy top-k doc IDs\s*: (?P<value>[0-9.]+) ms$"
    ),
}
PHASE_C_PATTERN = re.compile(
    r"^\[PROFILE\] Phase C: "
    r"wait_d2h=(?P<phase_c_wait_d2h_ms>[0-9.]+) ms, "
    r"topk=(?P<phase_c_topk_ms>[0-9.]+) ms, "
    r"identify=(?P<phase_c_identify_ms>[0-9.]+) ms, "
    r"gpu_extract=(?P<phase_c_gpu_extract_ms>[0-9.]+) ms, "
    r"cpu_ip_ex=(?P<phase_c_cpu_ip_ex_ms>[0-9.]+) ms, "
    r"wait_extract=(?P<phase_c_wait_extract_ms>[0-9.]+) ms, "
    r"combine=(?P<phase_c_combine_ms>[0-9.]+) ms "
    r"\(total=(?P<phase_c_total_ms>[0-9.]+) ms, (?P<phase_c_docs>[0-9]+) docs\)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a gpu_search raw-log timing breakdown for the config nearest to a target recall."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling root containing benchmark CSVs. Default: profiling.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("log/bench"),
        help="Benchmark log root. Default: log/bench.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example lotte.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="GPU-MVR version, for example v6 or gpu_search_v6.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        required=True,
        help="Target recall used to choose the nearest benchmark row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNG/PDF outputs will be written. Default: plot.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output file stem. Default is derived from dataset/version/target recall.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Render only the total-search summary as a single stacked bar. "
            "Component segments are proportionally fit into total search time."
        ),
    )
    return parser.parse_args()


def normalize_version(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("gpu_search_"):
        stripped = stripped.removeprefix("gpu_search_")
    if not stripped.startswith("v") or not stripped[1:].isdigit():
        raise ValueError(f"invalid version {value!r}; expected vN or gpu_search_vN")
    return stripped


def setup_matplotlib_cache(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    fontconfig_dir = cache_dir / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    os.environ["FONTCONFIG_CACHE"] = str(fontconfig_dir.resolve())


def find_latest_benchmark_csv(profiling_root: Path, dataset: str, version: str) -> Path:
    version_dir = profiling_root / dataset / f"gpu_search_{version}"
    candidates = sorted(version_dir.glob("benchmark_results*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no benchmark CSV found under {version_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def find_latest_log(log_root: Path, dataset: str, version: str) -> Path:
    version_dir = log_root / f"gpu_search_{version}" / dataset
    candidates = sorted(version_dir.glob("benchmark*.log"))
    if not candidates:
        raise FileNotFoundError(f"no benchmark log found under {version_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_row(rows: list[dict[str, str]], target_recall: float) -> dict[str, str]:
    if not rows:
        raise RuntimeError("benchmark CSV has no rows")
    return min(
        rows,
        key=lambda row: (
            abs(float(row["recall"]) - target_recall),
            -float(row["qps"]),
        ),
    )


def extract_config_lines(log_path: Path, label: str) -> list[str]:
    lines = log_path.read_text().splitlines()
    config_token = f"{CONFIG_PREFIX}{label} "
    start_index = None
    for index, line in enumerate(lines):
        if line.startswith(config_token):
            start_index = index
            break
    if start_index is None:
        raise RuntimeError(f"could not find config {label!r} in {log_path}")

    end_index = len(lines)
    next_run_token = RUN_START_PREFIX
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith(next_run_token):
            end_index = index
            break
    return lines[start_index:end_index]


def parse_profile_blocks(lines: list[str]) -> list[dict[str, float]]:
    blocks: list[dict[str, float]] = []
    current: dict[str, float] | None = None

    for line in lines:
        if line.startswith(PROFILE_START_PREFIX):
            if current is not None:
                blocks.append(current)
            match = PHASE_A_PATTERN.search(line)
            if match is None:
                raise RuntimeError(f"failed to parse Phase A line: {line}")
            current = {
                "phase_a_ms": float(match.group("phase_a_ms")),
                "phase_a_gpu_ms": float(match.group("phase_a_gpu_ms")),
            }
            continue

        if current is None:
            continue

        phase_c_match = PHASE_C_PATTERN.match(line)
        if phase_c_match is not None:
            for key, value in phase_c_match.groupdict().items():
                current[key] = float(value)
            continue

        matched = False
        for key, pattern in SIMPLE_VALUE_PATTERNS.items():
            match = pattern.match(line)
            if match is not None:
                current[key] = float(match.group("value"))
                matched = True
                break
        if matched:
            continue

        for key, pattern in STAGE1_COMPONENT_PATTERNS.items():
            match = pattern.match(line)
            if match is not None:
                current[key] = float(match.group("value"))
                break

    if current is not None:
        blocks.append(current)

    required_keys = {
        "phase_a_ms",
        "phase_b_ms",
        "phase_c_total_ms",
        "stage1_ms",
        "total_search_ms",
        "h2d_ms",
        "d2h_ms",
        "transfer_ms",
    }
    filtered_blocks: list[dict[str, float]] = []
    for block in blocks:
        if required_keys.issubset(block):
            filtered_blocks.append(block)

    if not filtered_blocks:
        raise RuntimeError("no complete profile blocks found")
    return filtered_blocks


def average_metrics(blocks: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for block in blocks for key in block})
    averaged: dict[str, float] = {}
    for key in keys:
        values = [block[key] for block in blocks if key in block]
        averaged[key] = mean(values)
    return averaged


def format_recall_for_name(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def render_plot(
    metrics: dict[str, float],
    dataset: str,
    version: str,
    target_recall: float,
    row: dict[str, str],
    num_profile_blocks: int,
    output_prefix: Path,
    summary_only: bool,
) -> tuple[Path, Path]:
    setup_matplotlib_cache(output_prefix.parent)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    dataset_title = {
        "lotte": "LoTTE",
        "hotpot": "HotpotQA",
        "msmarco": "MS MARCO",
    }.get(dataset, dataset.upper())

    summary_components = [
        ("Stage 1", metrics["stage1_ms"], "#2A9D8F"),
        ("Phase A", metrics["phase_a_ms"], "#457B9D"),
        ("Phase B", metrics["phase_b_ms"], "#8D5A97"),
        ("Phase C", metrics["phase_c_total_ms"], "#E76F51"),
    ]

    if summary_only:
        total_search_ms = metrics["total_search_ms"]
        raw_component_sum_ms = sum(value for _, value, _ in summary_components)
        scale = total_search_ms / raw_component_sum_ms if raw_component_sum_ms > 0.0 else 1.0

        fig, ax = plt.subplots(figsize=(11, 3.6))
        left = 0.0
        for label, value, color in summary_components:
            width = value * scale
            ax.barh(
                [0],
                [width],
                left=[left],
                color=color,
                height=0.56,
                edgecolor="white",
                linewidth=1.2,
                label=f"{label} ({value:.3f} ms)",
            )
            if width >= total_search_ms * 0.14:
                ax.text(
                    left + width / 2.0,
                    0,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
            left += width

        ax.barh([0], [total_search_ms], color="none", edgecolor="#1F2933", linewidth=1.5, height=0.56)
        ax.text(
            total_search_ms + total_search_ms * 0.02,
            0,
            f"{total_search_ms:.3f} ms",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1F2933",
        )
        ax.set_yticks([0])
        ax.set_yticklabels(["Total search"])
        ax.set_xlim(0.0, total_search_ms * 1.20)
        ax.set_xlabel("Time (ms)")
        ax.set_title("Total search breakdown")
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.30))

        fig.suptitle(
            f"{dataset_title} gpu_search_{version} breakdown near Recall@100={target_recall:.1f}",
            fontsize=13,
            y=0.98,
        )
        fig.text(
            0.5,
            0.88,
            (
                f"selected {row['label']} | recall={float(row['recall']):.6f} | "
                f"qps={float(row['qps']):.3f} | avg latency={float(row['avg_latency_ms']):.3f} ms | "
                f"avg over {num_profile_blocks} profiled warmup queries"
            ),
            ha="center",
            va="top",
            fontsize=9.5,
        )
        fig.text(
            0.5,
            0.07,
            (
                f"Segments are proportionally fit into total search because the raw component sum "
                f"({raw_component_sum_ms:.3f} ms) is slightly above total search ({total_search_ms:.3f} ms). "
                f"Transfer total={metrics['transfer_ms']:.3f} ms overlaps with these phases and is not stacked."
            ),
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )
        fig.subplots_adjust(top=0.68, bottom=0.28, left=0.10, right=0.96)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

        summary_labels = [
            "Total search",
            "Stage 1",
            "Phase C",
            "Phase A",
            "Phase B",
            "Transfer total",
        ]
        summary_values = [
            metrics["total_search_ms"],
            metrics["stage1_ms"],
            metrics["phase_c_total_ms"],
            metrics["phase_a_ms"],
            metrics["phase_b_ms"],
            metrics["transfer_ms"],
        ]
        summary_colors = ["#264653", "#2A9D8F", "#E76F51", "#457B9D", "#8D5A97", "#BC4749"]

        stage1_labels = [
            "CAGRA search",
            "Binary IP kernel",
            "Aggregation + tracking",
            "Top-k sort",
            "Sum doc scores",
            "D2D copy top-k IDs",
            "GPU IVF expansion",
        ]
        stage1_values = [
            metrics["stage1_cagra_ms"],
            metrics["stage1_binary_ip_ms"],
            metrics["stage1_aggregation_ms"],
            metrics["stage1_topk_sort_ms"],
            metrics["stage1_sum_scores_ms"],
            metrics["stage1_d2d_copy_ms"],
            metrics["stage1_ivf_ms"],
        ]
        stage1_colors = [
            "#1D3557",
            "#457B9D",
            "#4D908E",
            "#90BE6D",
            "#F9C74F",
            "#F8961E",
            "#F3722C",
        ]

        phase_c_labels = [
            "CPU IP extract",
            "Combine",
            "Wait D2H",
            "Top-k",
            "Identify",
            "Wait extract",
            "GPU extract",
        ]
        phase_c_values = [
            metrics["phase_c_cpu_ip_ex_ms"],
            metrics["phase_c_combine_ms"],
            metrics["phase_c_wait_d2h_ms"],
            metrics["phase_c_topk_ms"],
            metrics["phase_c_identify_ms"],
            metrics["phase_c_wait_extract_ms"],
            metrics["phase_c_gpu_extract_ms"],
        ]
        phase_c_colors = [
            "#6A4C93",
            "#8AC926",
            "#1982C4",
            "#FFCA3A",
            "#FF924C",
            "#C44536",
            "#6D6875",
        ]

        plot_specs = [
            ("Measured times", summary_labels, summary_values, summary_colors),
            ("Stage 1 detail", stage1_labels, stage1_values, stage1_colors),
            ("Phase C detail", phase_c_labels, phase_c_values, phase_c_colors),
        ]

        for ax, (title, labels, values, colors) in zip(axes, plot_specs):
            positions = list(range(len(labels)))
            ax.barh(positions, values, color=colors, height=0.68)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel("Time (ms)")
            ax.grid(axis="x", alpha=0.2, linewidth=0.5)
            for y, value in zip(positions, values):
                ax.text(value + max(values) * 0.02, y, f"{value:.3f}", va="center", ha="left", fontsize=9)

        fig.suptitle(
            f"{dataset_title} gpu_search_{version} breakdown near Recall@100={target_recall:.1f}",
            fontsize=13,
            y=0.98,
        )
        fig.text(
            0.5,
            0.92,
            (
                f"selected {row['label']} | recall={float(row['recall']):.6f} | "
                f"qps={float(row['qps']):.3f} | avg latency={float(row['avg_latency_ms']):.3f} ms | "
                f"avg over {num_profile_blocks} profiled warmup queries"
            ),
            ha="center",
            va="top",
            fontsize=9.5,
        )
        fig.text(
            0.5,
            0.04,
            "Bars are shown independently because raw profile components overlap and are not safely additive.",
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )
        fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.75)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    version = normalize_version(args.version)
    benchmark_csv = find_latest_benchmark_csv(args.profiling_root.resolve(), args.dataset, version)
    log_path = find_latest_log(args.log_root.resolve(), args.dataset, version)
    row = select_row(read_rows(benchmark_csv), args.target_recall)
    blocks = parse_profile_blocks(extract_config_lines(log_path, row["label"]))
    metrics = average_metrics(blocks)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or (
        f"gpu_search_{version}_{args.dataset}_recall_{format_recall_for_name(args.target_recall)}_breakdown"
    )
    output_prefix = output_dir / output_name
    png_path, pdf_path = render_plot(
        metrics=metrics,
        dataset=args.dataset,
        version=version,
        target_recall=args.target_recall,
        row=row,
        num_profile_blocks=len(blocks),
        output_prefix=output_prefix,
        summary_only=args.summary_only,
    )

    print(
        f"dataset={args.dataset} version={version} target_recall={args.target_recall:.3f} "
        f"selected_label={row['label']} selected_recall={float(row['recall']):.6f} "
        f"qps={float(row['qps']):.3f} blocks={len(blocks)} "
        f"benchmark_csv={benchmark_csv.resolve()} log={log_path.resolve()} "
        f"png={png_path.resolve()} pdf={pdf_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
