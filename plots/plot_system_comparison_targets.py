#!/usr/bin/env python3
"""Plot throughput for PLAID, PLAID+, IGP, and Chimera at target Recall@100."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plot_paths import latest_required


DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}
TARGET_ORDER = ["0.90", "0.95"]
TARGET_LABELS = {
    "0.90": "Recall@100=0.90",
    "0.95": "Recall@100=0.95",
}
SYSTEM_ORDER = ["plaid", "plaid_plus", "igp", "chimera"]
SYSTEM_LABELS = {
    "plaid": "PLAID",
    "plaid_plus": "PLAID+",
    "igp": "IGP",
    "chimera": "Chimera",
}
COLORS = {
    "plaid": "#4C78A8",
    "plaid_plus": "#B279A2",
    "igp": "#F58518",
    "chimera": "#54A24B",
}
HATCHES = {
    "plaid": "..",
    "plaid_plus": "xx",
    "igp": "\\\\",
    "chimera": "oo",
}
LEGACY_CHIMERA_SYSTEM = "v" "8"
CHIMERA_SYSTEM_ALIASES = {"chimera", "gpu_search", LEGACY_CHIMERA_SYSTEM}


@dataclass(frozen=True)
class Point:
    system: str
    dataset: str
    target: str
    qps: float
    recall: float
    label: str
    source: str
    meets_target: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot system throughput at target Recall@100.")
    parser.add_argument("--plaid", type=Path, default=repo_root / "profiling" / "plaid" / "summary.csv")
    parser.add_argument("--plaid-plus", type=Path, default=repo_root / "profiling" / "plaid_plus" / "summary.csv")
    parser.add_argument("--igp", type=Path, default=repo_root / "profiling" / "igp" / "summary_t32.csv")
    parser.add_argument(
        "--chimera",
        type=Path,
        default=None,
        help="Defaults to latest profiling/chimera_chunk_targets/chimera_chunk_targets_summary_*.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=19)
    parser.add_argument(
        "--strict-targets",
        action="store_true",
        help="Do not plot fallback rows that fall below the target recall.",
    )
    return parser.parse_args()


def target_str(value: str | float) -> str:
    return f"{float(value):.2f}"


def read_plaid_like(path: Path, system: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("k") == "100"]


def select_fastest_at_target(
    rows: list[dict[str, str]],
    *,
    system: str,
    dataset: str,
    target: str,
    qps_col: str,
    recall_col: str,
    label_col: str,
    source_col: str,
    strict: bool,
) -> Point | None:
    target_value = float(target)
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    if not dataset_rows:
        return None

    meeting = [row for row in dataset_rows if float(row[recall_col]) >= target_value]
    if meeting:
        chosen = max(meeting, key=lambda row: float(row[qps_col]))
        meets_target = True
    elif strict:
        return None
    else:
        chosen = max(dataset_rows, key=lambda row: (float(row[recall_col]), float(row[qps_col])))
        meets_target = False

    return Point(
        system=system,
        dataset=dataset,
        target=target,
        qps=float(chosen[qps_col]),
        recall=float(chosen[recall_col]),
        label=chosen.get(label_col, ""),
        source=chosen.get(source_col, ""),
        meets_target=meets_target,
    )


def collect_plaid_points(path: Path, system: str, strict: bool) -> dict[tuple[str, str, str], Point]:
    rows = read_plaid_like(path, system)
    points: dict[tuple[str, str, str], Point] = {}
    for dataset in DATASET_ORDER:
        for target in TARGET_ORDER:
            point = select_fastest_at_target(
                rows,
                system=system,
                dataset=dataset,
                target=target,
                qps_col="qps",
                recall_col="recall",
                label_col="label",
                source_col="source",
                strict=strict,
            )
            if point is not None:
                points[(dataset, target, system)] = point
    return points


def collect_igp_points(path: Path, strict: bool) -> dict[tuple[str, str, str], Point]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("phase") == "val" and row.get("topk") == "100" and row.get("metric_name") == "recall@100"
        ]
    points: dict[tuple[str, str, str], Point] = {}
    for dataset in DATASET_ORDER:
        for target in TARGET_ORDER:
            point = select_fastest_at_target(
                rows,
                system="igp",
                dataset=dataset,
                target=target,
                qps_col="throughput_qps",
                recall_col="recall_mean",
                label_col="chunk_json",
                source_col="chunk_json",
                strict=strict,
            )
            if point is not None:
                points[(dataset, target, "igp")] = point
    return points


def collect_chimera_points(path: Path, strict: bool) -> dict[tuple[str, str, str], Point]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("system") in CHIMERA_SYSTEM_ALIASES
            and row.get("slots") == "1"
            and 1 <= int(row["chunks"]) <= 9
        ]
    normalized = []
    for row in rows:
        row = dict(row)
        row["target"] = target_str(row["target_recall"])
        row["label"] = f"chunks={int(row['chunks'])}"
        row["source"] = str(path)
        normalized.append(row)

    points: dict[tuple[str, str, str], Point] = {}
    for dataset in DATASET_ORDER:
        for target in TARGET_ORDER:
            target_rows = [row for row in normalized if row["target"] == target]
            point = select_fastest_at_target(
                target_rows,
                system="chimera",
                dataset=dataset,
                target=target,
                qps_col="qps_mean",
                recall_col="recall_at_100_mean",
                label_col="label",
                source_col="source",
                strict=strict,
            )
            if point is not None:
                points[(dataset, target, "chimera")] = point
    return points


def set_rcparams(font_size: float) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.titleweight": "normal",
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size + 1,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(points: dict[tuple[str, str, str], Point], output_dir: Path, formats: list[str], dpi: int, font_size: float) -> list[Path]:
    set_rcparams(font_size)
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 13.2), sharey=False)
    x_positions = [idx * 1.30 for idx in range(len(SYSTEM_ORDER))]
    width = 0.82

    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, target in enumerate(TARGET_ORDER):
            ax = axes[row_idx][col_idx]
            values = []
            bar_colors = []
            bar_hatches = []
            missing_positions = []
            for system, xpos in zip(SYSTEM_ORDER, x_positions):
                point = points.get((dataset, target, system))
                if point is None or math.isnan(point.qps):
                    values.append(0.0)
                    bar_colors.append("#FFFFFF")
                    bar_hatches.append("")
                    missing_positions.append(xpos)
                else:
                    values.append(point.qps)
                    bar_colors.append(COLORS[system])
                    bar_hatches.append(HATCHES[system])

            bars = ax.bar(
                x_positions,
                values,
                width=width,
                color=bar_colors,
                edgecolor="#222222",
                linewidth=0.9,
            )
            for bar, system, hatch in zip(bars, SYSTEM_ORDER, bar_hatches):
                bar.set_hatch(hatch)
                point = points.get((dataset, target, system))
                if point is not None and not point.meets_target:
                    bar.set_alpha(0.55)

            ymax = max(values) if max(values) > 0 else 1.0
            for bar, system in zip(bars, SYSTEM_ORDER):
                point = points.get((dataset, target, system))
                if point is None:
                    continue
                label = f"{point.qps:.0f}"
                if not point.meets_target:
                    label += "*"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    point.qps,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=font_size - 3,
                    color="#222222",
                    clip_on=False,
                )
            for xpos in missing_positions:
                ax.text(
                    xpos,
                    ymax * 0.05,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=font_size - 3,
                    color="#666666",
                )

            ax.set_title(f"{DATASET_LABELS[dataset]} {TARGET_LABELS[target]}", pad=10, fontweight="normal")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([SYSTEM_LABELS[system] for system in SYSTEM_ORDER], rotation=0, ha="center")
            ax.set_ylim(0, ymax * 1.23)
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel("Throughput")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[system], edgecolor="#222222", hatch=HATCHES[system])
        for system in SYSTEM_ORDER
    ]
    fig.legend(
        handles,
        [SYSTEM_LABELS[system] for system in SYSTEM_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=len(SYSTEM_ORDER),
        columnspacing=1.8,
        handlelength=1.25,
    )
    fig.text(
        0.5,
        0.075,
        "* best available point falls below the target Recall@100",
        ha="center",
        va="center",
        fontsize=font_size - 3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.955, bottom=0.16, wspace=0.18, hspace=0.68)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        path = output_dir / f"system_comparison_qps_r090_r095_grid.{suffix}"
        fig.savefig(path, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_selected_csv(points: dict[tuple[str, str, str], Point], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "system_comparison_qps_r090_r095_selected.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "target_recall", "system", "qps", "recall_at_100", "meets_target", "label", "source"],
        )
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for target in TARGET_ORDER:
                for system in SYSTEM_ORDER:
                    point = points.get((dataset, target, system))
                    if point is None:
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "target_recall": target,
                                "system": system,
                                "qps": "",
                                "recall_at_100": "",
                                "meets_target": "",
                                "label": "",
                                "source": "",
                            }
                        )
                    else:
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "target_recall": target,
                                "system": system,
                                "qps": f"{point.qps:.6f}",
                                "recall_at_100": f"{point.recall:.6f}",
                                "meets_target": point.meets_target,
                                "label": point.label,
                                "source": point.source,
                            }
                        )
    return path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.chimera is None:
        args.chimera = latest_required(
            repo_root,
            "profiling/chimera_chunk_targets",
            "chimera_chunk_targets_summary_*.csv",
        )
    points: dict[tuple[str, str, str], Point] = {}
    points.update(collect_plaid_points(args.plaid, "plaid", args.strict_targets))
    points.update(collect_plaid_points(args.plaid_plus, "plaid_plus", args.strict_targets))
    points.update(collect_igp_points(args.igp, args.strict_targets))
    points.update(collect_chimera_points(args.chimera, args.strict_targets))

    selected_csv = write_selected_csv(points, args.output_dir)
    outputs = plot(points, args.output_dir, args.formats, args.dpi, args.font_size)

    print(f"plaid={args.plaid}")
    print(f"plaid_plus={args.plaid_plus}")
    print(f"igp={args.igp}")
    print(f"chimera={args.chimera}")
    print(selected_csv)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
