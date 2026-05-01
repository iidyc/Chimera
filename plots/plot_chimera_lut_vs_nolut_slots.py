#!/usr/bin/env python3
"""Plot Chimera LUT vs no-LUT throughput using one query slot."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SYSTEM_ORDER = ["chimera", "chimera_nolut"]
SYSTEM_LABELS = {
    "chimera": "Chimera",
    "chimera_nolut": "w/o LUT",
}
DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}
SLOT = 1
TARGET_RECALL = "0.95"
COLORS = {
    "chimera": "#0072B2",
    "chimera_nolut": "#D55E00",
}
SYSTEM_ALIASES = {
    "chimera": "chimera",
    "gpu_search": "chimera",
    "v" "8": "chimera",
    "chimera_nolut": "chimera_nolut",
    "gpu_search_nolut": "chimera_nolut",
    "v" "8_nolut": "chimera_nolut",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_dir = repo_root / "profiling" / "chimera_slot_sweep"
    parser = argparse.ArgumentParser(
        description="Plot Chimera vs no-LUT throughput at Recall@100=0.95."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=17)
    return parser.parse_args()


def read_rows(input_dir: Path) -> list[dict[str, str]]:
    paths = sorted(input_dir.glob("chimera_slot_sweep_summary_*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        if "smoke" in path.name:
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                system = SYSTEM_ALIASES.get(row.get("system", ""))
                if (
                    system in SYSTEM_ORDER
                    and row.get("dataset") in DATASET_ORDER
                    and int(row.get("slots", "-1")) == SLOT
                    and f"{float(row.get('target_recall', 'nan')):.2f}" == TARGET_RECALL
                ):
                    row = dict(row)
                    row["system"] = system
                    rows.append(row)
    return rows


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["system"], int(row["slots"]))].append(row)

    summary = {}
    for key, key_rows in grouped.items():
        qps = [float(row.get("qps_mean") or row["qps"]) for row in key_rows]
        recall = [
            float(row.get("recall_at_100_mean") or row.get("recall_mean") or row["recall"])
            for row in key_rows
        ]
        latency = [
            float(row.get("avg_latency_ms_mean") or row["avg_latency_ms"])
            for row in key_rows
        ]
        summary[key] = {
            "runs": float(len(key_rows)),
            "qps_mean": statistics.mean(qps),
            "qps_std": statistics.stdev(qps) if len(qps) > 1 else 0.0,
            "recall_mean": statistics.mean(recall),
            "avg_latency_ms_mean": statistics.mean(latency),
        }
    return summary


def plot(
    summary: dict[tuple[str, str, int], dict[str, float]],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    font_size: float,
) -> list[Path]:
    missing = [
        (dataset, system, SLOT)
        for dataset in DATASET_ORDER
        for system in SYSTEM_ORDER
        if (dataset, system, SLOT) not in summary
    ]
    if missing:
        raise ValueError(f"missing rows: {missing}")

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "font.size": font_size,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x_positions = list(range(len(DATASET_ORDER)))
    bar_width = 0.34
    offsets = {
        "chimera": -bar_width / 2,
        "chimera_nolut": bar_width / 2,
    }

    for system in SYSTEM_ORDER:
        means = [summary[(dataset, system, SLOT)]["qps_mean"] for dataset in DATASET_ORDER]
        stds = [summary[(dataset, system, SLOT)]["qps_std"] for dataset in DATASET_ORDER]
        ax.bar(
            [x + offsets[system] for x in x_positions],
            means,
            yerr=stds,
            width=bar_width,
            color=COLORS[system],
            edgecolor="#222222",
            linewidth=0.8,
            error_kw={"elinewidth": 1.4, "capsize": 4, "capthick": 1.4},
            label=SYSTEM_LABELS[system],
        )

    ax.set_ylabel("Throughput")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
    ax.set_title("Chimera LUT Impact at Recall@100 = 0.95")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(SYSTEM_ORDER))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.24)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        output = output_dir / f"chimera_lut_vs_nolut_slots.{suffix}"
        fig.savefig(output, dpi=dpi)
        outputs.append(output)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir)
    summary = aggregate(rows)
    outputs = plot(summary, args.output_dir, args.formats, args.dpi, args.font_size)
    print(f"input_dir={args.input_dir}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
