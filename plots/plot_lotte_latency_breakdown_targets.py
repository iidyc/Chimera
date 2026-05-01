#!/usr/bin/env python3
"""Plot Chimera latency breakdown on LoTTE at Recall@100 targets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_paths import latest_required


SEGMENTS = [
    ("Other Overhead", "other_ms", "#B279A2", "xx"),
    ("Candidate Refinement", "candidate_refinement_ms", "#4C78A8", ".."),
    ("Candidate Generation", "candidate_selection_ms", "#F58518", ""),
    ("Document Scoring", "document_scoring_ms", "#54A24B", "oo"),
]
BAR_SEGMENT_ORDER = ["other_ms", "candidate_selection_ms", "candidate_refinement_ms", "document_scoring_ms"]


@dataclass(frozen=True)
class BreakdownRow:
    target: str
    total_ms: float
    candidate_selection_ms: float
    candidate_refinement_ms: float
    document_scoring_ms: float
    other_ms: float

    def segment_ms(self, key: str) -> float:
        return getattr(self, key)

    def segment_pct(self, key: str) -> float:
        return self.segment_ms(key) / self.total_ms * 100.0


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"missing column {key!r}") from exc


def load_lotte_rows(path: Path) -> list[BreakdownRow]:
    rows: list[BreakdownRow] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "lotte" or row.get("target_recall") not in {"0.90", "0.95"}:
                continue

            total_ms = as_float(row, "profile_search_wall_ms")
            other_ms = as_float(row, "transfer_total_ms")
            raw_selection = as_float(row, "s1_cagra_ms")
            raw_refinement = (
                as_float(row, "s1_binary_ip_ms")
                + as_float(row, "s1_sum_scores_ms")
                + as_float(row, "phase_b_binary_ip_total_ms")
            )
            raw_document_scoring = (
                as_float(row, "phase_b_doc_score_total_ms")
                + as_float(row, "phase_c_cpu_refine_ms")
                + as_float(row, "phase_c_wait_d2h_ms")
            )

            raw_compute = raw_selection + raw_refinement + raw_document_scoring
            compute_budget = max(total_ms - other_ms, 0.0)
            scale = compute_budget / raw_compute if raw_compute > 0.0 else 1.0
            rows.append(
                BreakdownRow(
                    target=f"Recall@100 = {float(row['target_recall']):.2f}",
                    total_ms=total_ms,
                    candidate_selection_ms=raw_selection * scale,
                    candidate_refinement_ms=raw_refinement * scale,
                    document_scoring_ms=raw_document_scoring * scale,
                    other_ms=other_ms,
                )
            )
    rows.sort(key=lambda row: row.target)
    if len(rows) != 2:
        raise ValueError(f"expected two LoTTE rows for target recalls 0.90/0.95 in {path}, found {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot LoTTE Chimera latency breakdown at target recalls.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Chimera profiling summary CSV. Defaults to latest profiling/chimera_profile summary.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--output-stem", default="lotte_latency_breakdown_recall090_095")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=24)
    return parser.parse_args()


def plot(rows: list[BreakdownRow], args: argparse.Namespace) -> list[Path]:
    font_size = args.font_size
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": font_size,
            "axes.titlesize": font_size + 8,
            "axes.titleweight": "normal",
            "axes.labelsize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size + 1,
            "legend.fontsize": font_size + 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(17.2, 5.9))
    y_positions = list(range(len(rows)))
    bar_height = 0.58

    for y, row in zip(y_positions, rows):
        left = 0.0
        for key in BAR_SEGMENT_ORDER:
            _, _, color, hatch = next(segment for segment in SEGMENTS if segment[1] == key)
            width = row.segment_ms(key)
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="#000000",
                linewidth=1.2,
                hatch=hatch,
            )
            if key != "other_ms":
                ax.text(
                    left + width / 2.0,
                    y,
                    f"{row.segment_pct(key):.1f}%",
                    ha="center",
                    va="center",
                    fontsize=font_size - 2,
                    fontweight="bold",
                    color="white" if key in {"candidate_refinement_ms", "document_scoring_ms"} else "black",
                )
            left += width
        ax.text(
            row.total_ms + 0.05,
            y,
            f"{row.total_ms:.1f} ms",
            ha="left",
            va="center",
            fontsize=font_size + 1,
            fontweight="bold",
            color="#111111",
        )

    ax.set_title("LoTTE Latency Breakdown", pad=16)
    ax.set_xlabel("Average Query Latency (ms)", labelpad=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row.target for row in rows])
    ax.set_xlim(0.0, max(row.total_ms for row in rows) * 1.18)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=color, edgecolor="#000000", hatch=hatch, label=label) for label, _, color, hatch in SEGMENTS]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.34, -0.48),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.2,
        labelspacing=0.85,
    )

    fig.subplots_adjust(left=0.20, right=0.98, top=0.82, bottom=0.43)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in args.formats:
        path = args.output_dir / f"{args.output_stem}.{suffix}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input or latest_required(
        repo_root,
        "profiling/chimera_profile",
        "chimera_profile_summary_*.csv",
    )
    rows = load_lotte_rows(input_path)
    outputs = plot(rows, args)
    print(f"input={input_path}")
    for row in rows:
        parts = ", ".join(f"{label}={row.segment_ms(key):.3f} ms" for label, key, _, _ in SEGMENTS)
        print(f"{row.target}: total={row.total_ms:.3f} ms; {parts}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
