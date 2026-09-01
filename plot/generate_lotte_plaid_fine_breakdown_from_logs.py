#!/usr/bin/env python3

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PHASE_ORDER = [
    ("Probe", "probe_ms", "#4C78A8"),
    ("Transfer", "transfer_ms", "#B279A2"),
    ("Scan", "scan_ms", "#F58518"),
    ("Rescore", "rescore_ms", "#54A24B"),
    ("Ranking", "ranking_ms", "#E45756"),
]

SYSTEM_STYLES = {
    "PLAID": {"edgecolor": "#1F4E79"},
    "PLAID+": {"edgecolor": "#5F3DC4"},
}


@dataclass
class PlaidFineBreakdown:
    system: str
    avg_latency_ms: float
    raw_total_ms: float
    probe_raw_ms: float
    transfer_raw_ms: float
    scan_raw_ms: float
    rescore_raw_ms: float
    ranking_raw_ms: float

    @property
    def scale(self) -> float:
        if self.raw_total_ms <= 0.0:
            return 0.0
        return self.avg_latency_ms / self.raw_total_ms

    def scaled(self, key: str) -> float:
        raw_key = f"{key[:-3]}_raw_ms"
        return getattr(self, raw_key) * self.scale


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a standardized LoTTE PLAID vs PLAID+ stage breakdown from logs."
    )
    parser.add_argument(
        "--plaid-log",
        type=Path,
        default=repo_root / "profiling" / "lotte" / "plaid_vs_gpu_search_recall90_k100" / "plaid_c4m.log",
    )
    parser.add_argument(
        "--plaid-plus-log",
        type=Path,
        default=repo_root / "profiling" / "lotte" / "plaid_gpu_only_vs_gpu_search_recall90_k100" / "plaid_gpu_resident_c4m.log",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "plot" / "lotte_plaid_vs_plaid_plus_detailed_breakdown_from_logs",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text()


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"failed to parse {label} from log")
    return float(match.group(1))


def stage_avg(text: str, name: str, label: str) -> float:
    return parse_float(
        rf"\[PROFILE_STAGE\] name={re.escape(name)} avg_ms=([0-9.]+)",
        text,
        f"{label} {name}",
    )


def transfer_map(text: str) -> dict[str, float]:
    transfers = {}
    for name, value in re.findall(r"\[PROFILE_TRANSFER\] name=([^ ]+) avg_ms=([0-9.]+)", text):
        transfers[name] = float(value)
    return transfers


def parse_plaid_log(path: Path, system: str) -> PlaidFineBreakdown:
    text = read_text(path)
    transfers = transfer_map(text)
    avg_latency_ms = parse_float(r"avg_time=([0-9.]+)s", text, f"{system} latency") * 1000.0
    raw_total_ms = stage_avg(text, "search.total", system)
    probe = stage_avg(text, "candidate.centroid_score", system) + stage_avg(text, "candidate.cell_select", system)
    scan_lookup = stage_avg(text, "approx.pruned.lookup_codes", system)
    rescore_lookup = stage_avg(text, "approx.full.lookup_codes", system)
    ranking_codes_lookup = stage_avg(text, "final.lookup_codes", system)
    lookup_total = scan_lookup + rescore_lookup + ranking_codes_lookup
    codes_transfer = transfers.get("codes", 0.0)
    if lookup_total > 0.0:
        scan_codes_transfer = codes_transfer * (scan_lookup / lookup_total)
        rescore_codes_transfer = codes_transfer * (rescore_lookup / lookup_total)
        ranking_codes_transfer = codes_transfer * (ranking_codes_lookup / lookup_total)
    else:
        scan_codes_transfer = 0.0
        rescore_codes_transfer = 0.0
        ranking_codes_transfer = 0.0

    scan_original = (
        stage_avg(text, "candidate.ivf_lookup", system)
        + stage_avg(text, "candidate.pid_dedup", system)
        + scan_lookup
        + stage_avg(text, "approx.pruned.reduce", system)
        + stage_avg(text, "approx.pruned.topk", system)
    )
    rescore_original = (
        rescore_lookup
        + stage_avg(text, "approx.full.reduce", system)
        + stage_avg(text, "approx.full.topk", system)
    )
    ranking_original = (
        ranking_codes_lookup
        + stage_avg(text, "final.lookup_residuals", system)
        + stage_avg(text, "final.decompress", system)
        + stage_avg(text, "final.matmul", system)
        + stage_avg(text, "final.pack_scores", system)
        + stage_avg(text, "final.reduce", system)
        + stage_avg(text, "rank.sort", system)
    )
    overlapped_scan_transfer = transfers.get("ivf", 0.0) + scan_codes_transfer
    overlapped_rescore_transfer = rescore_codes_transfer
    overlapped_ranking_transfer = transfers.get("residuals", 0.0) + ranking_codes_transfer
    transfer = sum(transfers.values())
    scan = max(scan_original - overlapped_scan_transfer, 0.0)
    rescore = max(rescore_original - overlapped_rescore_transfer, 0.0)
    ranking = max(ranking_original - overlapped_ranking_transfer, 0.0)

    return PlaidFineBreakdown(
        system=system,
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        probe_raw_ms=probe,
        transfer_raw_ms=transfer,
        scan_raw_ms=scan,
        rescore_raw_ms=rescore,
        ranking_raw_ms=ranking,
    )


def add_segment_label(ax: plt.Axes, y: float, left: float, width: float, label: str) -> None:
    if width < 1.15:
        return
    ax.text(
        left + width / 2.0,
        y,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )


def plot(output_root: Path, systems: list[PlaidFineBreakdown]) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    y_positions = list(range(len(systems)))
    bar_height = 0.58

    for y, system in zip(y_positions, systems):
        left = 0.0
        edgecolor = SYSTEM_STYLES[system.system]["edgecolor"]
        for label, key, color in PHASE_ORDER:
            value = system.scaled(key)
            if value <= 0.0:
                continue
            hatch = "xx" if label == "Transfer" else "oo" if label == "Rescore" else None
            ax.barh(
                y,
                value,
                left=left,
                height=bar_height,
                color=color,
                edgecolor=edgecolor,
                linewidth=1.7,
                hatch=hatch,
            )
            add_segment_label(ax, y, left, value, label)
            left += value

    ax.set_xlabel("Average Query Latency Contribution (ms)", fontsize=15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([system.system for system in systems], fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.invert_yaxis()
    ax.set_title("LoTTE: PLAID vs PLAID+ Standardized Breakdown", fontsize=17)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="#000000" if label in {"Transfer", "Rescore"} else "none",
            hatch="xx" if label == "Transfer" else "oo" if label == "Rescore" else None,
            label=label,
        )
        for label, _, color in PHASE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d9d9d9",
        framealpha=0.95,
        fontsize=12,
        title="Bar Colors",
        title_fontsize=13,
        handlelength=1.1,
        labelspacing=0.45,
        columnspacing=1.0,
    )
    fig.tight_layout()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    systems = [
        parse_plaid_log(args.plaid_log, "PLAID"),
        parse_plaid_log(args.plaid_plus_log, "PLAID+"),
    ]
    plot(args.output_root, systems)
    print(f"[done] plot_png={args.output_root.with_suffix('.png')}")
    print(f"[done] plot_pdf={args.output_root.with_suffix('.pdf')}")
    for system in systems:
        print(
            f"[summary] system={system.system} total_ms={system.avg_latency_ms:.6f} "
            f"transfer_ms={system.transfer_raw_ms:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
