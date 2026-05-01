#!/usr/bin/env python3
"""Plot LoTTE latency breakdown for IGP, PLAID, and Chimera."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from plot_paths import latest_required_log


SEGMENT_ORDER = [
    ("CPU Compute", "cpu_compute_ms", "#4C78A8", ".."),
    ("Transfer", "transfer_ms", "#B279A2", "xx"),
    ("GPU Compute", "gpu_compute_ms", "#F58518", ""),
    ("GPU-CPU Compute", "gpu_cpu_compute_ms", "#54A24B", "oo"),
]

SYSTEM_STYLES = {
    "IGP": {"edgecolor": "#000000"},
    "PLAID": {"edgecolor": "#000000"},
    "Chimera": {"edgecolor": "#000000"},
}
SYSTEM_LABELS = {
    "IGP": "IGP (CPU)",
    "PLAID": "PLAID (GPU)",
    "Chimera": "Chimera\n(GPU-CPU)",
}
CUT_MARKER_STEPS = 80
CUT_MARKER_AMPLITUDE = 0.12
CUT_MARKER_OFFSETS = (-0.11, 0.11)
CUT_MARKER_WHITE_LW = 7.0
CUT_MARKER_BLACK_LW = 1.6

# PLAID profile split from the LoTTE c4m profile used by the older breakdown
# script. The current local profiling summary keeps PLAID's end-to-end latency,
# while this template provides the relative phase split. The transfer component
# is overridden to 6.6 ms by default below.
PLAID_PHASE_TEMPLATE_MS = {
    "probe_ms": 0.4226971379815504,
    "transfer_ms": 6.558789929223669,
    "scan_ms": 0.7338086851748777,
    "rescore_ms": 9.74813193409969,
    "ranking_ms": 0.0,
}


@dataclass(frozen=True)
class SystemBreakdown:
    system: str
    avg_latency_ms: float
    raw_total_ms: float
    cpu_compute_ms: float
    transfer_ms: float
    gpu_compute_ms: float
    gpu_cpu_compute_ms: float
    recall: float | None = None
    qps: float | None = None
    config: str = ""

    def phase(self, key: str) -> float:
        return getattr(self, key)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a LoTTE IGP vs PLAID vs Chimera latency-breakdown plot."
    )
    parser.add_argument(
        "--plaid-best-config",
        type=Path,
        default=repo_root / "profiling" / "plaid" / "best_config.csv",
    )
    parser.add_argument(
        "--igp-json",
        type=Path,
        default=repo_root / "profiling" / "igp" / "lotte_k100_r90_t1.json",
    )
    parser.add_argument(
        "--chimera-log",
        type=Path,
        default=None,
        help="Defaults to latest profiling/chimera_profile/logs/*/lotte_k100_r090_095_profile.log.",
    )
    parser.add_argument("--dataset", default="lotte")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--chimera-label", default="r090_c1")
    parser.add_argument("--plaid-transfer-ms", type=float, default=6.6)
    parser.add_argument("--plaid-latency-ms", type=float, default=19.4)
    parser.add_argument("--chimera-transfer-ms", type=float, default=0.04)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--output-stem", default="lotte_time_breakdown_igp_plaid_chimera")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=22)
    parser.add_argument("--max-x-ms", type=float, default=27.0)
    parser.add_argument("--break-x-ms", type=float, default=22.5)
    parser.add_argument("--title", default="LoTTE Recall@100 >= 0.90")
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"missing column {key!r}") from exc


def load_plaid(
    path: Path,
    dataset: str,
    topk: int,
    target_recall: float,
    transfer_ms: float,
    latency_ms: float | None,
) -> SystemBreakdown:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset
            and int(row.get("k", "-1")) == topk
            and abs(float(row.get("target_recall", "nan")) - target_recall) < 1e-6
        ]
    if not rows:
        raise ValueError(f"no PLAID row in {path} for dataset={dataset}, topk={topk}, target={target_recall}")
    row = rows[0]
    avg_latency_ms = latency_ms if latency_ms is not None else as_float(row, "avg_latency_ms")

    phases = {
        "cpu_compute_ms": 0.0,
        "transfer_ms": transfer_ms,
        "gpu_compute_ms": max(avg_latency_ms - transfer_ms, 0.0),
        "gpu_cpu_compute_ms": 0.0,
    }
    return SystemBreakdown(
        system="PLAID",
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=sum(phases.values()),
        recall=as_float(row, "recall"),
        qps=as_float(row, "qps"),
        config=row.get("label", ""),
        **phases,
    )


def load_igp(path: Path) -> SystemBreakdown:
    payload = json.loads(path.read_text())
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"no IGP results found in {path}")
    result = results[0]
    phases = {
        "cpu_compute_ms": float(result["avg_latency_ms"]),
        "transfer_ms": 0.0,
        "gpu_compute_ms": 0.0,
        "gpu_cpu_compute_ms": 0.0,
    }
    return SystemBreakdown(
        system="IGP",
        avg_latency_ms=float(result["avg_latency_ms"]),
        raw_total_ms=sum(phases.values()),
        recall=float(result["recall_mean"]),
        qps=float(result["qps"]),
        config=f"phi_pb={result.get('phi_pb', '')}, phi_ref={result.get('phi_ref', '')}",
        **phases,
    )


def get_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"could not find {label}")
    return float(match.group(1))


def get_chimera_block(text: str, label: str) -> str:
    pattern = re.compile(
        rf"(\[CONFIG\]\s+label={re.escape(label)}\b.*?)(?=\n\[CONFIG\]\s+label=|\n\[GPU_MEM\]|\Z)",
        flags=re.S,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"could not find Chimera block label={label!r}")
    return match.group(1)


def load_chimera(path: Path, label: str, transfer_ms: float) -> SystemBreakdown:
    block = get_chimera_block(path.read_text(), label)
    avg_latency_ms = get_float(r"\[SEARCH\]\s+Average latency per query:\s+([0-9.]+)\s+ms", block, "latency")
    recall = get_float(r"Recall@100:\s+([0-9.]+)", block, "Recall@100")
    qps = get_float(r"\[SEARCH\]\s+Throughput:\s+([0-9.]+)\s+qps", block, "throughput")
    probe = get_float(r"\[PROFILE_AVG\]\s+Stage 1 time\s+:\s+([0-9.]+)\s+ms", block, "Stage 1")
    transfer = get_float(
        r"\[PROFILE_AVG\]\s+Data transfer summary\s+:\s+.*total=([0-9.]+)\s+ms",
        block,
        "transfer",
    )
    scan = get_float(r"\[PROFILE_AVG\]\s+Phase B total kernel time\s+:\s+([0-9.]+)\s+ms", block, "Phase B")
    phase_c = re.search(
        r"\[PROFILE_AVG\]\s+Phase C:\s+.*?cpu_refine=([0-9.]+)\s+ms.*?total=([0-9.]+)\s+ms",
        block,
    )
    if not phase_c:
        raise ValueError("could not find Phase C timing")
    rescore = float(phase_c.group(1))
    ranking = max(float(phase_c.group(2)) - rescore, 0.0)
    raw_total_ms = probe + transfer + scan + rescore + ranking
    scale = avg_latency_ms / raw_total_ms if raw_total_ms > 0.0 else 1.0
    gpu_compute_ms = (probe + scan) * scale
    phases = {
        "cpu_compute_ms": 0.0,
        "transfer_ms": transfer_ms,
        "gpu_compute_ms": gpu_compute_ms,
        "gpu_cpu_compute_ms": max(avg_latency_ms - gpu_compute_ms - transfer_ms, 0.0),
    }
    return SystemBreakdown(
        system="Chimera",
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        recall=recall,
        qps=qps,
        config=label,
        **phases,
    )


def add_latency_label(ax: plt.Axes, y: float, total_ms: float, max_x_ms: float, font_size: float) -> None:
    is_clipped = total_ms > max_x_ms
    x = max_x_ms - 0.55 if is_clipped else total_ms + 0.35
    label = f"{total_ms:.0f} ms" if is_clipped else f"{total_ms:.1f} ms"
    ax.text(
        x,
        y,
        label,
        ha="right" if is_clipped else "left",
        va="center",
        fontsize=font_size,
        fontweight="bold",
        color="white" if is_clipped else "#111111",
        clip_on=False,
    )


def add_cut_marker(ax: plt.Axes, x: float, y: float, bar_height: float) -> None:
    ys = [
        y - bar_height * 0.62 + (bar_height * 1.24) * idx / (CUT_MARKER_STEPS - 1)
        for idx in range(CUT_MARKER_STEPS)
    ]
    xs = [
        x + CUT_MARKER_AMPLITUDE * math.sin((idx / (CUT_MARKER_STEPS - 1)) * 2.0 * math.pi)
        for idx in range(CUT_MARKER_STEPS)
    ]
    for offset in CUT_MARKER_OFFSETS:
        ax.plot(
            [value + offset for value in xs],
            ys,
            color="white",
            linewidth=CUT_MARKER_WHITE_LW,
            solid_capstyle="round",
            zorder=8,
            clip_on=False,
        )
        ax.plot(
            [value + offset for value in xs],
            ys,
            color="#111111",
            linewidth=CUT_MARKER_BLACK_LW,
            solid_capstyle="round",
            zorder=9,
            clip_on=False,
        )


def add_axis_cut_marker(ax: plt.Axes, x: float) -> None:
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ys = [-0.075 + 0.15 * idx / (CUT_MARKER_STEPS - 1) for idx in range(CUT_MARKER_STEPS)]
    xs = [
        x + CUT_MARKER_AMPLITUDE * math.sin((idx / (CUT_MARKER_STEPS - 1)) * 2.0 * math.pi)
        for idx in range(CUT_MARKER_STEPS)
    ]
    for offset in CUT_MARKER_OFFSETS:
        ax.plot(
            [value + offset for value in xs],
            ys,
            color="white",
            linewidth=CUT_MARKER_WHITE_LW,
            solid_capstyle="round",
            transform=transform,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [value + offset for value in xs],
            ys,
            color="#111111",
            linewidth=CUT_MARKER_BLACK_LW,
            solid_capstyle="round",
            transform=transform,
            zorder=11,
            clip_on=False,
        )


def add_transfer_annotation(ax: plt.Axes, y: float, left: float, width: float, font_size: float) -> None:
    if width <= 0.0:
        return
    center = left + width / 2.0
    label = f"{width:.2f} ms" if width < 0.1 else f"{width:.1f} ms"
    if width < 0.2:
        text_x = center + 2.25
        text_y = y - 0.50
        ha = "left"
    else:
        text_x = center + 1.15
        text_y = y - 0.52
        ha = "center"
    ax.annotate(
        f"Transfer: {label}",
        xy=(center, y),
        xytext=(text_x, text_y),
        ha=ha,
        va="center",
        fontsize=font_size - 3,
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "#111111",
            "lw": 1.2,
            "shrinkA": 3,
            "shrinkB": 2,
        },
        clip_on=False,
    )


def plot(systems: list[SystemBreakdown], args: argparse.Namespace) -> list[Path]:
    font_size = args.font_size
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size + 2,
            "legend.fontsize": font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    y_positions = list(range(len(systems)))
    bar_height = 0.58

    for y, system in zip(y_positions, systems):
        left = 0.0
        for phase_label, key, color, hatch in SEGMENT_ORDER:
            value = system.phase(key)
            if value <= 0.0:
                continue
            visible_width = max(min(left + value, args.max_x_ms) - left, 0.0)
            if visible_width > 0.0:
                ax.barh(
                    y,
                    visible_width,
                    left=left,
                    height=bar_height,
                    color=color,
                    edgecolor=SYSTEM_STYLES[system.system]["edgecolor"],
                    linewidth=1.2,
                    hatch=hatch,
                    label=phase_label,
                )
                if key == "transfer_ms":
                    add_transfer_annotation(ax, y, left, visible_width, font_size)
            left += value
            if left >= args.max_x_ms:
                break
        add_latency_label(ax, y, system.avg_latency_ms, args.max_x_ms, font_size)
        if system.avg_latency_ms > args.max_x_ms:
            add_cut_marker(ax, args.break_x_ms, y, bar_height)

    ax.set_xlabel("Average Query Latency (ms)", labelpad=12)
    ax.set_title(args.title, pad=18, fontweight="normal")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SYSTEM_LABELS[system.system] for system in systems])
    ax.set_xlim(0.0, args.max_x_ms)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_axis_cut_marker(ax, args.break_x_ms)

    handles = [
        Patch(facecolor=color, edgecolor="#000000", hatch=hatch, label=label)
        for label, _, color, hatch in SEGMENT_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d9d9d9",
        framealpha=0.92,
        columnspacing=0.95,
        handlelength=1.1,
        labelspacing=0.38,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.20, right=0.94, top=0.88, bottom=0.22)

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
    if args.chimera_log is None:
        args.chimera_log = latest_required_log(
            repo_root,
            "profiling/chimera_profile/logs/*/lotte_k100_r090_095_profile.log",
        )
    systems = [
        load_igp(args.igp_json),
        load_plaid(
            args.plaid_best_config,
            args.dataset,
            args.topk,
            args.target_recall,
            args.plaid_transfer_ms,
            args.plaid_latency_ms,
        ),
        load_chimera(args.chimera_log, args.chimera_label, args.chimera_transfer_ms),
    ]
    outputs = plot(systems, args)
    for system in systems:
        print(
            f"{system.system}: latency={system.avg_latency_ms:.3f} ms "
            f"recall={system.recall:.6f} qps={system.qps:.3f} config={system.config}"
        )
        print(
            "  "
            + ", ".join(
                f"{label}={system.phase(key):.3f} ms"
                for label, key, _, _ in SEGMENT_ORDER
            )
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
