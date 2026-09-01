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
    ("Probe", "probe_ms", "#4C78A8", ".."),
    ("Transfer", "transfer_ms", "#B279A2", "xx"),
    ("Scan", "scan_ms", "#F58518", ""),
    ("Rescore", "rescore_ms", "#54A24B", "oo"),
    ("Ranking", "ranking_ms", "#E45756", "\\\\"),
]

SYSTEM_STYLES = {
    "PLAID": {"edgecolor": "#000000"},
    "PLAID+": {"edgecolor": "#000000"},
    "Chimera": {"edgecolor": "#000000"},
}


@dataclass
class SystemBreakdown:
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
    plaid_root = repo_root / "profiling" / "lotte" / "plaid_vs_gpu_search_recall90_k100"
    plaid_gpu_only_root = repo_root / "profiling" / "lotte" / "plaid_gpu_only_vs_gpu_search_recall90_k100"
    default_output_root = repo_root / "plot" / "lotte_plaid_vs_plaid_gpu_only_vs_chimera_from_logs"
    parser = argparse.ArgumentParser(
        description="Generate a LoTTE PLAID vs PLAID (GPU-only) vs Chimera latency-breakdown plot directly from logs."
    )
    parser.add_argument(
        "--plaid-log",
        type=Path,
        default=plaid_root / "plaid_c4m.log",
    )
    parser.add_argument(
        "--plaid-gpu-only-log",
        type=Path,
        default=plaid_gpu_only_root / "plaid_gpu_resident_c4m.log",
    )
    parser.add_argument(
        "--chimera-log",
        type=Path,
        default=plaid_gpu_only_root / "gpu_search_v6_f4_profile_avg.log",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Output path prefix without file extension.",
    )
    parser.add_argument(
        "--omit-plaid-plus",
        action="store_true",
        help="Generate the PLAID vs Chimera version without PLAID+.",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    args.default_output_root = default_output_root
    return args


def read_text(path: Path) -> str:
    return path.read_text()


def parse_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"failed to parse {label} from log")
    return float(match.group(1))


def transfer_map(text: str) -> dict[str, float]:
    transfers = {}
    for name, value in re.findall(r"\[PROFILE_TRANSFER\] name=([^ ]+) avg_ms=([0-9.]+)", text):
        transfers[name] = float(value)
    return transfers


def parse_plaid_log(path: Path, system_label: str) -> SystemBreakdown:
    text = read_text(path)
    transfers = transfer_map(text)
    avg_latency_ms = parse_float(r"avg_time=([0-9.]+)s", text, f"{system_label} latency") * 1000.0
    raw_total_ms = parse_float(r"\[PROFILE_STAGE\] name=search\.total avg_ms=([0-9.]+)", text, f"{system_label} total")
    probe = (
        parse_float(r"\[PROFILE_STAGE\] name=candidate\.centroid_score avg_ms=([0-9.]+)", text, f"{system_label} centroid score")
        + parse_float(r"\[PROFILE_STAGE\] name=candidate\.cell_select avg_ms=([0-9.]+)", text, f"{system_label} cell select")
    )
    scan_lookup = parse_float(r"\[PROFILE_STAGE\] name=approx\.pruned\.lookup_codes avg_ms=([0-9.]+)", text, f"{system_label} pruned lookup")
    rescore_lookup = parse_float(r"\[PROFILE_STAGE\] name=approx\.full\.lookup_codes avg_ms=([0-9.]+)", text, f"{system_label} full lookup")
    ranking_codes_lookup = parse_float(r"\[PROFILE_STAGE\] name=final\.lookup_codes avg_ms=([0-9.]+)", text, f"{system_label} final lookup codes")
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
        parse_float(r"\[PROFILE_STAGE\] name=candidate\.ivf_lookup avg_ms=([0-9.]+)", text, f"{system_label} ivf lookup")
        + parse_float(r"\[PROFILE_STAGE\] name=candidate\.pid_dedup avg_ms=([0-9.]+)", text, f"{system_label} pid dedup")
        + scan_lookup
        + parse_float(r"\[PROFILE_STAGE\] name=approx\.pruned\.reduce avg_ms=([0-9.]+)", text, f"{system_label} pruned reduce")
        + parse_float(r"\[PROFILE_STAGE\] name=approx\.pruned\.topk avg_ms=([0-9.]+)", text, f"{system_label} pruned topk")
    )
    rescore_original = (
        rescore_lookup
        + parse_float(r"\[PROFILE_STAGE\] name=approx\.full\.reduce avg_ms=([0-9.]+)", text, f"{system_label} full reduce")
        + parse_float(r"\[PROFILE_STAGE\] name=approx\.full\.topk avg_ms=([0-9.]+)", text, f"{system_label} full topk")
    )
    ranking_original = (
        ranking_codes_lookup
        + parse_float(r"\[PROFILE_STAGE\] name=final\.lookup_residuals avg_ms=([0-9.]+)", text, f"{system_label} final lookup residuals")
        + parse_float(r"\[PROFILE_STAGE\] name=final\.decompress avg_ms=([0-9.]+)", text, f"{system_label} final decompress")
        + parse_float(r"\[PROFILE_STAGE\] name=final\.matmul avg_ms=([0-9.]+)", text, f"{system_label} final matmul")
        + parse_float(r"\[PROFILE_STAGE\] name=final\.pack_scores avg_ms=([0-9.]+)", text, f"{system_label} final pack")
        + parse_float(r"\[PROFILE_STAGE\] name=final\.reduce avg_ms=([0-9.]+)", text, f"{system_label} final reduce")
        + parse_float(r"\[PROFILE_STAGE\] name=rank\.sort avg_ms=([0-9.]+)", text, f"{system_label} rank sort")
    )
    overlapped_scan_transfer = transfers.get("ivf", 0.0) + scan_codes_transfer
    overlapped_rescore_transfer = rescore_codes_transfer
    overlapped_ranking_transfer = transfers.get("residuals", 0.0) + ranking_codes_transfer
    transfer = sum(transfers.values())
    scan = max(scan_original - overlapped_scan_transfer, 0.0)
    rescore = max(rescore_original - overlapped_rescore_transfer, 0.0)
    ranking = max(ranking_original - overlapped_ranking_transfer, 0.0)
    return SystemBreakdown(
        system=system_label,
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        probe_raw_ms=probe,
        transfer_raw_ms=transfer,
        scan_raw_ms=scan,
        rescore_raw_ms=rescore,
        ranking_raw_ms=ranking,
    )


def parse_chimera_log(path: Path) -> SystemBreakdown:
    text = read_text(path)
    avg_latency_ms = parse_float(r"Average latency per query:\s+([0-9.]+)\s+ms", text, "Chimera latency")
    raw_total_ms = parse_float(r"\[PROFILE_AVG\] Total search time\s+:\s+([0-9.]+)\s+ms", text, "Chimera total")
    stage1 = parse_float(r"\[PROFILE_AVG\] Stage 1 time:\s+([0-9.]+)\s+ms", text, "Chimera stage1")
    probe = parse_float(r"\[PROFILE_AVG\]\s+1\. CAGRA search\s+:\s+([0-9.]+)\s+ms", text, "Chimera probe")
    scan = max(stage1 - probe, 0.0)
    ranking = (
        parse_float(r"gpu_extract=([0-9.]+)\s+ms", text, "Chimera gpu extract")
        + parse_float(r"cpu_ip_ex=([0-9.]+)\s+ms", text, "Chimera cpu ip ex")
        + parse_float(r"wait_extract=([0-9.]+)\s+ms", text, "Chimera wait extract")
        + parse_float(r"combine=([0-9.]+)\s+ms", text, "Chimera combine")
    )
    rescore = max(raw_total_ms - probe - scan - ranking, 0.0)
    return SystemBreakdown(
        system="Chimera",
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        probe_raw_ms=probe,
        transfer_raw_ms=0.0,
        scan_raw_ms=scan,
        rescore_raw_ms=rescore,
        ranking_raw_ms=ranking,
    )


def add_segment_labels(ax: plt.Axes, system: str, y: float, left: float, width: float, label: str) -> None:
    if system != "PLAID":
        return
    if width < 0.42:
        return
    x = left + width / 2.0
    if label == "Rescore":
        x -= min(0.30, width * 0.20)
    elif label == "Ranking":
        x += min(0.18, width * 0.12)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
    )


def add_total_label(ax: plt.Axes, y: float, total_ms: float) -> None:
    if total_ms <= 0.0:
        return
    ax.text(
        total_ms + 0.10,
        y,
        f"{total_ms:.1f} ms",
        ha="left",
        va="center",
        fontsize=13,
        color="black",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def plot(output_root: Path, systems: list[SystemBreakdown], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    y_positions = list(range(len(systems)))
    bar_height = 0.6

    for y, system in zip(y_positions, systems):
        left = 0.0
        edgecolor = SYSTEM_STYLES[system.system]["edgecolor"]
        for label, key, color, hatch in PHASE_ORDER:
            value = system.scaled(key)
            if value <= 0.0:
                continue
            ax.barh(
                y,
                value,
                left=left,
                height=bar_height,
                color=color,
                edgecolor=edgecolor,
                linewidth=1.6,
                hatch=hatch,
            )
            add_segment_labels(ax, system.system, y, left, value, label)
            left += value
        add_total_label(ax, y, system.avg_latency_ms)

    ax.set_xlabel("Average Query Latency Contribution (ms)", fontsize=15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([system.system for system in systems], fontsize=15)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(0.0, 22.0)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=17)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    phase_handles = [
        Patch(facecolor=color, edgecolor="#000000", hatch=hatch, label=label)
        for label, _, color, hatch in PHASE_ORDER
    ]
    ax.legend(
        handles=phase_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.39),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d9d9d9",
        framealpha=0.95,
        fontsize=18,
        handlelength=1.1,
        labelspacing=0.55,
        borderpad=0.5,
    )
    fig.tight_layout()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    systems = [parse_plaid_log(args.plaid_log, "PLAID")]
    if not args.omit_plaid_plus:
        systems.append(parse_plaid_log(args.plaid_gpu_only_log, "PLAID+"))
    systems.append(parse_chimera_log(args.chimera_log))

    output_root = args.output_root
    if args.omit_plaid_plus and output_root == args.default_output_root:
        output_root = args.repo_root / "plot" / "lotte_plaid_vs_chimera_from_logs"

    title = (
        "LoTTE: PLAID vs Chimera (Recall@100 = 0.9)"
        if args.omit_plaid_plus
        else "LoTTE: PLAID vs PLAID+ vs Chimera (Recall@100 = 0.9)"
    )

    plot(output_root, systems, title)
    print(f"[done] plot_png={output_root.with_suffix('.png')}")
    print(f"[done] plot_pdf={output_root.with_suffix('.pdf')}")
    for system in systems:
        print(
            f"[summary] system={system.system} avg_latency_ms={system.avg_latency_ms:.6f} "
            f"raw_total_ms={system.raw_total_ms:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
