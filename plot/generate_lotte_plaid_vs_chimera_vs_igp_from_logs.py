#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


font_size = 18

from generate_lotte_plaid_vs_chimera_from_logs import (
    PHASE_ORDER,
    SYSTEM_STYLES,
    SystemBreakdown,
    parse_chimera_log,
    parse_plaid_log,
)


SYSTEM_STYLES["IGP"] = {"edgecolor": "#000000"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    plaid_root = repo_root / "profiling" / "lotte" / "plaid_vs_gpu_search_recall90_k100"
    chimera_root = repo_root / "profiling" / "lotte" / "plaid_gpu_only_vs_gpu_search_recall90_k100"
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LoTTE PLAID vs Chimera vs IGP latency-breakdown plot "
            "from PLAID/Chimera logs and the IGP summary CSV."
        )
    )
    parser.add_argument(
        "--plaid-log",
        type=Path,
        default=plaid_root / "plaid_c4m.log",
    )
    parser.add_argument(
        "--chimera-log",
        type=Path,
        default=chimera_root / "gpu_search_v6_f4_profile_avg.log",
    )
    parser.add_argument(
        "--igp-json",
        type=Path,
        default=repo_root / "profiling" / "igp" / "lotte_k100_r90_t1.json",
    )
    parser.add_argument(
        "--dataset",
        default="lotte",
        help="Dataset name to select from the IGP summary CSV.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Top-k value to select from the IGP summary CSV.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.9,
        help="Target recall used to select the closest IGP row when no phi values are given.",
    )
    parser.add_argument(
        "--igp-phi-pb",
        type=int,
        default=None,
        help="Optional exact phi_pb value for the IGP row.",
    )
    parser.add_argument(
        "--igp-phi-ref",
        type=int,
        default=None,
        help="Optional exact phi_ref value for the IGP row.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "plot" / "lotte_plaid_vs_chimera_vs_igp_from_logs",
        help="Output path prefix without file extension.",
    )
    parser.add_argument(
        "--max-x-ms",
        type=float,
        default=30.0,
        help="Optional x-axis maximum in milliseconds.",
    )
    return parser.parse_args()


def csv_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"IGP summary row is missing {key}") from exc
    except ValueError as exc:
        raise ValueError(f"IGP summary row has non-numeric {key}={row[key]!r}") from exc


def csv_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except KeyError as exc:
        raise ValueError(f"IGP summary row is missing {key}") from exc
    except ValueError as exc:
        raise ValueError(f"IGP summary row has non-integer {key}={row[key]!r}") from exc


def load_igp_rows(path: Path, dataset: str, topk: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset", "").strip() == dataset and csv_int(row, "topk") == topk
        ]
    if not rows:
        raise ValueError(f"no IGP rows found in {path} for dataset={dataset!r}, topk={topk}")
    return rows


def select_igp_row(
    rows: list[dict[str, str]],
    target_recall: float,
    phi_pb: int | None,
    phi_ref: int | None,
) -> dict[str, str]:
    filtered = rows
    if phi_pb is not None:
        filtered = [row for row in filtered if csv_int(row, "phi_pb") == phi_pb]
    if phi_ref is not None:
        filtered = [row for row in filtered if csv_int(row, "phi_ref") == phi_ref]
    if not filtered:
        raise ValueError(f"no IGP rows match phi_pb={phi_pb}, phi_ref={phi_ref}")

    return max(
        filtered,
        key=lambda row: (
            -abs(csv_float(row, "recall_mean") - target_recall),
            csv_float(row, "qps"),
        ),
    )


def parse_igp_summary(
    path: Path,
    dataset: str,
    topk: int,
    target_recall: float,
    phi_pb: int | None,
    phi_ref: int | None,
) -> tuple[SystemBreakdown, dict[str, str]]:
    row = select_igp_row(load_igp_rows(path, dataset, topk), target_recall, phi_pb, phi_ref)
    try:
        timing = json.loads(row["search_time_m"])
    except KeyError as exc:
        raise ValueError(f"IGP summary row is missing search_time_m") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse IGP search_time_m JSON") from exc

    avg_latency_ms = csv_float(row, "avg_latency_ms")
    probe = float(timing["filter_time_average(ms)"])
    scan = float(timing["decode_time_average(ms)"])
    rescore = float(timing["refine_time_average(ms)"])
    raw_total_ms = probe + scan + rescore
    if raw_total_ms <= 0.0:
        raw_total_ms = avg_latency_ms

    system = SystemBreakdown(
        system="IGP",
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        probe_raw_ms=probe,
        transfer_raw_ms=0.0,
        scan_raw_ms=scan,
        rescore_raw_ms=rescore,
        ranking_raw_ms=0.0,
    )
    return system, row


def parse_igp_json(path: Path) -> tuple[SystemBreakdown, dict[str, object]]:
    payload = json.loads(path.read_text())
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"no IGP results found in {path}")
    result = results[0]
    timing = result.get("search_time_m", {})
    avg_latency_ms = float(result["avg_latency_ms"])
    probe = float(timing["filter_time_average(ms)"])
    scan = float(timing["decode_time_average(ms)"])
    rescore = float(timing["refine_time_average(ms)"])
    raw_total_ms = probe + scan + rescore
    if raw_total_ms <= 0.0:
        raw_total_ms = avg_latency_ms

    system = SystemBreakdown(
        system="IGP",
        avg_latency_ms=avg_latency_ms,
        raw_total_ms=raw_total_ms,
        probe_raw_ms=probe,
        transfer_raw_ms=0.0,
        scan_raw_ms=scan,
        rescore_raw_ms=rescore,
        ranking_raw_ms=0.0,
    )
    return system, {
        "dataset": payload.get("dataset", ""),
        "topk": payload.get("topk", ""),
        "query_split": payload.get("query_split", ""),
        "search_threads": payload.get("retrieval_parameters_from_paper", {}).get("search_threads", ""),
        "phi_pb": result.get("phi_pb", ""),
        "phi_ref": result.get("phi_ref", ""),
        "recall_mean": result.get("recall_mean", ""),
        "qps": result.get("qps", ""),
    }


def parse_chimera_log_with_transfer(path: Path) -> SystemBreakdown:
    system = parse_chimera_log(path)
    text = path.read_text()
    match = re.search(r"\[PROFILE_AVG\]\s+Total transfer:\s+([0-9.]+)\s+ms", text)
    if not match:
        return system

    transfer_ms = float(match.group(1))
    return SystemBreakdown(
        system=system.system,
        avg_latency_ms=system.avg_latency_ms,
        raw_total_ms=system.raw_total_ms,
        probe_raw_ms=system.probe_raw_ms,
        transfer_raw_ms=transfer_ms,
        scan_raw_ms=system.scan_raw_ms,
        rescore_raw_ms=max(system.rescore_raw_ms - transfer_ms, 0.0),
        ranking_raw_ms=system.ranking_raw_ms,
    )


def add_latency_label(
    ax: plt.Axes,
    y: float,
    total_ms: float,
) -> None:
    if total_ms <= 0.0:
        return
    label = f"{total_ms:.1f} ms"
    ax.text(
        total_ms + 0.40,
        y,
        label,
        ha="left",
        va="center",
        fontsize=font_size + 2,
        color="black",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def add_transfer_time_annotation(ax: plt.Axes, y: float, transfer_ms: float) -> None:
    if transfer_ms <= 0.0:
        return
    transfer_text = f"{transfer_ms:.2f}ms" if transfer_ms < 0.1 else f"{transfer_ms:.1f}ms"
    arrow_x = max(transfer_ms / 2.0, 0.02)
    ax.annotate(
        f"Transfer Time: {transfer_text}",
        xy=(arrow_x, y),
        xytext=(5.4, y - 0.62),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=font_size + 1,
        color="black",
        fontweight="normal",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "lw": 1.6,
            "shrinkA": 4,
            "shrinkB": 2,
        },
    )


def plot(output_root: Path, systems: list[SystemBreakdown], title: str, max_x_ms: float | None) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    y_gap = 1.45
    y_positions = [idx * y_gap for idx in range(len(systems))]
    bar_height = 0.6

    if max_x_ms is None:
        max_x_ms = max(system.avg_latency_ms for system in systems) * 1.18

    for y, system in zip(y_positions, systems):
        if system.system == "IGP":
            bar_width = min(system.avg_latency_ms, max_x_ms)
            ax.barh(
                y,
                bar_width,
                left=0.0,
                height=bar_height,
                color="#4C78A8",
                edgecolor=SYSTEM_STYLES[system.system]["edgecolor"],
                linewidth=1.6,
            )
            ax.text(
                max(bar_width - 0.35, 0.0),
                y,
                f"{system.avg_latency_ms:.0f} ms",
                ha="right",
                va="center",
                fontsize=font_size + 2,
                color="white",
                fontweight="bold",
            )
            continue

        edgecolor = SYSTEM_STYLES[system.system]["edgecolor"]
        bar_width = min(system.avg_latency_ms, max_x_ms)
        ax.barh(
            y,
            bar_width,
            left=0.0,
            height=bar_height,
            color="#E45756",
            edgecolor=edgecolor,
            linewidth=1.6,
        )
        transfer = min(system.scaled("transfer_ms"), bar_width)
        if transfer > 0.0:
            ax.barh(
                y,
                transfer,
                left=0.0,
                height=bar_height,
                color="#B279A2",
                edgecolor=edgecolor,
                linewidth=1.6,
                hatch="xx",
            )
            add_transfer_time_annotation(ax, y, system.scaled("transfer_ms"))
        add_latency_label(ax, y, system.avg_latency_ms)

    ax.set_xlabel("Average Query Latency (ms)", fontsize=font_size + 7, labelpad=14)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([system.system for system in systems], fontsize=font_size + 5)
    ax.tick_params(axis="x", labelsize=font_size + 3)
    ax.set_xlim(0.0, max_x_ms)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=font_size + 9, pad=22)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    phase_handles = [
        Patch(facecolor="#B279A2", edgecolor="#000000", hatch="xx", label="Transfer"),
    ]
    ax.legend(
        handles=phase_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d9d9d9",
        framealpha=0.95,
        fontsize=font_size + 4,
        handlelength=1.2,
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
    igp, igp_row = parse_igp_json(args.igp_json)
    systems = [
        igp,
        parse_plaid_log(args.plaid_log, "PLAID"),
        parse_chimera_log_with_transfer(args.chimera_log),
    ]

    title = "LoTTE: IGP vs. PLAID vs. Chimera (Recall@100 >= 0.9)"
    plot(args.output_root, systems, title, args.max_x_ms)

    print(f"[done] plot_png={args.output_root.with_suffix('.png')}")
    print(f"[done] plot_pdf={args.output_root.with_suffix('.pdf')}")
    print(
        "[igp] "
        f"dataset={igp_row['dataset']} topk={igp_row['topk']} "
        f"threads={igp_row['search_threads']} "
        f"phi_pb={igp_row['phi_pb']} phi_ref={igp_row['phi_ref']} "
        f"recall_mean={float(igp_row['recall_mean']):.6f} "
        f"qps={float(igp_row['qps']):.6f}"
    )
    for system in systems:
        print(
            f"[summary] system={system.system} avg_latency_ms={system.avg_latency_ms:.6f} "
            f"raw_total_ms={system.raw_total_ms:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
