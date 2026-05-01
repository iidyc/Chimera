#!/usr/bin/env python3
"""Generate main-experiment latency/throughput tables at target recalls."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from plot_paths import latest_required


DATASET_ORDER = ["lotte", "hotpot", "msmarco"]
DATASET_LABELS = {
    "lotte": "LoTTE",
    "hotpot": "HotpotQA",
    "msmarco": "MS MARCO",
}
TOPK_ORDER = ["10", "100"]
TARGET_ORDER = ["0.90", "0.95"]
SYSTEM_ORDER = ["plaid", "plaid_plus", "igp", "chimera"]
SYSTEM_LABELS = {
    "plaid": "PLAID",
    "plaid_plus": "PLAID+",
    "igp": "IGP",
    "chimera": "Chimera",
}
LEGACY_CHIMERA_SYSTEM = "v" "8"
CHIMERA_SYSTEM_ALIASES = {"chimera", "gpu_search", LEGACY_CHIMERA_SYSTEM}


@dataclass(frozen=True)
class Point:
    system: str
    dataset: str
    topk: str
    recall: float
    qps: float
    latency_ms: float
    label: str
    source: str
    meets_target: bool = False


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate main-experiment latency/throughput tables for target recall levels."
    )
    parser.add_argument("--plaid", type=Path, default=repo_root / "profiling" / "plaid" / "summary.csv")
    parser.add_argument("--plaid-plus", type=Path, default=repo_root / "profiling" / "plaid_plus" / "summary.csv")
    parser.add_argument("--igp", type=Path, default=repo_root / "profiling" / "igp" / "summary_t32.csv")
    parser.add_argument(
        "--chimera",
        type=Path,
        default=None,
        help="Defaults to latest profiling/chimera_topk_baseline/chimera_topk_baseline_summary_*.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figure")
    parser.add_argument("--output-stem", default="main_experiment_latency_throughput")
    return parser.parse_args()


def select_at_target(points: list[Point], target: str) -> Point | None:
    target_value = float(target)
    meeting = [point for point in points if point.recall >= target_value]
    if meeting:
        chosen = max(meeting, key=lambda point: point.qps)
        return Point(**{**chosen.__dict__, "meets_target": True})
    if points:
        chosen = max(points, key=lambda point: (point.recall, point.qps))
        return Point(**{**chosen.__dict__, "meets_target": False})
    return None


def read_plaid_like(path: Path, system: str) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("k") not in TOPK_ORDER:
                continue
            points.append(
                Point(
                    system=system,
                    dataset=row["dataset"],
                    topk=row["k"],
                    recall=float(row["recall"]),
                    qps=float(row["qps"]),
                    latency_ms=float(row["avg_latency_ms"]),
                    label=row["label"],
                    source=row.get("source", str(path)),
                )
            )
    return points


def read_igp(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "val" or row.get("topk") not in TOPK_ORDER:
                continue
            if row.get("metric_name") != f"recall@{row['topk']}":
                continue
            points.append(
                Point(
                    system="igp",
                    dataset=row["dataset"],
                    topk=row["topk"],
                    recall=float(row["recall_mean"]),
                    qps=float(row["throughput_qps"]),
                    latency_ms=float(row["avg_latency_ms"]),
                    label=f"phi_pb={row['phi_pb']}, phi_ref={row['phi_ref']}",
                    source=row.get("chunk_json", str(path)),
                )
            )
    return points


def read_chimera(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") not in CHIMERA_SYSTEM_ALIASES or row.get("k") != "100":
                continue
            for topk, recall_column in (("10", "recall_at_10_mean"), ("100", "recall_at_100_mean")):
                points.append(
                    Point(
                        system="chimera",
                        dataset=row["dataset"],
                        topk=topk,
                        recall=float(row[recall_column]),
                        qps=float(row["qps_mean"]),
                        latency_ms=float(row["avg_latency_ms_mean"]),
                        label=f"{row['label']} (k=100 run)",
                        source=str(path),
                    )
                )
    return points


def collect_selected(points: list[Point]) -> dict[tuple[str, str, str, str], Point]:
    selected: dict[tuple[str, str, str, str], Point] = {}
    for dataset in DATASET_ORDER:
        for topk in TOPK_ORDER:
            for target in TARGET_ORDER:
                for system in SYSTEM_ORDER:
                    candidates = [
                        point
                        for point in points
                        if point.dataset == dataset and point.topk == topk and point.system == system
                    ]
                    point = select_at_target(candidates, target)
                    if point is not None:
                        selected[(dataset, topk, target, system)] = point
    return selected


def fmt_float(value: float, digits: int) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_point(point: Point | None) -> str:
    if point is None:
        return "N/A"
    suffix = "" if point.meets_target else "*"
    return f"{point.qps:.1f} qps / R={point.recall:.4f}{suffix}"


def speedups(
    selected: dict[tuple[str, str, str, str], Point],
    dataset: str,
    topk: str,
    target: str,
) -> tuple[float, float]:
    plaid = selected.get((dataset, topk, target, "plaid"))
    chimera = selected.get((dataset, topk, target, "chimera"))
    if plaid is None or chimera is None or not plaid.meets_target or not chimera.meets_target:
        return math.nan, math.nan
    return chimera.qps / plaid.qps, plaid.latency_ms / chimera.latency_ms


def write_long_csv(selected: dict[tuple[str, str, str, str], Point], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "topk",
                "target_recall",
                "system",
                "achieved_recall",
                "qps",
                "latency_ms",
                "meets_target",
                "label",
                "source",
            ],
        )
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for topk in TOPK_ORDER:
                for target in TARGET_ORDER:
                    for system in SYSTEM_ORDER:
                        point = selected.get((dataset, topk, target, system))
                        row = {
                            "dataset": dataset,
                            "topk": topk,
                            "target_recall": target,
                            "system": system,
                            "achieved_recall": "",
                            "qps": "",
                            "latency_ms": "",
                            "meets_target": "",
                            "label": "",
                            "source": "",
                        }
                        if point is not None:
                            row.update(
                                {
                                    "achieved_recall": fmt_float(point.recall, 6),
                                    "qps": fmt_float(point.qps, 6),
                                    "latency_ms": fmt_float(point.latency_ms, 6),
                                    "meets_target": str(point.meets_target),
                                    "label": point.label,
                                    "source": point.source,
                                }
                            )
                        writer.writerow(row)


def write_speedup_csv(selected: dict[tuple[str, str, str, str], Point], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "topk",
                "target_recall",
                "plaid_qps",
                "plaid_latency_ms",
                "chimera_qps",
                "chimera_latency_ms",
                "qps_speedup_chimera_vs_plaid",
                "latency_speedup_chimera_vs_plaid",
            ],
        )
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for topk in TOPK_ORDER:
                for target in TARGET_ORDER:
                    plaid = selected.get((dataset, topk, target, "plaid"))
                    chimera = selected.get((dataset, topk, target, "chimera"))
                    qps_speedup, latency_speedup = speedups(selected, dataset, topk, target)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "topk": topk,
                            "target_recall": target,
                            "plaid_qps": fmt_float(plaid.qps, 6) if plaid else "",
                            "plaid_latency_ms": fmt_float(plaid.latency_ms, 6) if plaid else "",
                            "chimera_qps": fmt_float(chimera.qps, 6) if chimera else "",
                            "chimera_latency_ms": fmt_float(chimera.latency_ms, 6) if chimera else "",
                            "qps_speedup_chimera_vs_plaid": fmt_float(qps_speedup, 6),
                            "latency_speedup_chimera_vs_plaid": fmt_float(latency_speedup, 6),
                        }
                    )


def write_markdown(selected: dict[tuple[str, str, str, str], Point], path: Path) -> None:
    lines = [
        "| Dataset | Top-k | Target | PLAID | PLAID+ | IGP | Chimera | Chimera/PLAID QPS Speedup |",
        "|---|---:|---:|---|---|---|---|---:|",
    ]
    for dataset in DATASET_ORDER:
        for topk in TOPK_ORDER:
            for target in TARGET_ORDER:
                qps_speedup, latency_speedup = speedups(selected, dataset, topk, target)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            DATASET_LABELS[dataset],
                            topk,
                            target,
                            fmt_point(selected.get((dataset, topk, target, "plaid"))),
                            fmt_point(selected.get((dataset, topk, target, "plaid_plus"))),
                            fmt_point(selected.get((dataset, topk, target, "igp"))),
                            fmt_point(selected.get((dataset, topk, target, "chimera"))),
                            f"{qps_speedup:.2f}x" if not math.isnan(qps_speedup) else "N/A",
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "*Each system entry is `throughput / achieved recall`.*",
            "*Rows marked with `*` are the best available point for that system but do not reach the target recall; Chimera-vs-PLAID speedups are reported only when both systems meet the target.*",
            "*PLAID+ has no local MS MARCO rows because it ran out of GPU memory in the main experiment.*",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.chimera is None:
        args.chimera = latest_required(
            repo_root,
            "profiling/chimera_topk_baseline",
            "chimera_topk_baseline_summary_*.csv",
        )
    points: list[Point] = []
    points.extend(read_plaid_like(args.plaid, "plaid"))
    points.extend(read_plaid_like(args.plaid_plus, "plaid_plus"))
    points.extend(read_igp(args.igp))
    points.extend(read_chimera(args.chimera))

    selected = collect_selected(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.output_dir / f"{args.output_stem}_selected.csv"
    speedup_csv = args.output_dir / f"{args.output_stem}_chimera_vs_plaid_speedup.csv"
    markdown = args.output_dir / f"{args.output_stem}.md"
    write_long_csv(selected, long_csv)
    write_speedup_csv(selected, speedup_csv)
    write_markdown(selected, markdown)

    print(f"plaid={args.plaid}")
    print(f"plaid_plus={args.plaid_plus}")
    print(f"igp={args.igp}")
    print(f"chimera={args.chimera}")
    print(long_csv)
    print(speedup_csv)
    print(markdown)


if __name__ == "__main__":
    main()
