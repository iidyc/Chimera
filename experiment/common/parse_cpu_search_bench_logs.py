#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DRIVER_RE = re.compile(r"^\[driver\] ([A-Za-z0-9_]+)=(.*)$")
CONFIG_RE = re.compile(
    r"^\[CONFIG\] label=(?P<label>\S+) "
    r"nprobe=(?P<nprobe>\d+) "
    r"k_rank_cluster=(?P<k_rank_cluster>\d+) "
    r"k_rank_all_tokens=(?P<k_rank_all_tokens>\d+) "
    r"itopk_size=(?P<itopk_size>\d+) "
    r"overlap_chunks=(?P<overlap_chunks>\d+)$"
)
TIMING_RE = re.compile(
    r"^\[(?P<search_seconds>[0-9.]+) s\] "
    r"label=(?P<label>\S+) "
    r"(?:v[0-9]+ )?(?:CPU|GPU) search time for (?P<queries>\d+) queries\.$"
)
END_TO_END_RE = re.compile(r"^\[SEARCH\] End-to-end measured time: ([0-9.]+) ms$")
THROUGHPUT_RE = re.compile(r"^\[SEARCH\] Throughput: ([0-9.]+) qps$")
AVG_LATENCY_RE = re.compile(r"^\[SEARCH\] Average latency per query: ([0-9.]+) ms$")
RECALL_RE = re.compile(r"^Recall@(\d+): ([0-9.]+)$")

RESULTS_HEADER = [
    "implementation",
    "dataset",
    "batch_id",
    "label",
    "nprobe",
    "k_rank_cluster",
    "k_rank_all_tokens",
    "itopk_size",
    "overlap_chunks",
    "index_source",
    "index_path",
    "queries",
    "k",
    "warmup",
    "search_seconds",
    "end_to_end_ms",
    "qps",
    "recall",
    "avg_latency_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "stddev_ms",
    "cpu_bind",
    "omp_num_threads",
    "log_file",
]


class ParseError(RuntimeError):
    pass


@dataclass
class ParsedLog:
    implementation: str
    dataset: str
    batch_id: str
    results_csv: Path
    pareto_csv: Path
    rows: list[dict[str, str]]
    expected_total_configs: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse cpu_search raw benchmark logs produced by the chunked "
            "benchmark launcher and combine them into one table per dataset."
        )
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Only parse logs for the given batch id.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write combined CSVs even if the number of configs is smaller than the expected batch total.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the repo root recorded in the log when resolving output paths.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="One or more log files or directories containing benchmark_*.log files.",
    )
    return parser.parse_args()


def parse_driver_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in lines:
        match = DRIVER_RE.match(raw_line.strip())
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


def require_metadata(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "")
    if not value:
        raise ParseError(f"missing [driver] {key}=... in log")
    return value


def remap_repo_path(raw_path: str, metadata: dict[str, str], repo_root_override: Path | None) -> Path:
    path = Path(raw_path)
    if repo_root_override is None:
        return path

    logged_repo_root = metadata.get("repo_root")
    if not logged_repo_root:
        return path

    try:
        relative_path = path.relative_to(Path(logged_repo_root))
    except ValueError:
        return path
    return repo_root_override / relative_path


def parse_key_value_fields(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in payload.split(","):
        chunk = chunk.strip()
        if "=" not in chunk:
            raise ParseError(f"unable to parse key=value field from: {payload}")
        key, value = chunk.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_dict_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def compute_pareto_rows(rows: list[dict[str, str]], qps_key: str, recall_key: str) -> list[dict[str, str]]:
    frontier: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        qps = float(row[qps_key])
        recall = float(row[recall_key])
        dominated = False
        for other_idx, other in enumerate(rows):
            if idx == other_idx:
                continue
            other_qps = float(other[qps_key])
            other_recall = float(other[recall_key])
            if (
                other_qps >= qps
                and other_recall >= recall
                and (other_qps > qps or other_recall > recall)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)

    frontier.sort(key=lambda row: (float(row[qps_key]), float(row[recall_key])))
    return frontier


def parse_log(log_file: Path, repo_root_override: Path | None, batch_id_filter: str | None) -> ParsedLog | None:
    lines = log_file.read_text().splitlines()
    metadata = parse_driver_metadata(lines)

    implementation = require_metadata(metadata, "implementation")
    dataset = require_metadata(metadata, "dataset")
    batch_id = require_metadata(metadata, "batch_id")
    if batch_id_filter is not None and batch_id != batch_id_filter:
        return None

    index_path = metadata.get("active_index_dir") or metadata.get("source_index_dir")
    if not index_path:
        raise ParseError(
            f"missing [driver] active_index_dir=... or [driver] source_index_dir=... in log {log_file}"
        )

    k = require_metadata(metadata, "k")
    warmup = require_metadata(metadata, "warmup")
    output_dir = remap_repo_path(require_metadata(metadata, "output_dir"), metadata, repo_root_override)
    results_csv = output_dir / dataset / implementation / f"benchmark_results_k{k}_{batch_id}.csv"
    pareto_csv = output_dir / dataset / implementation / f"pareto_frontier_k{k}_{batch_id}.csv"
    cpu_bind = metadata.get("cpu_bind", "")
    omp_num_threads = metadata.get("omp_num_threads", "")
    index_source = metadata.get("index_source", "nfs")
    expected_total_configs_raw = metadata.get("batch_total_config_count")
    expected_total_configs = int(expected_total_configs_raw) if expected_total_configs_raw else None

    rows: list[dict[str, str]] = []
    current_config: dict[str, str] | None = None
    current_metrics: dict[str, str] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = CONFIG_RE.match(line)
        if match:
            if current_config is not None:
                raise ParseError(f"incomplete metric block before next config in {log_file}")
            current_config = match.groupdict()
            current_metrics = {}
            continue

        if current_config is None:
            continue

        match = TIMING_RE.match(line)
        if match:
            if match.group("label") != current_config["label"]:
                raise ParseError(
                    f"timing label mismatch in {log_file}: expected {current_config['label']}, got {match.group('label')}"
                )
            current_metrics["search_seconds"] = match.group("search_seconds")
            current_metrics["queries"] = match.group("queries")
            continue

        match = END_TO_END_RE.match(line)
        if match:
            current_metrics["end_to_end_ms"] = match.group(1)
            continue

        match = THROUGHPUT_RE.match(line)
        if match:
            current_metrics["qps"] = match.group(1)
            continue

        match = AVG_LATENCY_RE.match(line)
        if match:
            current_metrics["avg_latency_ms"] = match.group(1)
            continue

        if line.startswith("[SEARCH] Query latency distribution (ms): "):
            latency_fields = parse_key_value_fields(line.split(": ", 1)[1])
            for field_name, output_name in (
                ("p50", "p50_ms"),
                ("p90", "p90_ms"),
                ("p95", "p95_ms"),
                ("p99", "p99_ms"),
                ("max", "max_ms"),
                ("stddev", "stddev_ms"),
            ):
                if field_name not in latency_fields:
                    raise ParseError(f"missing {field_name}=... in latency line: {line}")
                current_metrics[output_name] = latency_fields[field_name]
            continue

        match = RECALL_RE.match(line)
        if match:
            recall_k = match.group(1)
            current_metrics["recall"] = match.group(2)
            if recall_k != k:
                raise ParseError(f"recall depth mismatch in {log_file}: expected {k}, got {recall_k}")

            required_metrics = (
                "queries",
                "search_seconds",
                "end_to_end_ms",
                "qps",
                "recall",
                "avg_latency_ms",
                "p50_ms",
                "p90_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
                "stddev_ms",
            )
            missing = [name for name in required_metrics if name not in current_metrics]
            if missing:
                raise ParseError(
                    f"incomplete metric block for label={current_config['label']} in {log_file}: missing {', '.join(missing)}"
                )

            rows.append(
                {
                    "implementation": implementation,
                    "dataset": dataset,
                    "batch_id": batch_id,
                    "label": current_config["label"],
                    "nprobe": current_config["nprobe"],
                    "k_rank_cluster": current_config["k_rank_cluster"],
                    "k_rank_all_tokens": current_config["k_rank_all_tokens"],
                    "itopk_size": current_config["itopk_size"],
                    "overlap_chunks": current_config["overlap_chunks"],
                    "index_source": index_source,
                    "index_path": index_path,
                    "queries": current_metrics["queries"],
                    "k": k,
                    "warmup": warmup,
                    "search_seconds": current_metrics["search_seconds"],
                    "end_to_end_ms": current_metrics["end_to_end_ms"],
                    "qps": current_metrics["qps"],
                    "recall": current_metrics["recall"],
                    "avg_latency_ms": current_metrics["avg_latency_ms"],
                    "p50_ms": current_metrics["p50_ms"],
                    "p90_ms": current_metrics["p90_ms"],
                    "p95_ms": current_metrics["p95_ms"],
                    "p99_ms": current_metrics["p99_ms"],
                    "max_ms": current_metrics["max_ms"],
                    "stddev_ms": current_metrics["stddev_ms"],
                    "cpu_bind": cpu_bind,
                    "omp_num_threads": omp_num_threads,
                    "log_file": str(log_file),
                }
            )
            current_config = None
            current_metrics = {}

    if current_config is not None:
        raise ParseError(f"incomplete trailing metric block in {log_file}")
    if not rows:
        raise ParseError(f"no benchmark rows found in {log_file}")

    return ParsedLog(
        implementation=implementation,
        dataset=dataset,
        batch_id=batch_id,
        results_csv=results_csv,
        pareto_csv=pareto_csv,
        rows=rows,
        expected_total_configs=expected_total_configs,
    )


def discover_log_files(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file():
            discovered.append(path)
            continue
        if path.is_dir():
            discovered.extend(sorted(path.rglob("benchmark_*.log")))
            continue
        raise ParseError(f"path does not exist: {path}")
    return sorted(set(discovered))


def main() -> int:
    args = parse_args()

    try:
        log_files = discover_log_files(args.paths)
        if not log_files:
            raise ParseError("no benchmark_*.log files found")

        grouped_rows: dict[tuple[str, str, str, Path, Path], list[dict[str, str]]] = defaultdict(list)
        grouped_expected: dict[tuple[str, str, str, Path, Path], int | None] = {}

        for log_file in log_files:
            parsed = parse_log(log_file, args.repo_root, args.batch_id)
            if parsed is None:
                continue
            group_key = (
                parsed.implementation,
                parsed.dataset,
                parsed.batch_id,
                parsed.results_csv,
                parsed.pareto_csv,
            )
            grouped_rows[group_key].extend(parsed.rows)
            expected = grouped_expected.get(group_key)
            if expected is None:
                grouped_expected[group_key] = parsed.expected_total_configs
            elif parsed.expected_total_configs is not None and expected != parsed.expected_total_configs:
                raise ParseError(
                    f"conflicting expected config counts for batch={parsed.batch_id} dataset={parsed.dataset}"
                )

        if not grouped_rows:
            raise ParseError("no logs matched the requested batch id")

        for group_key, rows in grouped_rows.items():
            implementation, dataset, batch_id, results_csv, pareto_csv = group_key
            rows_by_label: dict[str, dict[str, str]] = {}
            for row in rows:
                label = row["label"]
                if label in rows_by_label:
                    raise ParseError(
                        f"duplicate label={label} for batch={batch_id} dataset={dataset}; "
                        f"refusing to merge ambiguous logs"
                    )
                rows_by_label[label] = row

            merged_rows = [rows_by_label[label] for label in sorted(rows_by_label)]
            expected_total = grouped_expected[group_key]
            if expected_total is not None and len(merged_rows) != expected_total and not args.allow_incomplete:
                raise ParseError(
                    f"batch={batch_id} dataset={dataset} has {len(merged_rows)} config rows, expected {expected_total}"
                )

            write_dict_csv(results_csv, RESULTS_HEADER, merged_rows)
            write_dict_csv(
                pareto_csv,
                RESULTS_HEADER,
                compute_pareto_rows(merged_rows, qps_key="qps", recall_key="recall"),
            )

            print(
                f"[ok] implementation={implementation} dataset={dataset} batch_id={batch_id} "
                f"rows={len(merged_rows)} results_csv={results_csv} pareto_csv={pareto_csv}"
            )

    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
