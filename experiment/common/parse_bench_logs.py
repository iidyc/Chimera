#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DRIVER_RE = re.compile(r"^\[driver\] ([A-Za-z0-9_]+)=(.*)$")

GPU_CONFIG_RE = re.compile(
    r"^\[CONFIG\] label=(?P<label>\S+) "
    r"nprobe=(?P<nprobe>\d+) "
    r"k_rank_cluster=(?P<k_rank_cluster>\d+) "
    r"k_rank_all_tokens=(?P<k_rank_all_tokens>\d+) "
    r"itopk_size=(?P<itopk_size>\d+) "
    r"overlap_chunks=(?P<overlap_chunks>\d+)$"
)
GPU_TIMING_RE = re.compile(
    r"^\[(?P<search_seconds>[0-9.]+) s\] "
    r"label=(?P<label>\S+) "
    r"(?:v[0-9]+ )?GPU search time for (?P<queries>\d+) queries\.$"
)
GPU_END_TO_END_RE = re.compile(r"^\[SEARCH\] End-to-end measured time: ([0-9.]+) ms$")
GPU_THROUGHPUT_RE = re.compile(r"^\[SEARCH\] Throughput: ([0-9.]+) qps$")
GPU_AVG_LATENCY_RE = re.compile(r"^\[SEARCH\] Average latency per query: ([0-9.]+) ms$")
GPU_RECALL_RE = re.compile(r"^Recall@(\d+): ([0-9.]+)$")
GPU_MEM_RE = re.compile(
    r"^\[GPU_MEM\] current=(?P<current>[0-9.]+) MiB, "
    r"peak=(?P<peak>[0-9.]+) MiB, "
    r"total=(?P<total>[0-9.]+) MiB, "
    r"samples=(?P<samples>\d+), "
    r"peak_label=(?P<peak_label>.+)$"
)

COLBERT_CONFIG_RE = re.compile(
    r"^\[CONFIG\] label=(?P<label>\S+) ncells=(?P<ncells>\d+) ndocs=(?P<ndocs>\d+)$"
)
COLBERT_QUERY_COUNT_RE = re.compile(
    r"^Expecting to read ([0-9]+) document embeddings of dimension [0-9]+ from .*$"
)
COLBERT_RESULT_RE = re.compile(
    r"^ncells=(?P<ncells>\d+) ndocs=(?P<ndocs>\d+) "
    r"avg_time=(?P<avg_time>[0-9.]+)s recall@[0-9]+=(?P<recall>[0-9.]+)$"
)


GPU_RESULTS_HEADER = [
    "implementation",
    "dataset",
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
    "gpu_mem_current_mib",
    "gpu_mem_peak_mib",
    "gpu_mem_total_mib",
    "gpu_mem_samples",
    "gpu_mem_peak_label",
]

COLBERT_RESULTS_HEADER = [
    "implementation",
    "dataset",
    "label",
    "ncells",
    "ndocs",
    "index_source",
    "index_root",
    "index_path",
    "query_file",
    "gt_file",
    "queries",
    "k",
    "num_runs",
    "avg_time_s",
    "qps",
    "recall",
]

COLBERT_SEARCH_OUTPUT_HEADER = ["avg_time", "recall"]


class ParseError(RuntimeError):
    pass


@dataclass
class ParsedOutputs:
    mode: str
    log_file: Path
    results_csv: Path
    pareto_csv: Path
    row_count: int


class HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=(
            "Parse raw benchmark logs and regenerate the structured CSV outputs.\n"
            "\n"
            "The parser reads benchmark*.log files produced by the benchmark drivers\n"
            "under log/bench, extracts the recorded metadata and metrics, and writes\n"
            "the corresponding CSV artifacts back to the output paths stored in the\n"
            "log itself.\n"
            "\n"
            "Generated outputs:\n"
            "  Chimera logs: benchmark_results_*.csv and pareto_frontier_*.csv\n"
            "  ColBERT logs: benchmark_results.csv, pareto_frontier.csv, and\n"
            "                search_py_output.csv when the log records that path"
        ),
        epilog=(
            "Examples:\n"
            "  Parse one Chimera log file:\n"
            "    python experiment/common/parse_bench_logs.py \\\n"
            "      log/bench/gpu_search_v4/lotte/benchmark_20260418T145245Z_job55741087.log\n"
            "\n"
            "  Parse every Chimera benchmark log under a directory:\n"
            "    python experiment/common/parse_bench_logs.py --mode chimera log/bench/gpu_search_v4\n"
            "\n"
            "  Parse every ColBERT benchmark log under a directory:\n"
            "    python experiment/common/parse_bench_logs.py --mode colbert log/bench/colbert\n"
            "\n"
            "  Parse multiple roots at once:\n"
            "    python experiment/common/parse_bench_logs.py log/bench/gpu_search_v4 log/bench/colbert\n"
            "\n"
            "  Replay logs from another checkout into the current repo layout:\n"
            "    python experiment/common/parse_bench_logs.py --repo-root /scratch3/workspace/.../Chimera \\\n"
            "      /path/to/copied/log/bench"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "chimera", "gpu-search", "colbert"),
        default="auto",
        help=(
            "Benchmark family to parse. Use 'auto' to infer the format from the log "
            "contents and [driver] metadata. Use 'chimera', legacy 'gpu-search', or 'colbert' to force one "
            "parser when you already know the log type or want stricter validation."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Override the repo root recorded in the log when resolving output paths. "
            "This is mainly for replaying logs that were produced from a different "
            "checkout or filesystem location. Only paths that originally lived under "
            "the logged repo_root are remapped; external absolute paths are left "
            "unchanged."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help=(
            "One or more inputs to parse. Each PATH may be either: "
            "(1) a single benchmark*.log file, or (2) a directory, in which case the "
            "script recursively discovers every matching benchmark*.log beneath it. "
            "You can mix files and directories in one command."
        ),
    )
    return parser.parse_args()


def parse_driver_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in lines:
        match = DRIVER_RE.match(raw_line.strip())
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


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


def require_metadata(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "")
    if not value:
        raise ParseError(f"missing [driver] {key}=... in log")
    return value


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


def write_simple_csv(path: Path, header: list[str], rows: list[tuple[str, ...]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
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


def parse_gpu_search_bench_log(log_file: Path, repo_root_override: Path | None) -> ParsedOutputs:
    lines = log_file.read_text().splitlines()
    metadata = parse_driver_metadata(lines)

    implementation = require_metadata(metadata, "implementation")
    dataset = require_metadata(metadata, "dataset")
    index_source = metadata.get("index_source", "nfs")
    index_path = metadata.get("active_index_dir") or metadata.get("source_index_dir")
    if not index_path:
        raise ParseError(
            f"missing [driver] active_index_dir=... or [driver] source_index_dir=... in log"
        )
    k = require_metadata(metadata, "k")
    warmup = require_metadata(metadata, "warmup")
    results_csv = remap_repo_path(require_metadata(metadata, "results_csv"), metadata, repo_root_override)
    pareto_csv = remap_repo_path(require_metadata(metadata, "pareto_csv"), metadata, repo_root_override)

    rows: list[dict[str, str]] = []
    current_config: dict[str, str] | None = None
    current_metrics: dict[str, str] = {}
    gpu_mem: dict[str, str] = {
        "gpu_mem_current_mib": "",
        "gpu_mem_peak_mib": "",
        "gpu_mem_total_mib": "",
        "gpu_mem_samples": "",
        "gpu_mem_peak_label": "",
    }

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = GPU_MEM_RE.match(line)
        if match:
            gpu_mem = {
                "gpu_mem_current_mib": match.group("current"),
                "gpu_mem_peak_mib": match.group("peak"),
                "gpu_mem_total_mib": match.group("total"),
                "gpu_mem_samples": match.group("samples"),
                "gpu_mem_peak_label": match.group("peak_label"),
            }
            continue

        match = GPU_CONFIG_RE.match(line)
        if match:
            if current_config is not None:
                raise ParseError(
                    f"incomplete Chimera metric block before next config in {log_file}"
                )
            current_config = match.groupdict()
            current_metrics = {}
            continue

        if current_config is None:
            continue

        match = GPU_TIMING_RE.match(line)
        if match:
            if match.group("label") != current_config["label"]:
                raise ParseError(
                    f"timing label mismatch: expected {current_config['label']}, got {match.group('label')}"
                )
            current_metrics["search_seconds"] = match.group("search_seconds")
            current_metrics["queries"] = match.group("queries")
            continue

        match = GPU_END_TO_END_RE.match(line)
        if match:
            current_metrics["end_to_end_ms"] = match.group(1)
            continue

        match = GPU_THROUGHPUT_RE.match(line)
        if match:
            current_metrics["qps"] = match.group(1)
            continue

        match = GPU_AVG_LATENCY_RE.match(line)
        if match:
            current_metrics["avg_latency_ms"] = match.group(1)
            continue

        if line.startswith("[SEARCH] Query latency distribution (ms): "):
            latency_fields = parse_key_value_fields(
                line.split(": ", 1)[1]
            )
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

        match = GPU_RECALL_RE.match(line)
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
                    f"incomplete Chimera metric block for label={current_config['label']}: missing {', '.join(missing)}"
                )

            rows.append(
                {
                    "implementation": implementation,
                    "dataset": dataset,
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
                    **gpu_mem,
                }
            )
            current_config = None
            current_metrics = {}

    if current_config is not None:
        raise ParseError(f"incomplete trailing Chimera metric block in {log_file}")

    if not rows:
        raise ParseError(f"no Chimera benchmark rows found in {log_file}")

    if any(value for value in gpu_mem.values()):
        for row in rows:
            row.update(gpu_mem)

    expected_count = metadata.get("config_count")
    if expected_count and len(rows) != int(expected_count):
        raise ParseError(
            f"parsed {len(rows)} Chimera rows from {log_file}, expected {expected_count}"
        )

    write_dict_csv(results_csv, GPU_RESULTS_HEADER, rows)
    write_dict_csv(
        pareto_csv,
        GPU_RESULTS_HEADER,
        compute_pareto_rows(rows, qps_key="qps", recall_key="recall"),
    )
    return ParsedOutputs(
        mode="chimera",
        log_file=log_file,
        results_csv=results_csv,
        pareto_csv=pareto_csv,
        row_count=len(rows),
    )


def parse_colbert_config_file(config_file: Path) -> list[tuple[str, str, str]]:
    configs: list[tuple[str, str, str]] = []
    with config_file.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            if not label or label == "label" or label.startswith("#"):
                continue
            if len(row) < 3:
                raise ParseError(f"malformed ColBERT config row in {config_file}: {row}")
            ncells = row[1].strip()
            ndocs = row[2].strip()
            configs.append((label, ncells, ndocs))
    if not configs:
        raise ParseError(f"no ColBERT configs found in {config_file}")
    return configs


def parse_colbert_log(log_file: Path, repo_root_override: Path | None) -> ParsedOutputs:
    lines = log_file.read_text().splitlines()
    metadata = parse_driver_metadata(lines)

    implementation = require_metadata(metadata, "implementation")
    dataset = require_metadata(metadata, "dataset")
    index_source = require_metadata(metadata, "index_source")
    index_root = require_metadata(metadata, "active_root_path")
    index_path = require_metadata(metadata, "active_index_dir")
    query_file = require_metadata(metadata, "query_file")
    gt_file = require_metadata(metadata, "gt_file")
    k = require_metadata(metadata, "k")
    num_runs = metadata.get("num_runs", "3")
    results_csv = remap_repo_path(require_metadata(metadata, "results_csv"), metadata, repo_root_override)
    pareto_csv = remap_repo_path(require_metadata(metadata, "pareto_csv"), metadata, repo_root_override)
    search_py_output_csv_raw = metadata.get("search_py_output_csv", "")
    search_py_output_csv = (
        remap_repo_path(search_py_output_csv_raw, metadata, repo_root_override)
        if search_py_output_csv_raw
        else None
    )

    configs: list[tuple[str, str, str]] = []
    metrics_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    query_count = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = COLBERT_CONFIG_RE.match(line)
        if match:
            configs.append((match.group("label"), match.group("ncells"), match.group("ndocs")))
            continue

        match = COLBERT_QUERY_COUNT_RE.match(line)
        if match:
            query_count = match.group(1)
            continue

        match = COLBERT_RESULT_RE.match(line)
        if match:
            metrics_by_pair[(match.group("ncells"), match.group("ndocs"))] = (
                match.group("avg_time"),
                match.group("recall"),
            )

    if not configs:
        config_file = remap_repo_path(require_metadata(metadata, "config_file"), metadata, repo_root_override)
        configs = parse_colbert_config_file(config_file)

    if not query_count:
        raise ParseError(f"failed to parse ColBERT query count from {log_file}")

    rows: list[dict[str, str]] = []
    search_output_rows: list[tuple[str, str]] = []
    for label, ncells, ndocs in configs:
        pair = (ncells, ndocs)
        if pair not in metrics_by_pair:
            raise ParseError(f"missing ColBERT metrics for ncells={ncells} ndocs={ndocs}")
        avg_time, recall = metrics_by_pair[pair]
        avg_time_value = float(avg_time)
        qps = f"{(1.0 / avg_time_value) if avg_time_value > 0.0 else 0.0:.6f}"

        rows.append(
            {
                "implementation": implementation,
                "dataset": dataset,
                "label": label,
                "ncells": ncells,
                "ndocs": ndocs,
                "index_source": index_source,
                "index_root": index_root,
                "index_path": index_path,
                "query_file": query_file,
                "gt_file": gt_file,
                "queries": query_count,
                "k": k,
                "num_runs": num_runs,
                "avg_time_s": avg_time,
                "qps": qps,
                "recall": recall,
            }
        )
        search_output_rows.append((f"{avg_time_value:.4f}", f"{float(recall):.4f}"))

    expected_count = metadata.get("config_count")
    if expected_count and len(rows) != int(expected_count):
        raise ParseError(
            f"parsed {len(rows)} ColBERT rows from {log_file}, expected {expected_count}"
        )

    write_dict_csv(results_csv, COLBERT_RESULTS_HEADER, rows)
    write_dict_csv(
        pareto_csv,
        COLBERT_RESULTS_HEADER,
        compute_pareto_rows(rows, qps_key="qps", recall_key="recall"),
    )
    if search_py_output_csv is not None:
        write_simple_csv(search_py_output_csv, COLBERT_SEARCH_OUTPUT_HEADER, search_output_rows)

    return ParsedOutputs(
        mode="colbert",
        log_file=log_file,
        results_csv=results_csv,
        pareto_csv=pareto_csv,
        row_count=len(rows),
    )


def infer_mode(lines: list[str], metadata: dict[str, str]) -> str:
    implementation = metadata.get("implementation", "")
    if implementation == "colbert":
        return "colbert"
    if implementation.startswith("gpu_search_"):
        return "chimera"

    for raw_line in lines:
        line = raw_line.strip()
        if GPU_CONFIG_RE.match(line) or GPU_TIMING_RE.match(line):
            return "chimera"
        if COLBERT_RESULT_RE.match(line):
            return "colbert"

    raise ParseError("unable to auto-detect benchmark family")


def parse_log(log_file: Path, mode: str, repo_root_override: Path | None) -> ParsedOutputs:
    lines = log_file.read_text().splitlines()
    metadata = parse_driver_metadata(lines)
    effective_mode = infer_mode(lines, metadata) if mode == "auto" else mode

    if effective_mode in {"chimera", "gpu-search"}:
        return parse_gpu_search_bench_log(log_file, repo_root_override)
    if effective_mode == "colbert":
        return parse_colbert_log(log_file, repo_root_override)
    raise ParseError(f"unsupported mode: {effective_mode}")


def discover_log_files(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file():
            discovered.append(path)
            continue
        if path.is_dir():
            discovered.extend(sorted(path.rglob("benchmark*.log")))
            continue
        raise ParseError(f"path not found: {path}")

    unique_logs = sorted(dict.fromkeys(discovered))
    if not unique_logs:
        raise ParseError("no benchmark*.log files found")
    return unique_logs


def main() -> int:
    args = parse_args()

    try:
        log_files = discover_log_files(args.paths)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for log_file in log_files:
        try:
            parsed = parse_log(log_file, mode=args.mode, repo_root_override=args.repo_root)
        except Exception as exc:
            failures += 1
            print(f"error: {log_file}: {exc}", file=sys.stderr)
            continue

        print(
            f"parsed {parsed.mode} {log_file} -> {parsed.results_csv} "
            f"({parsed.row_count} rows)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
