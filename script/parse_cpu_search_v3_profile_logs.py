#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DRIVER_RE = re.compile(r"^\[driver\] ([A-Za-z0-9_]+)=(.*)$")
PROFILE_RE = re.compile(r"^\[PROFILE_AVG\] (.*)$")

OUTPUT_HEADER = [
    "implementation",
    "dataset",
    "batch_id",
    "selection_label",
    "target_recall",
    "k",
    "queries",
    "avg_end_to_end_ms",
    "avg_profiled_total_ms",
    "avg_accounted_ms",
    "avg_unaccounted_ms",
    "avg_query_setup_ms",
    "avg_stage1_ms",
    "avg_stage1_probe_ms",
    "avg_stage1_prepare_ms",
    "avg_stage1_scan_ms",
    "avg_stage1_reduce_ms",
    "avg_stage1_cleanup_ms",
    "avg_stage2_ms",
    "avg_stage2_lut_ms",
    "avg_stage2_score_docs_ms",
    "avg_stage2_select_topk_ms",
    "avg_stage2_materialize_ms",
    "avg_stage3_ms",
    "avg_stage3_prepare_ms",
    "avg_stage3_score_docs_ms",
    "avg_stage3_select_topk_ms",
    "avg_stage12_ms",
    "stage1_pct_end_to_end",
    "stage2_pct_end_to_end",
    "stage3_pct_end_to_end",
    "stage12_pct_end_to_end",
    "stage1_pct_profiled_total",
    "stage2_pct_profiled_total",
    "stage3_pct_profiled_total",
    "stage12_pct_profiled_total",
    "query_setup_pct_end_to_end",
    "log_file",
]


class ParseError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse cpu_search_v3 aggregate profiling logs into a CSV summary."
        )
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Only parse profile logs for this batch id.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repo root override for output paths.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to profiling/cpu_search_v3_profile/profile_summary_<batch>.csv.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more log files or directories containing profile_*.log files.",
    )
    return parser.parse_args()


def parse_driver_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in lines:
        match = DRIVER_RE.match(raw_line.strip())
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


def parse_key_value_fields(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in payload.split():
        if "=" not in chunk:
            raise ParseError(f"unable to parse key=value field from: {payload}")
        key, value = chunk.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def discover_log_files(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("profile_*.log")))
        else:
            raise ParseError(f"path does not exist: {path}")
    return sorted(set(discovered))


def require_metadata(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "")
    if not value:
        raise ParseError(f"missing [driver] {key}=... in log")
    return value


def remap_repo_path(path: Path, repo_root_override: Path | None) -> Path:
    if repo_root_override is None:
        return path
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        return path
    return repo_root_override / relative


def parse_profile_log(log_file: Path, batch_id: str) -> dict[str, str] | None:
    lines = log_file.read_text().splitlines()
    metadata = parse_driver_metadata(lines)
    if metadata.get("batch_id") != batch_id:
        return None

    summary_fields: dict[str, str] = {}
    stage_fields: dict[str, dict[str, str]] = {}
    for raw_line in lines:
        match = PROFILE_RE.match(raw_line.strip())
        if not match:
            continue
        fields = parse_key_value_fields(match.group(1))
        stage_name = fields.get("stage")
        if stage_name is None:
            summary_fields = fields
        else:
            stage_fields[stage_name] = fields

    if not summary_fields:
        raise ParseError(f"missing [PROFILE_AVG] summary line in {log_file}")
    for required_stage in ("query_setup", "stage1", "stage2", "stage3", "stage12"):
        if required_stage not in stage_fields:
            raise ParseError(f"missing [PROFILE_AVG] stage={required_stage} line in {log_file}")

    return {
        "implementation": require_metadata(metadata, "implementation"),
        "dataset": require_metadata(metadata, "dataset"),
        "batch_id": require_metadata(metadata, "batch_id"),
        "selection_label": require_metadata(metadata, "selection_label"),
        "target_recall": require_metadata(metadata, "target_recall"),
        "k": require_metadata(metadata, "k"),
        "queries": summary_fields["queries"],
        "avg_end_to_end_ms": summary_fields["end_to_end_ms"],
        "avg_profiled_total_ms": summary_fields["profiled_total_ms"],
        "avg_accounted_ms": summary_fields["accounted_ms"],
        "avg_unaccounted_ms": summary_fields["unaccounted_ms"],
        "avg_query_setup_ms": stage_fields["query_setup"]["total_ms"],
        "avg_stage1_ms": stage_fields["stage1"]["total_ms"],
        "avg_stage1_probe_ms": stage_fields["stage1"]["probe_ms"],
        "avg_stage1_prepare_ms": stage_fields["stage1"]["prepare_ms"],
        "avg_stage1_scan_ms": stage_fields["stage1"]["scan_ms"],
        "avg_stage1_reduce_ms": stage_fields["stage1"]["reduce_ms"],
        "avg_stage1_cleanup_ms": stage_fields["stage1"]["cleanup_ms"],
        "avg_stage2_ms": stage_fields["stage2"]["total_ms"],
        "avg_stage2_lut_ms": stage_fields["stage2"]["lut_ms"],
        "avg_stage2_score_docs_ms": stage_fields["stage2"]["score_docs_ms"],
        "avg_stage2_select_topk_ms": stage_fields["stage2"]["select_topk_ms"],
        "avg_stage2_materialize_ms": stage_fields["stage2"]["materialize_ms"],
        "avg_stage3_ms": stage_fields["stage3"]["total_ms"],
        "avg_stage3_prepare_ms": stage_fields["stage3"]["prepare_ms"],
        "avg_stage3_score_docs_ms": stage_fields["stage3"]["score_docs_ms"],
        "avg_stage3_select_topk_ms": stage_fields["stage3"]["select_topk_ms"],
        "avg_stage12_ms": stage_fields["stage12"]["total_ms"],
        "stage1_pct_end_to_end": stage_fields["stage1"]["pct_end_to_end"],
        "stage2_pct_end_to_end": stage_fields["stage2"]["pct_end_to_end"],
        "stage3_pct_end_to_end": stage_fields["stage3"]["pct_end_to_end"],
        "stage12_pct_end_to_end": stage_fields["stage12"]["pct_end_to_end"],
        "stage1_pct_profiled_total": stage_fields["stage1"]["pct_profiled_total"],
        "stage2_pct_profiled_total": stage_fields["stage2"]["pct_profiled_total"],
        "stage3_pct_profiled_total": stage_fields["stage3"]["pct_profiled_total"],
        "stage12_pct_profiled_total": stage_fields["stage12"]["pct_profiled_total"],
        "query_setup_pct_end_to_end": stage_fields["query_setup"]["pct_end_to_end"],
        "log_file": str(log_file),
    }


def main() -> int:
    args = parse_args()

    try:
        log_files = discover_log_files(args.paths)
        if not log_files:
            raise ParseError("no profile_*.log files found")

        rows: list[dict[str, str]] = []
        for log_file in log_files:
            parsed = parse_profile_log(log_file, args.batch_id)
            if parsed is not None:
                rows.append(parsed)

        if not rows:
            raise ParseError(f"no logs matched batch id {args.batch_id}")

        rows.sort(key=lambda row: (int(row["k"]), row["dataset"]))
        output_csv = args.output_csv
        if output_csv is None:
            output_csv = Path("profiling") / "cpu_search_v3_profile" / f"profile_summary_{args.batch_id}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADER)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[ok] rows={len(rows)} output_csv={output_csv}")

    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
