#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATASETS = ("lotte", "msmarco", "hotpot")
LEGACY_CHIMERA_INDEX_PREFIX = "gpu_" "mvr"
DATASET_CONFIG_CANDIDATES = {
    "lotte": ("config/latte_config.csv", "profiling/lotte_config.csv", "profiling/latte_config.csv"),
    "msmarco": ("config/msmarco_config.csv", "profiling/msmarco_config.csv"),
    "hotpot": ("config/hotpot_config.csv", "profiling/hotpot_config.csv"),
}


@dataclass(frozen=True)
class SystemSpec:
    name: str
    version: str
    binary: str
    label: str


@dataclass(frozen=True)
class RecallTargetConfig:
    dataset: str
    target_recall: str
    label: str
    nprobe: int
    k_rank_cluster: int
    k_rank_all_tokens: int
    itopk_size: int
    overlap_chunks: int

    def as_row(self, label: str | None = None, overlap_chunks: int | None = None) -> dict[str, str]:
        return {
            "label": label or self.label,
            "nprobe": str(self.nprobe),
            "k_rank_cluster": str(self.k_rank_cluster),
            "k_rank_all_tokens": str(self.k_rank_all_tokens),
            "itopk_size": str(self.itopk_size),
            "overlap_chunks": str(self.overlap_chunks if overlap_chunks is None else overlap_chunks),
        }


SYSTEMS = {
    "gpu_search": SystemSpec("gpu_search", "gpu_search", "gpu_search", "chimera"),
    "gpu_search_nosum": SystemSpec(
        "gpu_search_nosum",
        "gpu_search_nosum",
        "gpu_search_nosum",
        "chimera_nosum",
    ),
    "gpu_search_nolut": SystemSpec(
        "gpu_search_nolut",
        "gpu_search_nolut",
        "gpu_search_nolut",
        "chimera_nolut",
    ),
    "gpu_search_nolut_nosum": SystemSpec(
        "gpu_search_nolut_nosum",
        "gpu_search_nolut_nosum",
        "gpu_search_nolut_nosum",
        "chimera_nolut_nosum",
    ),
}

R095_LABELS = {
    "lotte": "g2",
    "msmarco": "d1",
    "hotpot": "k1",
}


RECALL_TARGET_CONFIGS = [
    RecallTargetConfig("lotte", "0.90", "c1", 64, 2400, 240, 128, 5),
    RecallTargetConfig("lotte", "0.95", "f1", 128, 4000, 400, 192, 5),
    RecallTargetConfig("msmarco", "0.90", "b2", 64, 1800, 180, 96, 4),
    RecallTargetConfig("msmarco", "0.95", "q2", 96, 3200, 320, 160, 6),
    RecallTargetConfig("hotpot", "0.90", "q4", 160, 5000, 500, 224, 6),
    RecallTargetConfig("hotpot", "0.95", "q9", 384, 28000, 1800, 448, 5),
]


GPU_DIRECT_PATTERNS = {
    "end_to_end_ms": re.compile(r"^\[SEARCH\] End-to-end measured time: ([0-9.]+) ms$", re.MULTILINE),
    "avg_latency_ms": re.compile(r"^\[SEARCH\] Average latency per query: ([0-9.]+) ms$", re.MULTILINE),
    "qps": re.compile(r"^\[SEARCH\] Throughput: ([0-9.]+) qps$", re.MULTILINE),
    "gpu_mem_current_mib": re.compile(r"^\[GPU_MEM\] current=([0-9.]+) MiB,", re.MULTILINE),
    "gpu_mem_peak_mib": re.compile(r"^\[GPU_MEM\].* peak=([0-9.]+) MiB,", re.MULTILINE),
    "effective_concurrent_queries": re.compile(r"effective_concurrent_queries=(\d+)"),
    "stage3_threads": re.compile(r"stage3_threads=(\d+)"),
}
GPU_RECALL_ANY_RE = re.compile(r"^Recall@(\d+): ([0-9.]+)$", re.MULTILINE)
GPU_LATENCY_RE = re.compile(r"^\[SEARCH\] Query latency distribution \(ms\): (.+)$", re.MULTILINE)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def quote_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing directory: {path}")


def chimera_index_dir(repo_root: Path, dataset: str) -> Path:
    dataset_root = repo_root / "dataset" / dataset
    candidates = [
        dataset_root / "gpu_search_2m",
        dataset_root / "gpu_search_1m",
        dataset_root / "gpu_search",
        dataset_root / "gpu_search_index",
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_2m",
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_1m",
        dataset_root / LEGACY_CHIMERA_INDEX_PREFIX,
        dataset_root / f"{LEGACY_CHIMERA_INDEX_PREFIX}_index",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "doclens.bin").is_file():
            return candidate
    return candidates[0]


def dataset_config_path(repo_root: Path, dataset: str) -> Path:
    for rel in DATASET_CONFIG_CANDIDATES[dataset]:
        path = repo_root / rel
        if path.is_file():
            return path
    candidates = ", ".join(DATASET_CONFIG_CANDIDATES[dataset])
    raise FileNotFoundError(f"missing config for dataset={dataset}; tried {candidates}")


def validate_dataset_inputs(repo_root: Path, dataset: str) -> None:
    require_file(repo_root / "dataset" / dataset / "raw" / "query.bin")
    require_file(repo_root / "dataset" / dataset / "raw" / "gt.tsv")
    require_dir(chimera_index_dir(repo_root, dataset))


def normalize_datasets(values: Iterable[str]) -> list[str]:
    datasets = list(values)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return datasets


def normalize_systems(values: Iterable[str]) -> list[SystemSpec]:
    unknown = sorted(set(values) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"unknown system(s): {', '.join(unknown)}")
    return [SYSTEMS[value] for value in values]


def normalize_targets(values: Iterable[str]) -> list[str]:
    targets = [f"{float(value):.2f}" for value in values]
    known = {cfg.target_recall for cfg in RECALL_TARGET_CONFIGS}
    unknown = sorted(set(targets) - known)
    if unknown:
        raise ValueError(f"unknown target recall(s): {', '.join(unknown)}")
    return targets


def selected_recall_target_configs(
    datasets: Iterable[str],
    targets: Iterable[str],
) -> list[RecallTargetConfig]:
    dataset_set = set(normalize_datasets(datasets))
    target_set = set(normalize_targets(targets))
    return [
        cfg for cfg in RECALL_TARGET_CONFIGS
        if cfg.dataset in dataset_set and cfg.target_recall in target_set
    ]


def read_config_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label or label.startswith("#"):
                continue
            rows.append({key: (value or "").strip() for key, value in row.items()})
    if not rows:
        raise ValueError(f"no runtime configs found in {path}")
    return rows


def find_config_row(path: Path, label: str) -> dict[str, str]:
    for row in read_config_rows(path):
        if row["label"] == label:
            return row
    raise ValueError(f"config label={label!r} not found in {path}")


def write_single_config(path: Path, source_row: dict[str, str], label: str | None = None) -> None:
    write_config_rows(path, [{**source_row, "label": label or source_row["label"]}])


def write_chunk_config(path: Path, source_row: dict[str, str], chunks: Iterable[int]) -> None:
    rows = []
    for chunk in chunks:
        row = dict(source_row)
        row["label"] = f"c{chunk}"
        row["overlap_chunks"] = str(chunk)
        rows.append(row)
    write_config_rows(path, rows)


def write_config_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "nprobe",
        "k_rank_cluster",
        "k_rank_all_tokens",
        "itopk_size",
        "overlap_chunks",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def build_systems(repo_root: Path, build_dir: Path, systems: list[SystemSpec], dry_run: bool) -> None:
    targets = sorted({system.binary for system in systems})
    cmd = ["cmake", "--build", str(build_dir), "--target", *targets, "-j", str(os.cpu_count() or 8)]
    print(f"[build] {quote_cmd(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=repo_root, check=True)


def ensure_binaries(build_dir: Path, systems: list[SystemSpec], dry_run: bool) -> None:
    if dry_run:
        return
    missing = [str(build_dir / system.binary) for system in systems if not os.access(build_dir / system.binary, os.X_OK)]
    if missing:
        raise FileNotFoundError("missing executable benchmark binary/binaries:\n" + "\n".join(missing))


def bench_command(
    repo_root: Path,
    build_dir: Path,
    output_run_dir: Path,
    log_run_dir: Path,
    dataset: str,
    system: SystemSpec,
    config_file: Path,
    k: int,
    nq: int,
    warmup: int,
    dry_run: bool,
    concurrent_queries: int | None = None,
    stage3_threads: int | None = None,
) -> list[str]:
    cmd = [
        str(repo_root / "experiment" / "chimera" / "bench_gpu_search.sh"),
        "--dataset", dataset,
        "--version", system.version,
        "--implementation-label", system.label,
        "--binary", str(build_dir / system.binary),
        "--config-file", str(config_file),
        "--output-dir", str(output_run_dir),
        "--log-dir", str(log_run_dir),
        "--build-dir", str(build_dir),
        "--k", str(k),
        "--nq", str(nq),
        "--warmup", str(warmup),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if concurrent_queries is not None:
        cmd.extend(["--concurrent-queries", str(concurrent_queries)])
    if stage3_threads is not None:
        cmd.extend(["--stage3-threads", str(stage3_threads)])
    return cmd


def run_bench(
    repo_root: Path,
    build_dir: Path,
    output_run_dir: Path,
    log_run_dir: Path,
    dataset: str,
    system: SystemSpec,
    config_file: Path,
    k: int,
    nq: int,
    warmup: int,
    dry_run: bool,
    env: dict[str, str],
    concurrent_queries: int | None = None,
    stage3_threads: int | None = None,
) -> None:
    validate_dataset_inputs(repo_root, dataset)
    cmd = bench_command(
        repo_root,
        build_dir,
        output_run_dir,
        log_run_dir,
        dataset,
        system,
        config_file,
        k,
        nq,
        warmup,
        dry_run,
        concurrent_queries,
        stage3_threads,
    )
    print(f"[run] dataset={dataset} system={system.label} config={config_file}", flush=True)
    print(f"[cmd] {quote_cmd(cmd)}", flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def parse_logs(repo_root: Path, log_run_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    parser = repo_root / "experiment" / "common" / "parse_bench_logs.py"
    cmd = [sys.executable, str(parser), "--mode", "chimera", "--repo-root", str(repo_root), str(log_run_dir)]
    print(f"[parse] {quote_cmd(cmd)}", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def collect_result_rows(output_run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for csv_path in sorted(output_run_dir.rglob("benchmark_results_*.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            file_rows = list(reader)
        if not file_rows:
            continue
        dataset = file_rows[0].get("dataset", "")
        implementation = file_rows[0].get("implementation", "")
        key = (dataset, implementation)
        counters[key] += 1
        run_index = counters[key]
        for row in file_rows:
            row = dict(row)
            row["system"] = row.get("implementation", "")
            row["run_index"] = str(run_index)
            row["source_csv"] = str(csv_path)
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = preferred + extras
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def _mean(values: list[float]) -> str:
    return "" if not values else f"{sum(values) / len(values):.6f}"


def _std(values: list[float]) -> str:
    if len(values) < 2:
        return "0.000000" if values else ""
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{math.sqrt(var):.6f}"


def _max(values: list[float]) -> str:
    return "" if not values else f"{max(values):.6f}"


def summarize_rows(rows: list[dict[str, str]], group_fields: list[str]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary = []
    metric_fields = [
        "recall",
        "recall_at_10",
        "recall_at_100",
        "qps",
        "avg_latency_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "stddev_ms",
        "gpu_mem_peak_mib",
    ]
    for key, group in sorted(groups.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["runs"] = str(len({row.get("run_index", "") for row in group}))
        out["rows"] = str(len(group))
        for metric in metric_fields:
            values = _float_values(group, metric)
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        out["gpu_mem_peak_mib_max"] = _max(_float_values(group, "gpu_mem_peak_mib"))
        summary.append(out)
    return summary


def write_summary(path: Path, rows: list[dict[str, str]], group_fields: list[str]) -> None:
    summary = summarize_rows(rows, group_fields)
    preferred = group_fields + [
        "runs",
        "rows",
        "recall_mean",
        "recall_std",
        "recall_at_10_mean",
        "recall_at_10_std",
        "recall_at_100_mean",
        "recall_at_100_std",
        "qps_mean",
        "qps_std",
        "avg_latency_ms_mean",
        "avg_latency_ms_std",
        "p50_ms_mean",
        "p90_ms_mean",
        "p95_ms_mean",
        "p99_ms_mean",
        "max_ms_mean",
        "stddev_ms_mean",
        "gpu_mem_peak_mib_mean",
        "gpu_mem_peak_mib_std",
        "gpu_mem_peak_mib_max",
    ]
    write_rows(path, summary, preferred)


def write_best_at_or_above_recall(path: Path, rows: list[dict[str, str]], target_recall: float) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", ""), row.get("system", ""))].append(row)

    out_rows = []
    for (dataset, system), group in sorted(groups.items()):
        valid = []
        fallback = []
        for row in group:
            try:
                qps = float(row.get("qps", ""))
                recall = float(row.get("recall", ""))
            except ValueError:
                continue
            fallback.append((qps, recall, row))
            if recall >= target_recall:
                valid.append((qps, recall, row))
        candidates = valid or fallback
        if not candidates:
            continue
        _qps, _recall, row = max(candidates, key=lambda item: item[0])
        out = {
            "dataset": dataset,
            "system": system,
            "target_recall": f"{target_recall:.2f}",
            "met_target": str(bool(valid)),
            "label": row.get("label", ""),
            "recall": row.get("recall", ""),
            "qps": row.get("qps", ""),
            "avg_latency_ms": row.get("avg_latency_ms", ""),
            "gpu_mem_peak_mib": row.get("gpu_mem_peak_mib", ""),
            "source_csv": row.get("source_csv", ""),
        }
        out_rows.append(out)
    write_rows(
        path,
        out_rows,
        [
            "dataset",
            "system",
            "target_recall",
            "met_target",
            "label",
            "recall",
            "qps",
            "avg_latency_ms",
            "gpu_mem_peak_mib",
            "source_csv",
        ],
    )


def common_env(cuda_visible_devices: str | None, omp_num_threads: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if omp_num_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_num_threads)
    return env


def reset_run_dir(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_key_value_payload(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in payload.split(","):
        key, value = item.strip().split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_gpu_search_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    metrics: dict[str, str] = {}
    for name, pattern in GPU_DIRECT_PATTERNS.items():
        match = pattern.search(text)
        metrics[name] = match.group(1) if match else ""

    for recall_k, recall_value in GPU_RECALL_ANY_RE.findall(text):
        metrics[f"recall_at_{recall_k}"] = recall_value
        metrics["recall"] = recall_value

    if "recall_at_100" in metrics:
        metrics["recall"] = metrics["recall_at_100"]

    latency_match = GPU_LATENCY_RE.search(text)
    if latency_match:
        latency = parse_key_value_payload(latency_match.group(1))
        metrics.update({
            "min_ms": latency.get("min", ""),
            "p50_ms": latency.get("p50", ""),
            "p90_ms": latency.get("p90", ""),
            "p95_ms": latency.get("p95", ""),
            "p99_ms": latency.get("p99", ""),
            "max_ms": latency.get("max", ""),
            "stddev_ms": latency.get("stddev", ""),
        })
    return metrics


def gpu_search_direct_command(
    repo_root: Path,
    build_dir: Path,
    system: SystemSpec,
    dataset: str,
    config: dict[str, str],
    k: int,
    nq: int,
    warmup: int,
    concurrent_queries: int | None = None,
    stage3_threads: int | None = None,
) -> list[str]:
    cmd = [
        str(build_dir / system.binary),
        "--query", str(repo_root / "dataset" / dataset / "raw" / "query.bin"),
        "--gt", str(repo_root / "dataset" / dataset / "raw" / "gt.tsv"),
        "--index", str(chimera_index_dir(repo_root, dataset)),
        "--k", str(k),
        "--nq", str(nq),
        "--warmup", str(warmup),
        "--nprobe", config["nprobe"],
        "--k-rank-cluster", config["k_rank_cluster"],
        "--k-rank-all-tokens", config["k_rank_all_tokens"],
        "--itopk-size", config["itopk_size"],
        "--overlap-chunks", config["overlap_chunks"],
    ]
    if concurrent_queries is not None:
        cmd.extend(["--concurrent-queries", str(concurrent_queries)])
    if stage3_threads is not None and stage3_threads > 0:
        cmd.extend(["--stage3-threads", str(stage3_threads)])
    return cmd


def run_gpu_search_direct(
    repo_root: Path,
    build_dir: Path,
    log_path: Path,
    system: SystemSpec,
    dataset: str,
    config: dict[str, str],
    k: int,
    nq: int,
    warmup: int,
    run_index: int,
    dry_run: bool,
    env: dict[str, str],
    concurrent_queries: int | None = None,
    stage3_threads: int | None = None,
    extra_row: dict[str, str] | None = None,
) -> dict[str, str]:
    validate_dataset_inputs(repo_root, dataset)
    cmd = gpu_search_direct_command(
        repo_root,
        build_dir,
        system,
        dataset,
        config,
        k,
        nq,
        warmup,
        concurrent_queries,
        stage3_threads,
    )
    row = {
        "system": system.label,
        "dataset": dataset,
        "label": config["label"],
        "run_index": str(run_index),
        "nprobe": config["nprobe"],
        "k_rank_cluster": config["k_rank_cluster"],
        "k_rank_all_tokens": config["k_rank_all_tokens"],
        "itopk_size": config["itopk_size"],
        "overlap_chunks": config["overlap_chunks"],
        "k": str(k),
        "nq": str(nq),
        "warmup": str(warmup),
        "slots": "" if concurrent_queries is None else str(concurrent_queries),
        "binary": system.binary,
        "log": str(log_path),
        "command": quote_cmd(cmd),
    }
    if extra_row:
        row.update(extra_row)
    if dry_run:
        print(f"[dry-run] {quote_cmd(cmd)}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "[dry-run]\n"
            f"[driver] command={quote_cmd(cmd)}\n"
            f"[driver] dataset={dataset}\n"
            f"[driver] system={system.label}\n"
            f"[driver] config_label={config['label']}\n"
        )
        return row

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[run] dataset={dataset} system={system.label} label={config['label']} "
        f"k={k} slots={row['slots'] or 'default'} run={run_index}",
        flush=True,
    )
    with log_path.open("w") as handle:
        handle.write(f"[driver] command={quote_cmd(cmd)}\n")
        handle.write(f"[driver] dataset={dataset}\n")
        handle.write(f"[driver] system={system.label}\n")
        handle.write(f"[driver] config_label={config['label']}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    row["returncode"] = str(proc.returncode)
    if proc.returncode == 0:
        row.update(parse_gpu_search_log(log_path))
    else:
        row["error_tail"] = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
    return row
