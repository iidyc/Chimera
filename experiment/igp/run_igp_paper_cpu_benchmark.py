#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PAPER_PHI_PB_VALUES = [1, 2, 4, 8, 16, 32, 64]
PAPER_PHI_REF_MULTIPLIERS = [1, 2, 5, 10, 20, 30, 40, 60]
PAPER_NB = 8
PAPER_BS = 16
DEFAULT_QUERY_THREADS = max(1, os.cpu_count() or 1)
PAPER_VAL_QUERY_COUNT = 50
PAPER_TOPKS = [10, 100]


def parse_config_spec(spec: str) -> tuple[int, int]:
    normalized = spec.replace(",", ":")
    parts = normalized.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid config '{spec}', expected phi_pb:phi_ref"
        )
    try:
        phi_pb = int(parts[0])
        phi_ref = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid config '{spec}', expected integers"
        ) from exc
    if phi_pb <= 0 or phi_ref <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid config '{spec}', phi_pb and phi_ref must be positive"
        )
    return phi_pb, phi_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CPU IGP using the parameter-setting procedure from "
            "SIGIR'25 IGP Section 5.1: 50 validation queries, grid search over "
            "phi_pb and phi_ref, single-thread search."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["lotte", "msmarco", "hotpot"],
        choices=["lotte", "msmarco", "hotpot"],
    )
    parser.add_argument(
        "--topks",
        nargs="+",
        type=int,
        default=PAPER_TOPKS,
        choices=PAPER_TOPKS,
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=PAPER_VAL_QUERY_COUNT,
        help="Number of validation queries, matching the paper default of 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for validation/test split. The paper says random sample but does not specify a seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="profiling/igp_paper_cpu",
        help="Directory for benchmark JSON outputs.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Recompile the CPU IGP module before benchmarking.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_QUERY_THREADS,
        help="Number of CPU threads for IGP search. Default: all visible CPU cores.",
    )
    parser.add_argument(
        "--n-bit",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="Residual scalar quantization bits for the IGP index. Default: 4.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=parse_config_spec,
        help=(
            "Explicit retrieval configs as phi_pb:phi_ref. "
            "When provided, the script runs only this config set instead of the full paper grid."
        ),
    )
    parser.add_argument(
        "--query-split",
        choices=["val", "test", "all"],
        default="val",
        help=(
            "Query subset to run when --configs is provided. "
            "Default: val."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_igp_imports(repo: Path) -> None:
    igp_root = repo / "IGP"
    sys.path.insert(0, str(igp_root / "script" / "evaluation"))
    sys.path.insert(1, str(igp_root))


def configure_igp_env(repo: Path, dataset: str) -> None:
    os.environ.setdefault("IGP_DATA_MODE", "binary")
    os.environ.setdefault("CHIMERA_DATASET_ROOT", str(repo / "dataset"))
    os.environ.setdefault("GPU_" "MVR_DATASET_ROOT", str(repo / "dataset"))
    os.environ["IGP_LOCAL_DATA_ROOT"] = str(repo / "dataset" / dataset / "igp")


def load_raw_gt(raw_gt_path: Path) -> dict[int, list[tuple[int, int]]]:
    gt: dict[int, list[tuple[int, int]]] = {}
    with open(raw_gt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            arr = line.rstrip("\n").split("\t")
            if len(arr) < 3:
                raise ValueError(f"unexpected groundtruth row in {raw_gt_path}: {line!r}")
            qid = int(arr[0])
            doc_id = int(arr[1])
            rank = int(arr[2])
            gt.setdefault(qid, []).append((doc_id, rank))
    return gt


def metric_name(topk: int, end_to_end_metric_available: bool) -> str:
    if topk == 10 and end_to_end_metric_available:
        return "mrr@10"
    if topk == 100:
        return "recall@100"
    return f"recall@{topk}"


def compute_recall_at_k(
    est_ids: np.ndarray,
    query_ids: np.ndarray,
    gt: dict[int, list[tuple[int, int]]],
    topk: int,
) -> tuple[np.ndarray, float]:
    recall_values = np.zeros(len(query_ids), dtype=np.float64)
    for local_idx, qid in enumerate(query_ids.tolist()):
        gnd = gt.get(int(qid), [])
        gnd_ids = [doc_id for doc_id, rank in gnd if rank <= topk]
        if not gnd_ids:
            recall_values[local_idx] = 1.0
            continue
        est = est_ids[local_idx, :topk]
        recall_values[local_idx] = (
            len(set(int(x) for x in est.tolist()) & set(gnd_ids)) / len(gnd_ids)
        )
    return recall_values, float(np.mean(recall_values))


def load_igp_index_for_dataset(dataset: str, compile_module: bool, n_bit: int):
    repo = repo_root()
    configure_igp_env(repo, dataset)
    configure_igp_imports(repo)

    from script.data import dataset_io, util
    from script.evaluation import eval_igp
    import IGP

    if compile_module:
        util.compile_file(
            username="juelin",
            module_name="IGP",
            is_debug=False,
            move_path="evaluation",
        )

    item_n_vec_l = dataset_io.load_doclens(username="juelin", dataset=dataset).astype(np.uint32)
    n_item = int(item_n_vec_l.shape[0])
    n_vecs = int(np.sum(item_n_vec_l, dtype=np.uint64))
    vec_dim = dataset_io.embedding_dim(username="juelin", dataset=dataset)
    n_centroid = util.paper_n_centroid(n_vecs)

    constructor_insert_item = {
        "item_n_vec_l": item_n_vec_l.tolist(),
        "n_item": n_item,
        "vec_dim": vec_dim,
        "n_centroid": n_centroid,
        "n_bit": n_bit,
    }

    index_path = dataset_io.index_dir(dataset) / f"IGP-n_centroid_{n_centroid}-n_bit_{n_bit}"
    if not index_path.exists():
        raise FileNotFoundError(
            f"missing CPU IGP index for {dataset}: {index_path}. "
            "Build the paper-configured IGP index first."
        )

    index = eval_igp.load_index(str(index_path), module=IGP, constructor_insert_item=constructor_insert_item)
    query_l = dataset_io.load_query_embeddings(username="juelin", dataset=dataset)
    query_ids = dataset_io.load_query_ids(username="juelin", dataset=dataset, num_queries=len(query_l))
    raw_gt = load_raw_gt(repo / "dataset" / dataset / "raw" / "gt.tsv")

    return {
        "repo": repo,
        "dataset_io": dataset_io,
        "eval_igp": eval_igp,
        "index": index,
        "query_l": query_l,
        "query_ids": query_ids,
        "raw_gt": raw_gt,
        "n_centroid": n_centroid,
        "n_bit": n_bit,
        "index_path": str(index_path),
    }


def evaluate_config(
    eval_igp,
    index,
    query_l: np.ndarray,
    query_ids: np.ndarray,
    raw_gt: dict[int, list[tuple[int, int]]],
    local_indices: np.ndarray,
    topk: int,
    phi_pb: int,
    phi_ref: int,
    n_thread: int,
) -> dict:
    subset_queries = query_l[local_indices]
    subset_query_ids = query_ids[local_indices]
    retrieval_config = {
        "nprobe": phi_pb,
        "probe_topk": phi_ref,
        "n_thread": n_thread,
    }

    start = time.time()
    est_dist_l, est_id_l, retrieval_suffix, search_time_m, retrieval_time_ms_l = (
        eval_igp.approximate_solution_retrieval(
            index=index,
            retrieval_config=retrieval_config,
            query_l=subset_queries,
            topk=topk,
        )
    )
    wall_sec = time.time() - start

    recall_l, recall_mean = compute_recall_at_k(
        est_id_l, subset_query_ids, raw_gt, topk)

    avg_latency_ms = float(np.mean(retrieval_time_ms_l))
    qps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "phi_pb": phi_pb,
        "phi_ref": phi_ref,
        "retrieval_suffix": retrieval_suffix,
        "queries": int(len(local_indices)),
        "avg_latency_ms": avg_latency_ms,
        "qps": qps,
        "wall_sec": wall_sec,
        "recall_mean": recall_mean,
        "recall_p50": float(np.percentile(recall_l, 50)),
        "recall_p95": float(np.percentile(recall_l, 95)),
        "search_time_m": search_time_m,
    }


def choose_best(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda row: (
            row["recall_mean"],
            row["qps"],
            -row["avg_latency_ms"],
        ),
    )


def print_config_result(
    *,
    dataset: str,
    topk: int,
    phase: str,
    ordinal: int,
    total: int,
    row: dict,
) -> None:
    print(
        f"[{phase}] dataset={dataset} topk={topk} "
        f"config={ordinal}/{total} phi_pb={row['phi_pb']} phi_ref={row['phi_ref']} "
        f"qps={row['qps']:.3f} avg_latency_ms={row['avg_latency_ms']:.3f} "
        f"recall={row['recall_mean']:.6f}",
        flush=True,
    )


def build_query_split(
    *,
    n_query: int,
    seed: int,
    val_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_count <= 0 or val_count >= n_query:
        raise ValueError(f"val-count must be in [1, n_query-1], got {val_count}")

    rng = np.random.default_rng(seed)
    val_indices = np.sort(rng.choice(n_query, size=val_count, replace=False))
    is_val = np.zeros(n_query, dtype=bool)
    is_val[val_indices] = True
    test_indices = np.nonzero(~is_val)[0]
    return val_indices, test_indices


def choose_query_indices(
    *,
    query_split: str,
    n_query: int,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    if query_split == "val":
        return val_indices
    if query_split == "test":
        return test_indices
    if query_split == "all":
        return np.arange(n_query, dtype=np.int64)
    raise ValueError(f"unknown query_split: {query_split}")


def benchmark_config_set(
    *,
    dataset: str,
    topk: int,
    compile_module: bool,
    seed: int,
    val_count: int,
    output_dir: Path,
    n_thread: int,
    n_bit: int,
    configs: list[tuple[int, int]],
    query_split: str,
) -> Path:
    runtime = load_igp_index_for_dataset(
        dataset,
        compile_module=compile_module,
        n_bit=n_bit)
    query_l = runtime["query_l"]
    query_ids = runtime["query_ids"]
    n_query = len(query_l)
    val_indices, test_indices = build_query_split(
        n_query=n_query,
        seed=seed,
        val_count=val_count,
    )
    local_indices = choose_query_indices(
        query_split=query_split,
        n_query=n_query,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    results: list[dict] = []
    total_configs = len(configs)
    for ordinal, (phi_pb, phi_ref) in enumerate(configs, start=1):
        result = evaluate_config(
            eval_igp=runtime["eval_igp"],
            index=runtime["index"],
            query_l=query_l,
            query_ids=query_ids,
            raw_gt=runtime["raw_gt"],
            local_indices=local_indices,
            topk=topk,
            phi_pb=phi_pb,
            phi_ref=phi_ref,
            n_thread=n_thread,
        )
        results.append(result)
        print_config_result(
            dataset=dataset,
            topk=topk,
            phase=query_split.upper(),
            ordinal=ordinal,
            total=total_configs,
            row=result,
        )

    result = {
        "mode": "config_set",
        "dataset": dataset,
        "topk": topk,
        "query_split": query_split,
        "metric_name": metric_name(topk, end_to_end_metric_available=False),
        "build_index_parameters": {
            "n_centroid_formula": "16 * sqrt(N_v)",
            "n_centroid": int(runtime["n_centroid"]),
            "n_bit": int(runtime["n_bit"]),
            "efConstruction": 200,
            "M": 64,
        },
        "retrieval_parameters_from_paper": {
            "n_b": PAPER_NB,
            "b_s": PAPER_BS,
            "search_threads": n_thread,
            "validation_query_count": val_count,
            "validation_seed": seed,
            "note": (
                "Explicit config-set mode for cluster/distributed execution. "
                "The current vendored CPU IGP Python API only exposes "
                "nprobe, probe_topk, and n_thread, so n_b/b_s are recorded but not set."
            ),
        },
        "index_path": runtime["index_path"],
        "query_count_total": int(n_query),
        "query_count_val": int(len(val_indices)),
        "query_count_test": int(len(test_indices)),
        "query_count_run": int(len(local_indices)),
        "configs": [
            {"phi_pb": phi_pb, "phi_ref": phi_ref} for phi_pb, phi_ref in configs
        ],
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_top{topk}_{query_split}_config_set.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


def benchmark_dataset(
    dataset: str,
    topk: int,
    compile_module: bool,
    seed: int,
    val_count: int,
    output_dir: Path,
    n_thread: int,
    n_bit: int,
) -> Path:
    runtime = load_igp_index_for_dataset(
        dataset,
        compile_module=compile_module,
        n_bit=n_bit)
    query_l = runtime["query_l"]
    query_ids = runtime["query_ids"]
    n_query = len(query_l)
    val_indices, test_indices = build_query_split(
        n_query=n_query,
        seed=seed,
        val_count=val_count,
    )

    validation_results: list[dict] = []
    total_validation_configs = len(PAPER_PHI_PB_VALUES) * len(PAPER_PHI_REF_MULTIPLIERS)
    validation_ordinal = 0
    for phi_pb in PAPER_PHI_PB_VALUES:
        for mult in PAPER_PHI_REF_MULTIPLIERS:
            phi_ref = mult * topk
            validation_ordinal += 1
            result = evaluate_config(
                eval_igp=runtime["eval_igp"],
                index=runtime["index"],
                query_l=query_l,
                query_ids=query_ids,
                raw_gt=runtime["raw_gt"],
                local_indices=val_indices,
                topk=topk,
                phi_pb=phi_pb,
                phi_ref=phi_ref,
                n_thread=n_thread,
            )
            validation_results.append(result)
            print_config_result(
                dataset=dataset,
                topk=topk,
                phase="VAL",
                ordinal=validation_ordinal,
                total=total_validation_configs,
                row=result,
            )

    best_val = choose_best(validation_results)
    print(
        f"[BEST-VAL] dataset={dataset} topk={topk} "
        f"phi_pb={best_val['phi_pb']} phi_ref={best_val['phi_ref']} "
        f"qps={best_val['qps']:.3f} avg_latency_ms={best_val['avg_latency_ms']:.3f} "
        f"recall={best_val['recall_mean']:.6f}",
        flush=True,
    )
    test_result = evaluate_config(
        eval_igp=runtime["eval_igp"],
        index=runtime["index"],
        query_l=query_l,
        query_ids=query_ids,
        raw_gt=runtime["raw_gt"],
        local_indices=test_indices,
        topk=topk,
        phi_pb=int(best_val["phi_pb"]),
        phi_ref=int(best_val["phi_ref"]),
        n_thread=n_thread,
    )
    print_config_result(
        dataset=dataset,
        topk=topk,
        phase="TEST",
        ordinal=1,
        total=1,
        row=test_result,
    )

    result = {
        "paper_section": "5.1 Experimental settings",
        "dataset": dataset,
        "topk": topk,
        "metric_note": (
            "This local benchmark follows the paper's search-parameter procedure exactly, "
            "but reports recall from raw gt.tsv universally. "
            "End-to-end MRR/NDCG require queries.gnd.jsonl/collection.tsv assets that are not "
            "present for every local dataset."
        ),
        "target_metric_name": metric_name(topk, end_to_end_metric_available=False),
        "build_index_parameters": {
            "n_centroid_formula": "16 * sqrt(N_v)",
            "n_centroid": int(runtime["n_centroid"]),
            "n_bit": int(runtime["n_bit"]),
            "efConstruction": 200,
            "M": 64,
        },
        "retrieval_parameters_from_paper": {
            "phi_pb_grid": PAPER_PHI_PB_VALUES,
            "phi_ref_grid": [mult * topk for mult in PAPER_PHI_REF_MULTIPLIERS],
            "n_b": PAPER_NB,
            "b_s": PAPER_BS,
            "search_threads": n_thread,
            "validation_query_count": val_count,
            "validation_seed": seed,
            "note": (
                "The paper lists n_b=8 and b_s=16 as retrieval-side settings. "
                "The current vendored CPU IGP Python API only exposes "
                "nprobe, probe_topk, and n_thread, so this script records "
                "the paper values but cannot explicitly set n_b/b_s. "
                "This local driver defaults n_thread to all visible CPU cores."
            ),
        },
        "index_path": runtime["index_path"],
        "query_count_total": int(n_query),
        "query_count_val": int(len(val_indices)),
        "query_count_test": int(len(test_indices)),
        "best_validation_config": best_val,
        "test_result": test_result,
        "validation_results": validation_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_top{topk}_paper_cpu.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


def main() -> int:
    args = parse_args()
    out_dir = repo_root() / args.output_dir
    produced = []
    for dataset in args.datasets:
        for topk in args.topks:
            print(
                f"[RUN] dataset={dataset} topk={topk} "
                f"val_count={args.val_count} seed={args.seed} "
                f"n_thread={args.threads} n_bit={args.n_bit}",
                flush=True,
            )
            if args.configs:
                output_path = benchmark_config_set(
                    dataset=dataset,
                    topk=topk,
                    compile_module=args.compile,
                    seed=args.seed,
                    val_count=args.val_count,
                    output_dir=out_dir,
                    n_thread=args.threads,
                    n_bit=args.n_bit,
                    configs=args.configs,
                    query_split=args.query_split,
                )
            else:
                output_path = benchmark_dataset(
                    dataset=dataset,
                    topk=topk,
                    compile_module=args.compile,
                    seed=args.seed,
                    val_count=args.val_count,
                    output_dir=out_dir,
                    n_thread=args.threads,
                    n_bit=args.n_bit,
                )
            print(f"[DONE] wrote {output_path}", flush=True)
            produced.append(str(output_path))

    print("[SUMMARY] outputs:", flush=True)
    for path in produced:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
