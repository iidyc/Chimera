#!/usr/bin/env python3
"""Normalize official dataset downloads into dataset/<name>/text files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def clean(value: object) -> str:
    return " ".join(str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    return path


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise SystemExit(f"could not find {name} under {root}")
    return matches[0]


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"output directory already exists and is not empty: {path} (pass --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(out_dir: Path, metadata: dict) -> None:
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def iter_tsv(path: Path, *, has_header: bool) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        if has_header:
            next(handle, None)
        for line in handle:
            line = line.rstrip("\n\r")
            if line:
                rows.append(line.split("\t"))
    return rows


def normalize_msmarco(args: argparse.Namespace) -> None:
    prepare_output(args.output_dir, args.overwrite)
    collection_path = find_one(args.input_dir, "collection.tsv")
    queries_path = find_one(args.input_dir, "queries.dev.small.tsv")
    qrels_path = find_one(args.input_dir, "qrels.dev.small.tsv")

    collection_count = 0
    with collection_path.open("r", encoding="utf-8") as src, (args.output_dir / "collection.tsv").open(
        "w", encoding="utf-8"
    ) as dst:
        for row_idx, line in enumerate(src):
            if args.limit is not None and row_idx >= args.limit:
                break
            dst.write(line)
            collection_count += 1

    source_to_local: dict[str, str] = {}
    query_count = 0
    with queries_path.open("r", encoding="utf-8") as src, (args.output_dir / "queries.tsv").open(
        "w", encoding="utf-8"
    ) as queries, (args.output_dir / "query_id_map.tsv").open("w", encoding="utf-8") as mapping:
        mapping.write("local_qid\tsource_qid\n")
        for local_qid, line in enumerate(src):
            if args.limit is not None and local_qid >= args.limit:
                break
            parts = line.rstrip("\n\r").split("\t", 1)
            if len(parts) != 2:
                continue
            source_qid, text = parts
            source_to_local[source_qid] = str(local_qid)
            queries.write(f"{local_qid}\t{clean(text)}\n")
            mapping.write(f"{local_qid}\t{source_qid}\n")
            query_count += 1

    qrels_count = 0
    with qrels_path.open("r", encoding="utf-8") as src, (args.output_dir / "qrels.tsv").open(
        "w", encoding="utf-8"
    ) as qrels:
        qrels.write("query-id\tcorpus-id\tscore\n")
        for line in src:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 4:
                continue
            source_qid, _, doc_id, score = parts[:4]
            local_qid = source_to_local.get(source_qid)
            if local_qid is None:
                continue
            qrels.write(f"{local_qid}\t{doc_id}\t{score}\n")
            qrels_count += 1

    write_metadata(
        args.output_dir,
        {
            "dataset": "msmarco",
            "source": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
            "query_subset": "queries.dev.small.tsv",
            "files": {
                "collection": "collection.tsv",
                "queries": "queries.tsv",
                "qrels": "qrels.tsv",
                "query_id_map": "query_id_map.tsv",
            },
            "counts": {"collection": collection_count, "queries": query_count, "qrels": qrels_count},
            "limit": args.limit,
        },
    )


def load_hotpot_queries(path: Path) -> dict[str, str]:
    queries = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = clean(row.get("_id"))
            text = clean(row.get("text"))
            if query_id and text:
                queries[query_id] = text
    return queries


def normalize_hotpot(args: argparse.Namespace) -> None:
    prepare_output(args.output_dir, args.overwrite)
    corpus_path = find_one(args.input_dir, "corpus.jsonl")
    queries_path = find_one(args.input_dir, "queries.jsonl")
    qrels_path = require_file(args.input_dir / "qrels" / f"{args.qrels_split}.tsv")

    collection_count = 0
    with corpus_path.open("r", encoding="utf-8") as src, (args.output_dir / "collection.tsv").open(
        "w", encoding="utf-8"
    ) as collection:
        for row_idx, line in enumerate(src):
            if args.limit is not None and row_idx >= args.limit:
                break
            row = json.loads(line)
            doc_id = clean(row.get("_id"))
            title = clean(row.get("title"))
            text = clean(row.get("text"))
            body = f"{title} {text}".strip() if title else text
            if doc_id and body:
                collection.write(f"{doc_id}\t{body}\n")
                collection_count += 1

    qrels_rows = iter_tsv(qrels_path, has_header=True)
    selected_source_qids: list[str] = []
    seen_qids: set[str] = set()
    for row in qrels_rows:
        if len(row) < 3:
            continue
        source_qid = row[0]
        if source_qid not in seen_qids:
            if args.max_queries is not None and len(selected_source_qids) >= args.max_queries:
                break
            selected_source_qids.append(source_qid)
            seen_qids.add(source_qid)

    source_to_local = {source_qid: str(local_qid) for local_qid, source_qid in enumerate(selected_source_qids)}
    query_text = load_hotpot_queries(queries_path)

    query_count = 0
    with (args.output_dir / "queries.tsv").open("w", encoding="utf-8") as queries, (
        args.output_dir / "query_id_map.tsv"
    ).open("w", encoding="utf-8") as mapping:
        mapping.write("local_qid\tsource_qid\n")
        for source_qid in selected_source_qids:
            local_qid = source_to_local[source_qid]
            text = query_text.get(source_qid)
            if text is None:
                continue
            queries.write(f"{local_qid}\t{text}\n")
            mapping.write(f"{local_qid}\t{source_qid}\n")
            query_count += 1

    qrels_count = 0
    with (args.output_dir / "qrels.tsv").open("w", encoding="utf-8") as qrels:
        qrels.write("query-id\tcorpus-id\tscore\n")
        for row in qrels_rows:
            if len(row) < 3:
                continue
            source_qid, doc_id, score = row[:3]
            local_qid = source_to_local.get(source_qid)
            if local_qid is None:
                continue
            qrels.write(f"{local_qid}\t{doc_id}\t{score}\n")
            qrels_count += 1

    write_metadata(
        args.output_dir,
        {
            "dataset": "hotpot",
            "source": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
            "qrels_split": args.qrels_split,
            "max_queries": args.max_queries,
            "files": {
                "collection": "collection.tsv",
                "queries": "queries.tsv",
                "qrels": "qrels.tsv",
                "query_id_map": "query_id_map.tsv",
            },
            "counts": {"collection": collection_count, "queries": query_count, "qrels": qrels_count},
            "limit": args.limit,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    msmarco = subparsers.add_parser("msmarco")
    msmarco.add_argument("--input-dir", type=Path, required=True)
    msmarco.add_argument("--output-dir", type=Path, required=True)
    msmarco.add_argument("--overwrite", action="store_true")
    msmarco.add_argument("--limit", type=int, default=None)
    msmarco.set_defaults(func=normalize_msmarco)

    hotpot = subparsers.add_parser("hotpot")
    hotpot.add_argument("--input-dir", type=Path, required=True)
    hotpot.add_argument("--output-dir", type=Path, required=True)
    hotpot.add_argument("--overwrite", action="store_true")
    hotpot.add_argument("--limit", type=int, default=None)
    hotpot.add_argument("--qrels-split", default="test")
    hotpot.add_argument("--max-queries", type=int, default=1000)
    hotpot.set_defaults(func=normalize_hotpot)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
