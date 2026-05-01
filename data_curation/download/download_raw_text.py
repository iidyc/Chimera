#!/usr/bin/env python3
"""Download raw retrieval text into dataset/<name>/text.

The generated files use a simple TSV layout:

  collection.tsv: doc_id<TAB>text
  queries.tsv:    query_id<TAB>text
  qrels.tsv:      query-id<TAB>corpus-id<TAB>score

The script streams from the source datasets by default so it can handle large
corpora without materializing the full dataset in memory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    layout: str


DATASET_SPECS = {
    # LoTTE Pooled in this artifact corresponds to ColBERTv2 pooled/dev search.
    "lotte": DatasetSpec("lotte", "colbertv2/lotte_passages", "colbert_lotte"),
    "hotpot": DatasetSpec("hotpot", "official BEIR hotpotqa.zip", "official_script"),
    "msmarco": DatasetSpec("msmarco", "official MS MARCO collectionandqueries.tar.gz", "official_script"),
}

COLBERT_LOTTE_BASE_URL = "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main"
COLBERT_LOTTE_SPLIT = "dev"
COLBERT_LOTTE_QUERY_TYPE = "search"


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"output directory already exists and is not empty: {path} (pass --overwrite)")
    path.mkdir(parents=True, exist_ok=True)


def download_url_lines(url: str, output_path: Path, *, limit: int | None) -> int:
    count = 0
    with urlopen(url) as response, output_path.open("wb") as output:
        for line in response:
            if limit is not None and count >= limit:
                break
            output.write(line)
            count += 1
    return count


def download_colbert_lotte(out_dir: Path, *, limit: int | None) -> dict:
    """Download the LoTTE pooled-dev search split used by the artifact."""

    base = f"{COLBERT_LOTTE_BASE_URL}/pooled/{COLBERT_LOTTE_SPLIT}"
    collection_url = f"{base}/collection.tsv"
    questions_url = f"{base}/questions.{COLBERT_LOTTE_QUERY_TYPE}.tsv"
    qas_url = f"{base}/qas.{COLBERT_LOTTE_QUERY_TYPE}.jsonl"

    counts = {"collection": 0, "queries": 0, "qrels": 0}
    counts["collection"] = download_url_lines(collection_url, out_dir / "collection.tsv", limit=limit)
    counts["queries"] = download_url_lines(questions_url, out_dir / "queries.tsv", limit=limit)

    with urlopen(qas_url) as response, (out_dir / "qrels.tsv").open("w", encoding="utf-8") as qrels:
        qrels.write("query-id\tcorpus-id\tscore\n")
        for line_index, line in enumerate(response):
            if limit is not None and line_index >= limit:
                break
            row = json.loads(line.decode("utf-8"))
            query_id = normalize_cell(row.get("qid"))
            for answer_pid in row.get("answer_pids", []):
                doc_id = normalize_cell(answer_pid)
                if query_id and doc_id:
                    qrels.write(f"{query_id}\t{doc_id}\t1\n")
                    counts["qrels"] += 1

    return counts


def download_with_official_script(
    name: str,
    output_root: Path,
    *,
    overwrite: bool,
    limit: int | None,
) -> dict:
    script = REPO_ROOT / "data_curation" / "plaid" / f"download_{name}.sh"
    if not script.is_file():
        raise SystemExit(f"missing official download script: {script}")

    cmd = [str(script), "--output-root", str(output_root)]
    if overwrite:
        cmd.append("--overwrite")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd, check=True)

    metadata_path = output_root / name / "text" / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"official download script did not write metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("counts", {})


def download_dataset(name: str, output_root: Path, *, overwrite: bool, limit: int | None) -> None:
    spec = DATASET_SPECS[name]
    if spec.layout == "colbert_lotte":
        out_dir = output_root / spec.name / "text"
        ensure_output_dir(out_dir, overwrite=overwrite)
        counts = download_colbert_lotte(out_dir, limit=limit)
    elif spec.layout == "official_script":
        counts = download_with_official_script(spec.name, output_root, overwrite=overwrite, limit=limit)
        out_dir = output_root / spec.name / "text"
        print(f"[{name}] wrote {out_dir}")
        print(f"[{name}] collection={counts['collection']} queries={counts['queries']} qrels={counts['qrels']}")
        return
    else:
        raise AssertionError(f"unsupported layout: {spec.layout}")

    metadata = {
        "dataset": spec.name,
        "source_repo": spec.repo,
        "layout": spec.layout,
        "files": {
            "collection": "collection.tsv",
            "queries": "queries.tsv",
            "qrels": "qrels.tsv",
        },
        "counts": counts,
        "limit_per_split": limit,
    }
    if spec.layout == "colbert_lotte":
        metadata["lotte_split"] = COLBERT_LOTTE_SPLIT
        metadata["lotte_query_type"] = COLBERT_LOTTE_QUERY_TYPE
        metadata["source_urls"] = {
            "collection": f"{COLBERT_LOTTE_BASE_URL}/pooled/{COLBERT_LOTTE_SPLIT}/collection.tsv",
            "queries": (
                f"{COLBERT_LOTTE_BASE_URL}/pooled/{COLBERT_LOTTE_SPLIT}/"
                f"questions.{COLBERT_LOTTE_QUERY_TYPE}.tsv"
            ),
            "qrels_source": (
                f"{COLBERT_LOTTE_BASE_URL}/pooled/{COLBERT_LOTTE_SPLIT}/"
                f"qas.{COLBERT_LOTTE_QUERY_TYPE}.jsonl"
            ),
        }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"[{name}] wrote {out_dir}")
    print(f"[{name}] collection={counts['collection']} queries={counts['queries']} qrels={counts['qrels']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_SPECS.keys()],
        default="all",
        help="Dataset to download. Default: all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing dataset/<name>. Default: repository dataset/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in an existing text directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug option: write at most this many rows per source split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        download_dataset(
            name,
            args.output_root,
            overwrite=args.overwrite,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
