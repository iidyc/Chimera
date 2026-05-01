#!/usr/bin/env python3
"""Encode queries.tsv into Chimera/PLAID query embedding binaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path, help="Input queries.tsv: query_id<TAB>text")
    parser.add_argument("embedding_filename", type=Path, help="Output query.bin")
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", help="ColBERT checkpoint name/path.")
    parser.add_argument(
        "--nranks",
        type=int,
        default=None,
        help="ColBERT Run nranks. Default: CUDA device count, or 1 when no GPU is visible.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Query encoding batch size.")
    return parser.parse_args()


def encode_queries(args: argparse.Namespace) -> None:
    if not args.queries.is_file():
        raise SystemExit(f"missing query file: {args.queries}")

    import torch
    from colbert.data import Queries
    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.modeling.checkpoint import Checkpoint

    args.embedding_filename.parent.mkdir(parents=True, exist_ok=True)
    nranks = args.nranks if args.nranks is not None else max(1, torch.cuda.device_count())

    with Run().context(RunConfig(nranks=nranks)):
        config = ColBERTConfig()
        checkpoint = Checkpoint(args.checkpoint, colbert_config=config)
        queries = list(Queries(str(args.queries)).values())
        batch_size = args.batch_size
        if batch_size is None:
            batch_size = 128 if len(queries) > 128 else None

        checkpoint.query_tokenizer.query_maxlen = config.query_maxlen
        query_embeddings = checkpoint.queryFromText(
            queries,
            bsize=batch_size,
            to_cpu=True,
            full_length_search=False,
        )

    query_np = query_embeddings.numpy().astype("float32")
    num_queries, query_doclen, dim = query_np.shape
    with args.embedding_filename.open("wb") as output:
        np.array([num_queries], dtype=np.int32).tofile(output)
        np.array([query_doclen], dtype=np.int32).tofile(output)
        np.array([dim], dtype=np.int32).tofile(output)
        query_np.tofile(output)

    print(f"wrote {args.embedding_filename}")
    print(f"num_queries={num_queries}, query_doclen={query_doclen}, d={dim}")


def main() -> None:
    encode_queries(parse_args())


if __name__ == "__main__":
    main()
