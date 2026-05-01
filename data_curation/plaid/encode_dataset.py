#!/usr/bin/env python3
"""Encode collection.tsv into Chimera/PLAID document embedding binaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path, help="Input collection.tsv: doc_id<TAB>text")
    parser.add_argument("embedding_filename", type=Path, help="Output data.bin")
    parser.add_argument("doclens_filename", type=Path, help="Output doclens.bin")
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", help="ColBERT checkpoint name/path.")
    parser.add_argument("--chunk-size", type=int, default=25000, help="Passages streamed per encoding chunk.")
    parser.add_argument("--start-doc", type=int, default=0, help="First collection row to encode, inclusive.")
    parser.add_argument("--end-doc", type=int, default=None, help="Last collection row to encode, exclusive.")
    parser.add_argument("--cuda-device", default=None, help="Optional CUDA_VISIBLE_DEVICES value for this process.")
    parser.add_argument(
        "--nranks",
        type=int,
        default=1,
        help="ColBERT Run nranks for this process. Keep this at 1 for sharded multi-GPU encoding.",
    )
    return parser.parse_args()


def iter_passage_chunks(path: Path, *, chunk_size: int, start_doc: int, end_doc: int | None):
    chunk = []
    chunk_start = None
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if row_idx < start_doc:
                continue
            if end_doc is not None and row_idx >= end_doc:
                break

            parts = line.rstrip("\n\r").split("\t", 1)
            if len(parts) != 2:
                continue
            if chunk_start is None:
                chunk_start = row_idx
            chunk.append(parts[1])
            if len(chunk) >= chunk_size:
                yield chunk_start, chunk
                chunk = []
                chunk_start = None

    if chunk:
        yield chunk_start, chunk


def encode_collection(args: argparse.Namespace) -> None:
    if not args.collection.is_file():
        raise SystemExit(f"missing collection file: {args.collection}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if args.start_doc < 0:
        raise SystemExit("--start-doc must be >= 0")
    if args.end_doc is not None and args.end_doc < args.start_doc:
        raise SystemExit("--end-doc must be >= --start-doc")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    args.embedding_filename.parent.mkdir(parents=True, exist_ok=True)
    args.doclens_filename.parent.mkdir(parents=True, exist_ok=True)

    from colbert.indexing.collection_encoder import CollectionEncoder
    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.modeling.checkpoint import Checkpoint

    with Run().context(RunConfig(nranks=args.nranks)):
        config = ColBERTConfig()
        checkpoint = Checkpoint(args.checkpoint, config)
        encoder = CollectionEncoder(config, checkpoint)
        batches = iter_passage_chunks(
            args.collection,
            chunk_size=args.chunk_size,
            start_doc=args.start_doc,
            end_doc=args.end_doc,
        )

        total_tokens = 0
        total_docs = 0
        dim = None
        with args.embedding_filename.open("wb") as embeddings, args.doclens_filename.open("wb") as doclens_file:
            np.array([0], dtype=np.int32).tofile(embeddings)
            np.array([0], dtype=np.int32).tofile(embeddings)
            np.array([0], dtype=np.int32).tofile(doclens_file)

            for chunk_idx, (offset, passages) in enumerate(tqdm.tqdm(batches, disable=config.rank > 0)):
                encoded, doclens = encoder.encode_passages(passages)
                encoded_np = encoded.numpy().astype("float32")
                doclens_np = np.array(doclens, dtype=np.int32)
                total_tokens += encoded_np.shape[0]
                total_docs += doclens_np.shape[0]
                dim = encoded_np.shape[1]
                encoded_np.tofile(embeddings)
                doclens_np.tofile(doclens_file)
                embeddings.flush()
                doclens_file.flush()
                print(f"encoded chunk={chunk_idx} offset={offset} shape={encoded_np.shape}")
                del encoded, encoded_np, doclens_np

            if dim is None:
                raise SystemExit(f"collection produced no embeddings: {args.collection}")

            embeddings.seek(0)
            np.array([total_tokens], dtype=np.int32).tofile(embeddings)
            np.array([dim], dtype=np.int32).tofile(embeddings)
            doclens_file.seek(0)
            np.array([total_docs], dtype=np.int32).tofile(doclens_file)

    print(f"wrote {args.embedding_filename}")
    print(f"wrote {args.doclens_filename}")
    print(f"total embeddings: n={total_tokens}, d={dim}, docs={total_docs}")


def main() -> None:
    encode_collection(parse_args())


if __name__ == "__main__":
    main()
