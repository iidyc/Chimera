#!/usr/bin/env python3
"""Merge contiguous document-embedding shards into data.bin and doclens.bin."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


COPY_CHUNK_BYTES = 128 * 1024 * 1024


def read_data_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
    if len(header) != 2:
        raise SystemExit(f"invalid embedding shard header: {path}")
    return int(header[0]), int(header[1])


def read_doclens_header(path: Path) -> int:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=1)
    if len(header) != 1:
        raise SystemExit(f"invalid doclens shard header: {path}")
    return int(header[0])


def copy_after_header(src: Path, dst, header_bytes: int) -> None:
    with src.open("rb") as handle:
        handle.seek(header_bytes)
        shutil.copyfileobj(handle, dst, length=COPY_CHUNK_BYTES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--doclens-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-doclens", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.data_shards) != len(args.doclens_shards):
        raise SystemExit("--data-shards and --doclens-shards must have the same length")

    total_tokens = 0
    total_docs = 0
    dim = None
    for data_shard, doclens_shard in zip(args.data_shards, args.doclens_shards):
        tokens, shard_dim = read_data_header(data_shard)
        docs = read_doclens_header(doclens_shard)
        if dim is None:
            dim = shard_dim
        elif dim != shard_dim:
            raise SystemExit(f"dimension mismatch in {data_shard}: expected {dim}, got {shard_dim}")
        total_tokens += tokens
        total_docs += docs

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_doclens.parent.mkdir(parents=True, exist_ok=True)

    with args.output_data.open("wb") as output:
        np.array([total_tokens, dim], dtype=np.int32).tofile(output)
        for shard in args.data_shards:
            copy_after_header(shard, output, header_bytes=8)

    with args.output_doclens.open("wb") as output:
        np.array([total_docs], dtype=np.int32).tofile(output)
        for shard in args.doclens_shards:
            copy_after_header(shard, output, header_bytes=4)

    print(f"wrote {args.output_data}: tokens={total_tokens}, d={dim}")
    print(f"wrote {args.output_doclens}: docs={total_docs}")


if __name__ == "__main__":
    main()
