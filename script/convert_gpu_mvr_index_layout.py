#!/usr/bin/env python3

import argparse
import ctypes
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"MVRIDXv1"
VERSION = 1
DOC_1BIT = "doc_1bit.bin"
DOC_4BIT = "doc_4bit.bin"
CLUSTER_1BIT = "cluster_1bit.bin"
IVF = "ivf.bin"
CENTROIDS = "centroids.carga"
METADATA = "index_metadata.json"
LEGACY_CPU = "cpu_index.bin"
LEGACY_QUANTIZED = "quantized_data.bin"
LEGACY_GPU = "gpu_index.bin"
LEGACY_CLUSTERED = "clustered_stage1.bin"
COPY_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass
class IndexHeader:
    n: int
    d: int
    n_clusters: int
    ex_bits: int
    padded_dim: int
    rotator_type: int


def fail(message: str) -> None:
    raise SystemExit(message)


def require_supported_platform() -> None:
    if sys.byteorder != "little":
        fail("Only little-endian hosts are supported")
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        fail("This converter currently expects 64-bit size_t")


def read_header(path: Path) -> IndexHeader:
    with path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            fail(f"{path} does not look like an MVRIDXv1 file")
        version = struct.unpack("<I", handle.read(4))[0]
        if version != VERSION:
            fail(f"{path} has unsupported version {version}")
        n, d, n_clusters, ex_bits, padded_dim = struct.unpack(
            "<QQQQQ", handle.read(5 * 8)
        )
        rotator_type = struct.unpack("<B", handle.read(1))[0]
    return IndexHeader(
        n=n,
        d=d,
        n_clusters=n_clusters,
        ex_bits=ex_bits,
        padded_dim=padded_dim,
        rotator_type=rotator_type,
    )


def rotator_type_name(rotator_type: int) -> str:
    if rotator_type == 0:
        return "MatrixRotator"
    if rotator_type == 1:
        return "FhtKacRotator"
    return "Unknown"


def one_bit_code_bytes(header: IndexHeader) -> int:
    return header.n * (header.padded_dim // 8)


def full_code_bytes(header: IndexHeader) -> int:
    return header.n * (header.padded_dim * (1 + header.ex_bits) // 8)


def factor_bytes(header: IndexHeader) -> int:
    return header.n * 4


def doc_1bit_payload_bytes(header: IndexHeader) -> int:
    return one_bit_code_bytes(header) + factor_bytes(header) + factor_bytes(header)


def combined_payload_bytes(header: IndexHeader) -> int:
    return one_bit_code_bytes(header) + full_code_bytes(header) + factor_bytes(header) + factor_bytes(header)


def prefix_bytes(path: Path, payload_bytes: int) -> int:
    size = path.stat().st_size
    if size < payload_bytes:
        fail(f"{path} is smaller than its expected payload")
    return size - payload_bytes


def copy_range(
    src_path: Path,
    dst_path: Path,
    offset: int,
    length: int,
    *,
    mode: str,
) -> None:
    with src_path.open("rb") as src, dst_path.open(mode) as dst:
        src.seek(offset)
        remaining = length
        while remaining > 0:
            chunk = src.read(min(COPY_BUFFER_BYTES, remaining))
            if len(chunk) == 0:
                fail(f"Unexpected EOF while copying from {src_path}")
            dst.write(chunk)
            remaining -= len(chunk)


def copy_file_if_needed(src_path: Path | None, dst_path: Path) -> None:
    if src_path is None or not src_path.exists():
        return
    if src_path.resolve() == dst_path.resolve():
        return
    shutil.copyfile(src_path, dst_path)


def write_metadata(output_dir: Path, header: IndexHeader) -> None:
    metadata = {
        "format": "gpu_mvr_split_index_v1",
        "doc_1bit": DOC_1BIT,
        "doc_4bit": DOC_4BIT,
        "cluster_1bit": CLUSTER_1BIT,
        "ivf": IVF,
        "centroids": CENTROIDS,
        "n": header.n,
        "d": header.d,
        "n_clusters": header.n_clusters,
        "ex_bits": header.ex_bits,
        "padded_dim": header.padded_dim,
        "rotator_type": rotator_type_name(header.rotator_type),
    }
    with (output_dir / METADATA).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def resolve_source_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    source_dir = Path(args.source_index_dir).resolve() if args.source_index_dir else None

    def explicit_or_dir(explicit: str | None, name: str) -> Path | None:
        if explicit:
            return Path(explicit).resolve()
        if source_dir is None:
            return None
        candidate = source_dir / name
        return candidate if candidate.exists() else None

    source_doc_1bit = explicit_or_dir(args.source_doc_1bit, DOC_1BIT)
    source_doc_4bit = explicit_or_dir(args.source_doc_4bit, DOC_4BIT)
    source_cluster = (
        explicit_or_dir(args.source_cluster_1bit, CLUSTER_1BIT)
        or explicit_or_dir(None, LEGACY_GPU)
        or explicit_or_dir(None, LEGACY_CLUSTERED)
    )
    source_ivf = explicit_or_dir(args.source_ivf, IVF)
    source_centroids = explicit_or_dir(args.source_centroids, CENTROIDS)
    legacy_cpu = (
        explicit_or_dir(args.source_cpu_index, LEGACY_CPU)
        or explicit_or_dir(None, LEGACY_QUANTIZED)
    )

    if source_doc_1bit and source_doc_4bit:
        return {
            "layout": "split",
            "doc_1bit": source_doc_1bit,
            "doc_4bit": source_doc_4bit,
            "cluster": source_cluster,
            "ivf": source_ivf,
            "centroids": source_centroids,
        }

    if legacy_cpu is None:
        fail("Could not locate source cpu_index.bin or quantized_data.bin")

    return {
        "layout": "legacy",
        "cpu_index": legacy_cpu,
        "cluster": source_cluster,
        "ivf": source_ivf,
        "centroids": source_centroids,
    }


def convert_legacy_layout(source_cpu_index: Path, output_dir: Path) -> IndexHeader:
    header = read_header(source_cpu_index)
    prefix = prefix_bytes(source_cpu_index, combined_payload_bytes(header))
    doc_1bit_path = output_dir / DOC_1BIT
    doc_4bit_path = output_dir / DOC_4BIT

    copy_range(source_cpu_index, doc_1bit_path, 0, prefix, mode="wb")
    copy_range(source_cpu_index, doc_4bit_path, 0, prefix, mode="wb")

    offset = prefix
    copy_range(source_cpu_index, doc_1bit_path, offset, one_bit_code_bytes(header), mode="ab")
    offset += one_bit_code_bytes(header)
    copy_range(source_cpu_index, doc_4bit_path, offset, full_code_bytes(header), mode="ab")
    offset += full_code_bytes(header)
    copy_range(source_cpu_index, doc_1bit_path, offset, factor_bytes(header), mode="ab")
    offset += factor_bytes(header)
    copy_range(source_cpu_index, doc_1bit_path, offset, factor_bytes(header), mode="ab")
    return header


def convert_split_layout(source_doc_1bit: Path, source_doc_4bit: Path, output_dir: Path) -> IndexHeader:
    header = read_header(source_doc_1bit)
    copy_file_if_needed(source_doc_1bit, output_dir / DOC_1BIT)
    copy_file_if_needed(source_doc_4bit, output_dir / DOC_4BIT)
    return header


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy gpu-mvr cpu_index.bin/gpu_index.bin outputs into the split doc_1bit/doc_4bit/cluster_1bit layout."
    )
    parser.add_argument("--source-index-dir", help="Existing index directory to read from")
    parser.add_argument("--source-cpu-index", help="Legacy cpu_index.bin path")
    parser.add_argument("--source-doc-1bit", help="Existing doc_1bit.bin path")
    parser.add_argument("--source-doc-4bit", help="Existing doc_4bit.bin path")
    parser.add_argument("--source-cluster-1bit", help="Existing cluster_1bit.bin path")
    parser.add_argument("--source-ivf", help="Existing ivf.bin path")
    parser.add_argument("--source-centroids", help="Existing centroids.carga path")
    parser.add_argument("--output-dir", required=True, help="Directory to write the split layout into")
    return parser


def main() -> None:
    require_supported_platform()
    args = build_parser().parse_args()
    source = resolve_source_paths(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if source["layout"] == "split":
        header = convert_split_layout(source["doc_1bit"], source["doc_4bit"], output_dir)
    else:
        header = convert_legacy_layout(source["cpu_index"], output_dir)

    copy_file_if_needed(source.get("cluster"), output_dir / CLUSTER_1BIT)
    copy_file_if_needed(source.get("ivf"), output_dir / IVF)
    copy_file_if_needed(source.get("centroids"), output_dir / CENTROIDS)
    write_metadata(output_dir, header)

    print(f"wrote {output_dir / DOC_1BIT}")
    print(f"wrote {output_dir / DOC_4BIT}")
    if source.get("cluster") is not None:
        print(f"wrote {output_dir / CLUSTER_1BIT}")
    if source.get("ivf") is not None:
        print(f"wrote {output_dir / IVF}")
    if source.get("centroids") is not None:
        print(f"wrote {output_dir / CENTROIDS}")
    print(f"wrote {output_dir / METADATA}")


if __name__ == "__main__":
    main()
