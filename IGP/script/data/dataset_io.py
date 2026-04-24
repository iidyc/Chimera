from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_data_root() -> Path:
    return Path(os.environ.get("IGP_LOCAL_DATA_ROOT", project_root() / "multi-vector-retrieval-data"))


def external_binary_root() -> Path:
    default_root = project_root().parent / "dataset"
    return Path(os.environ.get("GPU_MVR_DATASET_ROOT", default_root))


def embedding_dir(dataset: str) -> Path:
    return local_data_root() / "Embedding" / dataset


def index_dir(dataset: str) -> Path:
    return local_data_root() / "Index" / dataset


def rawdata_dir(dataset: str) -> Path:
    legacy_raw = local_data_root() / "RawData" / dataset
    if legacy_raw.exists():
        return legacy_raw
    return external_binary_root() / dataset / "raw"


def result_answer_dir() -> Path:
    return local_data_root() / "Result" / "answer"


def result_performance_dir() -> Path:
    return local_data_root() / "Result" / "performance"


def result_single_query_dir() -> Path:
    return local_data_root() / "Result" / "single_query_performance"


def ensure_runtime_layout(dataset: str) -> None:
    embedding_dir(dataset).mkdir(parents=True, exist_ok=True)
    index_dir(dataset).mkdir(parents=True, exist_ok=True)
    result_answer_dir().mkdir(parents=True, exist_ok=True)
    result_performance_dir().mkdir(parents=True, exist_ok=True)
    result_single_query_dir().mkdir(parents=True, exist_ok=True)


def _legacy_doclens_path(dataset: str) -> Path:
    return embedding_dir(dataset) / "doclens.npy"


def _legacy_query_path(dataset: str) -> Path:
    return embedding_dir(dataset) / "query_embedding.npy"


def _legacy_base_embedding_dir(dataset: str) -> Path:
    return embedding_dir(dataset) / "base_embedding"


def _binary_raw_dir(dataset: str) -> Path:
    return external_binary_root() / dataset / "raw"


def _binary_doclens_path(dataset: str) -> Path:
    return _binary_raw_dir(dataset) / "doclens.bin"


def _binary_query_path(dataset: str) -> Path:
    return _binary_raw_dir(dataset) / "query.bin"


def _binary_data_path(dataset: str) -> Path:
    return _binary_raw_dir(dataset) / "data.bin"


def dataset_mode(dataset: str) -> str:
    forced_mode = os.environ.get("IGP_DATA_MODE", "auto").strip().lower()
    if forced_mode not in {"auto", "legacy", "binary"}:
        raise ValueError(f"unsupported IGP_DATA_MODE={forced_mode}")

    legacy_ready = _legacy_doclens_path(dataset).exists()
    binary_ready = _binary_doclens_path(dataset).exists() and _binary_data_path(dataset).exists()

    if forced_mode == "legacy":
        if not legacy_ready:
            raise FileNotFoundError(f"legacy dataset layout not found for {dataset}")
        return "legacy"
    if forced_mode == "binary":
        if not binary_ready:
            raise FileNotFoundError(f"binary dataset layout not found for {dataset}")
        return "binary"

    if binary_ready:
        return "binary"
    if legacy_ready:
        return "legacy"
    raise FileNotFoundError(
        f"could not resolve dataset layout for {dataset}: "
        f"missing both legacy and binary inputs"
    )


def load_doclens(username: str, dataset: str) -> np.ndarray:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy":
        return np.load(_legacy_doclens_path(dataset)).astype(np.uint32)

    path = _binary_doclens_path(dataset)
    with open(path, "rb") as handle:
        count = np.fromfile(handle, dtype=np.int32, count=1)
        if len(count) != 1:
            raise ValueError(f"failed to read doc count from {path}")
        doclens = np.fromfile(handle, dtype=np.int32, count=int(count[0]))
    if len(doclens) != int(count[0]):
        raise ValueError(f"failed to read full doclens payload from {path}")
    return doclens.astype(np.uint32)


def load_query_embeddings(username: str, dataset: str) -> np.ndarray:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy" and _legacy_query_path(dataset).exists():
        return np.load(_legacy_query_path(dataset)).astype(np.float32)

    path = _binary_query_path(dataset)
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=3)
        if len(header) != 3:
            raise ValueError(f"failed to read query header from {path}")
        num_q, q_doclen, dim = [int(v) for v in header]
        query = np.fromfile(handle, dtype=np.float32, count=num_q * q_doclen * dim)
    if query.size != num_q * q_doclen * dim:
        raise ValueError(f"failed to read full query payload from {path}")
    return query.reshape(num_q, q_doclen, dim)


def load_query_ids(username: str, dataset: str, num_queries: int) -> np.ndarray:
    del username
    path = rawdata_dir(dataset) / "document" / "queries.dev.tsv"
    if path.exists():
        query_ids = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                arr = line.split("\t", 1)
                query_ids.append(int(arr[0]))
        if len(query_ids) != num_queries:
            raise ValueError(
                f"query id count mismatch for {dataset}: "
                f"{len(query_ids)} ids vs {num_queries} query embeddings"
            )
        return np.array(query_ids, dtype=np.uint32)
    return np.arange(num_queries, dtype=np.uint32)


def load_all_embeddings(username: str, dataset: str) -> np.ndarray:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy":
        chunks = []
        for _, embs in iter_embedding_chunks("", dataset):
            chunks.append(embs)
        if not chunks:
            raise ValueError(f"no embedding chunks found for {dataset}")
        return np.concatenate(chunks, axis=0)

    path = _binary_data_path(dataset)
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if len(header) != 2:
            raise ValueError(f"failed to read embedding header from {path}")
        n_vec, dim = [int(v) for v in header]
        embs = np.fromfile(handle, dtype=np.float32, count=n_vec * dim)
    if embs.size != n_vec * dim:
        raise ValueError(f"failed to read full embedding payload from {path}")
    return embs.reshape(n_vec, dim)


def load_embedding_memmap(username: str, dataset: str) -> np.ndarray:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy":
        raise ValueError("legacy layout does not expose a single embedding memmap")

    data_path = _binary_data_path(dataset)
    with open(data_path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if len(header) != 2:
            raise ValueError(f"failed to read embedding header from {data_path}")
        n_vec, dim = [int(v) for v in header]

    return np.memmap(data_path, dtype=np.float32, mode="r", offset=8, shape=(n_vec, dim))


def embedding_dim(username: str, dataset: str) -> int:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy":
        first_chunk = _legacy_base_embedding_dir(dataset) / "encoding0_float32.npy"
        return int(np.load(first_chunk, mmap_mode="r").shape[1])

    path = _binary_data_path(dataset)
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
    if len(header) != 2:
        raise ValueError(f"failed to read embedding header from {path}")
    return int(header[1])


def n_items(username: str, dataset: str) -> int:
    return int(load_doclens(username, dataset).shape[0])


def _legacy_chunk_files(dataset: str) -> list[Tuple[Path, Path]]:
    base_dir = _legacy_base_embedding_dir(dataset)
    files = []
    chunk_id = 0
    while True:
        doclens_path = base_dir / f"doclens{chunk_id}.npy"
        embs_path = base_dir / f"encoding{chunk_id}_float32.npy"
        if not doclens_path.exists() or not embs_path.exists():
            break
        files.append((doclens_path, embs_path))
        chunk_id += 1
    return files


def iter_embedding_chunks(username: str, dataset: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    del username
    mode = dataset_mode(dataset)
    if mode == "legacy":
        for doclens_path, embs_path in _legacy_chunk_files(dataset):
            yield (
                np.load(doclens_path).astype(np.uint32),
                np.load(embs_path).astype(np.float32),
            )
        return

    doclens = load_doclens("", dataset)
    data_path = _binary_data_path(dataset)
    with open(data_path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if len(header) != 2:
            raise ValueError(f"failed to read embedding header from {data_path}")
        n_vec, dim = [int(v) for v in header]

    mmap = np.memmap(data_path, dtype=np.float32, mode="r", offset=8, shape=(n_vec, dim))
    max_tokens_per_chunk = int(os.environ.get("IGP_BINARY_MAX_TOKENS_PER_CHUNK", "262144"))
    start_doc = 0
    start_vec = 0
    n_doc = int(doclens.shape[0])
    while start_doc < n_doc:
        total_tokens = 0
        end_doc = start_doc
        while end_doc < n_doc:
            next_tokens = int(doclens[end_doc])
            if total_tokens > 0 and total_tokens + next_tokens > max_tokens_per_chunk:
                break
            total_tokens += next_tokens
            end_doc += 1
        chunk_doclens = doclens[start_doc:end_doc].astype(np.uint32, copy=True)
        chunk_embs = np.array(mmap[start_vec:start_vec + total_tokens], dtype=np.float32, copy=True)
        yield chunk_doclens, chunk_embs
        start_doc = end_doc
        start_vec += total_tokens


def groundtruth_tsv_path(dataset: str, topk: int) -> Path:
    return embedding_dir(dataset) / f"{dataset}-groundtruth-top{topk}--.tsv"
