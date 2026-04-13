import os
os.environ["CUDA_HOME"] = "/work/pi_ameliou_umass_edu/yanqichen/cuda12.8"

from colbert import EmbeddingIndexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import torch
import numpy as np

# data_root = "/scratch3/workspace/yanqichen_umass_edu-dataset/data_prep/msmarco"
data_root = "/scratch3/workspace/yanqichen_umass_edu-dataset/data_prep/hotpot"
# read pre-computed document embeddings and doclens
# embeddings_path = f"{data_root}/msmarco_emb.bin"
embeddings_path = f"{data_root}/hotpot_emb.bin"
with open(embeddings_path, "rb") as base_file:
    n, d = np.fromfile(base_file, dtype=np.int32, count=2)
# base_file = open(embeddings_path, "rb")
# n, d = torch.from_numpy(np.fromfile(base_file, dtype=np.int32, count=2))
n, d = np.int64(n), np.int64(d)
print(f"Expecting to read {n} document embeddings of dimension {d} from {embeddings_path}")
header_bytes = 2 * np.dtype(np.int32).itemsize
data = np.memmap(embeddings_path, dtype=np.float32, mode="r",
                 offset=header_bytes, shape=(n, d))
doc_embs = torch.from_numpy(data)
# data = np.fromfile(base_file, dtype=np.float32, count=n * d)
# doc_embs = torch.from_numpy(data).reshape(n, d)
print(f"Loaded document embeddings: {doc_embs.shape}")

# doclens_file = open(f"{data_root}/msmarco_doclens.bin", "rb")
doclens_file = open(f"{data_root}/hotpot_doclens.bin", "rb")
num_docs = torch.from_numpy(np.fromfile(doclens_file, dtype=np.int32, count=1)).item()
doclens = torch.from_numpy(np.fromfile(doclens_file, dtype=np.int32, count=num_docs)).tolist()
print(f"Loaded doclens: {len(doclens)} documents, total tokens = {sum(doclens)}")

# Build index from pre-computed embeddings
with Run().context(RunConfig(experiment="colbert", root=data_root)):
    config = ColBERTConfig(dim=128, nbits=4, root=data_root)
    indexer = EmbeddingIndexer(config=config)
    indexer.index("4bit", embeddings=doc_embs, doclens=doclens)
