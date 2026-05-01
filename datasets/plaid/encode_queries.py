from sys import argv

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert.modeling.checkpoint import Checkpoint
import torch
import os
import numpy as np

checkpoint = 'colbert-ir/colbertv2.0'
colbert_project_path = "/home/yanqichen_umass_edu/work/ColBERT"
n_gpu = torch.cuda.device_count()

if __name__ == "__main__":
    queries = argv[1]
    emb_filename = argv[2]
    with Run().context(RunConfig(nranks=n_gpu)):
        config = ColBERTConfig()
        checkpoint = Checkpoint(checkpoint, colbert_config=config)
        queries = Queries(queries)
        queries = list(queries.values())
        bsize = 128 if len(queries) > 128 else None
        checkpoint.query_tokenizer.query_maxlen = config.query_maxlen
        Q = checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=False)

        Q = Q.numpy().astype("float32")
        f = open(emb_filename, "wb")
        num_queries, query_doclen, d = Q.shape
        print(f'num_queries={num_queries}, query_doclen={query_doclen}, d={d}')
        np.array([num_queries], dtype=np.int32).tofile(f)
        np.array([query_doclen], dtype=np.int32).tofile(f)
        np.array([d], dtype=np.int32).tofile(f)
        Q.tofile(f)