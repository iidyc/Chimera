from sys import argv

from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.data import Collection
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import tqdm
import numpy as np
import torch

# args
checkpoint = 'colbert-ir/colbertv2.0'
n_gpu = torch.cuda.device_count()

if __name__ == "__main__":
    collection = argv[1]
    embedding_filename = argv[2]
    doc_lens_filename = argv[3]
    with Run().context(RunConfig(nranks=n_gpu)):
        config = ColBERTConfig()
        checkpoint = Checkpoint(checkpoint, config)
        encoder = CollectionEncoder(config, checkpoint)
        collection = Collection(collection)
        batches = collection.enumerate_batches(rank=config.rank)

        total_n = 0
        total_doclens = 0
        d = None
        f_emb = open(embedding_filename, "wb")
        f_doclens = open(doc_lens_filename, "wb")
        # Write placeholder headers (will be patched after encoding)
        np.array([0], dtype=np.int32).tofile(f_emb)
        np.array([0], dtype=np.int32).tofile(f_emb)
        np.array([0], dtype=np.int32).tofile(f_doclens)

        for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=config.rank > 0):
            embs, doclens = encoder.encode_passages(passages)
            embs_np = embs.numpy().astype("float32")
            doclens_np = np.array(doclens, dtype=np.int32)
            total_n += embs_np.shape[0]
            total_doclens += doclens_np.shape[0]
            d = embs_np.shape[1]
            embs_np.tofile(f_emb)
            doclens_np.tofile(f_doclens)
            print(f'encoded chunkID {chunk_idx} of size {embs_np.shape}')
            del embs, embs_np, doclens_np

        # Patch headers with actual counts
        f_emb.seek(0)
        np.array([total_n], dtype=np.int32).tofile(f_emb)
        np.array([d], dtype=np.int32).tofile(f_emb)
        f_emb.close()
        f_doclens.seek(0)
        np.array([total_doclens], dtype=np.int32).tofile(f_doclens)
        f_doclens.close()
        print(f'total embeddings: n={total_n}, d={d}, doclens={total_doclens}')
