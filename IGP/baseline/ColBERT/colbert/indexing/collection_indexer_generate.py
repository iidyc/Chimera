import os
import tqdm
import time
import ujson
import torch
import random
import re

try:
    import faiss
except ImportError as e:
    print("WARNING: faiss must be imported for indexing")

import numpy as np
import torch.multiprocessing as mp
from colbert.infra.config.config import ColBERTConfig

import colbert.utils.distributed as distributed

from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats
from colbert.modeling.checkpoint import Checkpoint
from colbert.data.collection import Collection

from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.index_saver_generate import IndexSaverGenerate
from colbert.indexing.utils import optimize_ivf
from colbert.utils.utils import flatten, print_message

from colbert.indexing.codecs.residual_generate import ResidualCodecGenerate


class CollectionIndexerGenerate():
    '''
    Given a collection and config, encode collection into index and
    stores the index on the disk in chunks.
    '''

    def __init__(self, username: str, dataset: str):

        self.username = username
        self.dataset = dataset

        embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
        self.item_n_vec_l = np.load(os.path.join(embedding_path, 'doclens.npy')).astype(np.int32)
        self.n_item = len(self.item_n_vec_l)
        print(f'n_item {self.n_item}')

        self.use_gpu = True

        self.index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/plaid'
        os.system(f'rm -r {self.index_path}')
        os.makedirs(self.index_path, exist_ok=False)

        embedding_path = f'/home/{self.username}/Dataset/multi-vector-retrieval/Embedding/{self.dataset}'
        vec_l = np.load(os.path.join(embedding_path, 'base_embedding', 'encoding0_float32.npy'))
        self.vec_dim = vec_l.shape[1]
        self.nbits = 2

        self.saver = IndexSaverGenerate(index_path=self.index_path, vec_dim=self.vec_dim, nbits=self.nbits, use_gpu=self.use_gpu)

    def run(self):
        with torch.inference_mode():
            start_time = time.time()
            self.setup()  # Computes and saves plan for whole collection
            setup_time = time.time() - start_time
            print_message("finish setup")

            start_time = time.time()
            if not self.saver.try_load_codec():
                self.train()  # Trains centroids from selected passages
            train_time = time.time() - start_time
            print_message("finish train")

            start_time = time.time()
            encode_passage_time = self.index()  # Encodes and saves all tokens into residuals
            index_time = time.time() - start_time
            print_message("finish encode passage")

            start_time = time.time()
            self.finalize()  # Builds metadata and centroid to passage mapping
            finalize_time = time.time() - start_time
            print_message("finish finalize")
        build_index_time = setup_time + train_time + index_time - encode_passage_time + finalize_time
        return build_index_time, encode_passage_time

    def setup(self):
        '''
        Calculates and saves plan.json for the whole collection.

        plan.json { config, num_chunks, num_partitions, num_embeddings_est, avg_doclen_est}
        num_partitions is the number of centroids to be generated.
        '''

        embedding_path = f'/home/{self.username}/Dataset/multi-vector-retrieval/Embedding/{self.dataset}'
        item_n_vec0_l = np.load(os.path.join(embedding_path, 'base_embedding', 'doclens0.npy'))
        self.chunk_size = len(item_n_vec0_l)
        n_item = len(self.item_n_vec_l)
        self.num_chunks = int(np.ceil(n_item / self.chunk_size))

        # Saves sampled passages and embeddings for training k-means centroids later
        sampled_pids = self._sample_pids(n_item=n_item)
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        # Select the number of partitions
        num_passages = n_item
        self.num_embeddings_est = num_passages * avg_doclen_est
        self.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))

        print_message(f'Creaing {self.num_partitions:,} partitions.')
        print_message(f'*Estimated* {int(self.num_embeddings_est):,} embeddings.')

        # save the plan into json file
        self._save_plan()

    # generate the itemID_l that used for kmeans
    def _sample_pids(self, n_item: int):
        num_passages = n_item

        # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        # Keep in mind that, say, 15% still means at least 100k.
        # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
        # Then we subsample the vectors to 100 * num_partitions

        typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
        # sampled_pids = 16 * np.sqrt(typical_doclen * num_passages)
        sampled_pids = np.sqrt(typical_doclen * num_passages) / 2
        # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
        sampled_pids = min(1 + int(sampled_pids), num_passages)

        sampled_pids = random.sample(range(num_passages), sampled_pids)
        print_message(f"# of sampled PIDs = {len(sampled_pids)} \t sampled_pids[:3] = {sampled_pids[:3]}")

        return set(sampled_pids)

    def _sample_embeddings(self, sampled_pids):
        sampled_pids = list(sampled_pids)

        sample_vecs_l, sample_item_n_vec_l, vecsID_l = get_sample_vecs_l(sample_itemID_l=sampled_pids,
                                                                         DEFAULT_CHUNKSIZE=self.chunk_size,
                                                                         username=self.username, dataset=self.dataset,
                                                                         vec_dim=self.vec_dim)

        sample_vecs_l = torch.Tensor(sample_vecs_l)
        self.num_sample_embs = sample_vecs_l.shape[0]

        avg_doclen_est = np.average(self.item_n_vec_l[sampled_pids])
        self.avg_doclen_est = avg_doclen_est

        print_message(f'avg_doclen_est = {avg_doclen_est}')

        torch.save(sample_vecs_l.half(), os.path.join(self.index_path, f'sample_vec.pt'))

        return avg_doclen_est

    def _save_plan(self):
        self.plan_path = os.path.join(self.index_path, 'plan.json')
        print_message("#> Saving the indexing plan to", self.plan_path, "..")

        with open(self.plan_path, 'w') as f:
            d = {'config': {}}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings_est'] = self.num_embeddings_est
            d['avg_doclen_est'] = self.avg_doclen_est

            f.write(ujson.dumps(d, indent=4) + '\n')

    def train(self):

        sample, heldout = self._concatenate_and_split_sample()

        centroids = self._train_kmeans(sample)

        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)

        print_message(f'avg_residual = {avg_residual}')

        # Compute and save codec into avg_residual.pt, buckets.pt and centroids.pt
        codec = ResidualCodecGenerate(vec_dim=self.vec_dim, nbits=self.nbits, use_gpu=self.use_gpu,
                                      centroids=centroids, avg_residual=avg_residual,
                                      bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
        self.saver.save_codec(codec)

    def _concatenate_and_split_sample(self):
        print_memory_stats(f'_concatenate_and_split_sample ***1***')

        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        sample = torch.empty(self.num_sample_embs, self.vec_dim, dtype=torch.float16)

        offset = 0
        sub_sample_path = os.path.join(self.index_path, f'sample_vec.pt')
        sub_sample = torch.load(sub_sample_path)
        os.remove(sub_sample_path)

        endpos = offset + sub_sample.size(0)
        sample[offset:endpos] = sub_sample
        offset = endpos

        assert endpos == sample.size(0), (endpos, sample.size())

        print_memory_stats(f'_concatenate_and_split_sample ***2***')

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample = sample[torch.randperm(sample.size(0))]

        print_memory_stats(f'_concatenate_and_split_sample ***3***')

        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, sample_heldout = sample.split([sample.size(0) - heldout_size, heldout_size], dim=0)

        print_memory_stats(f'_concatenate_and_split_sample ***4***')

        return sample, sample_heldout

    def _train_kmeans(self, sample):
        if self.use_gpu:
            torch.cuda.empty_cache()

        centroids = compute_faiss_kmeans(dim=self.vec_dim, num_partitions=self.num_partitions, kmeans_niters=20,
                                         use_gpu=self.use_gpu, sample_vec_l=sample)

        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        if self.use_gpu:
            centroids = centroids.half()
        else:
            centroids = centroids.float()

        return centroids

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodecGenerate(vec_dim=self.vec_dim, nbits=self.nbits, use_gpu=self.use_gpu,
                                           centroids=centroids, avg_residual=None)

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device='cuda' if self.use_gpu else 'cpu')
        heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct,
                                                          out_device='cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            heldout_avg_residual = heldout.cuda() - heldout_reconstruct
        else:
            heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        print([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2 ** self.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        print_message(
            f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}")
        print_message(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

        # EVENTAULLY: Compare the above with non-heldout sample. If too different, we can do better!
        # sample = sample[subsample_idxs]
        # sample_reconstruct = get_centroids_for(centroids, sample)
        # sample_avg_residual = (sample - sample_reconstruct).mean(dim=0)

    def index(self):
        '''
        Encode embeddings for all passages in collection.
        Each embedding is converted to code (centroid id) and residual.
        Embeddings stored according to passage order in contiguous chunks of memory.

        Saved data files described below:
            {CHUNK#}.codes.pt:      centroid id for each embedding in chunk
            {CHUNK#}.residuals.pt:  16-bits residual for each embedding in chunk
            doclens.{CHUNK#}.pt:    number of embeddings within each passage in chunk
        '''
        base_dir = f'/home/{self.username}/Dataset/multi-vector-retrieval/Embedding/{self.dataset}/base_embedding'
        n_chunk = get_n_chunk(base_dir)
        embedding_path = f'/home/{self.username}/Dataset/multi-vector-retrieval/Embedding/{self.dataset}'

        encode_passage_time = 0
        with self.saver.thread():
            offset = 0
            for chunk_idx in tqdm.tqdm(range(n_chunk)):
                embs = np.load(os.path.join(base_dir, f'encoding{chunk_idx}_float32.npy'))
                doclens = np.load(os.path.join(base_dir, f'doclens{chunk_idx}.npy')).tolist()
                embs = torch.Tensor(embs).half()

                if self.use_gpu:
                    assert embs.dtype == torch.float16
                else:
                    assert embs.dtype == torch.float32
                    embs = embs.half()

                print_message(f"#> Saving chunk {chunk_idx}: \t {len(doclens):,} passages "
                              f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")

                self.saver.save_chunk(chunk_idx, offset, embs, doclens)  # offset = first passage index in chunk
                offset += len(doclens)
                del embs, doclens
        return encode_passage_time

    def finalize(self):
        '''
        Aggregates and stores metadata for each chunk and the whole index
        Builds and saves inverse mapping from centroids to passage IDs

        Saved data files described below:
            {CHUNK#}.metadata.json: [ passage_offset, num_passages, num_embeddings, embedding_offset ]
            metadata.json: [ num_chunks, num_partitions, num_embeddings, avg_doclen ]
            inv.pid.pt: [ ivf, ivf_lengths ]
                ivf is an array of passage IDs for centroids 0, 1, ...
                ivf_length contains the number of passage IDs for each centroid
        '''

        self._check_all_files_are_saved()
        self._collect_embedding_id_offset()

        self._build_ivf()
        self._update_metadata()

    def _check_all_files_are_saved(self):
        print_message("#> Checking all files were saved...")
        success = True
        for chunk_idx in range(self.num_chunks):
            if not self.saver.check_chunk_exists(chunk_idx):
                success = False
                print_message(f"#> ERROR: Could not find chunk {chunk_idx}!")
                # TODO: Fail here?
        if success:
            print_message("Found all files!")

    def _collect_embedding_id_offset(self):
        passage_offset = 0
        embedding_offset = 0

        self.embedding_offsets = []

        for chunk_idx in range(self.num_chunks):
            metadata_path = os.path.join(self.index_path, f'{chunk_idx}.metadata.json')

            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                chunk_metadata['embedding_offset'] = embedding_offset
                self.embedding_offsets.append(embedding_offset)

                assert chunk_metadata['passage_offset'] == passage_offset, (chunk_idx, passage_offset, chunk_metadata)

                passage_offset += chunk_metadata['num_passages']
                embedding_offset += chunk_metadata['num_embeddings']

            with open(metadata_path, 'w') as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + '\n')

        self.num_embeddings = embedding_offset
        assert len(self.embedding_offsets) == self.num_chunks

    def _build_ivf(self):
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.

        print_message("#> Building IVF...")

        codes = torch.zeros(self.num_embeddings, ).long()

        print_message("#> Loading codes...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodecGenerate.Embeddings.load_codes(self.index_path, chunk_idx)

            codes[offset:offset + chunk_codes.size(0)] = chunk_codes

        assert offset + chunk_codes.size(0) == codes.size(0), (offset, chunk_codes.size(0), codes.size())

        print_message(f"Sorting codes...")

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        print_message(f"Getting unique codes...")

        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(ivf, ivf_lengths, self.index_path)

    def _update_metadata(self):
        self.metadata_path = os.path.join(self.index_path, 'metadata.json')
        print_message("#> Saving the indexing metadata to", self.metadata_path, "..")

        with open(self.metadata_path, 'w') as f:
            d = {'config': {'dim': self.vec_dim, 'nbits': self.nbits}}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings'] = self.num_embeddings
            d['avg_doclen'] = self.num_embeddings / self.n_item
            d['dim'] = self.vec_dim

            f.write(ujson.dumps(d, indent=4) + '\n')


def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, use_gpu, sample_vec_l):
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)

    sample_vec_l = np.array(sample_vec_l, dtype=np.float32)

    kmeans.train(sample_vec_l)

    centroids = torch.from_numpy(kmeans.centroids)

    print_memory_stats(f'RANK:0*')

    return centroids


def get_sample_vecs_l(sample_itemID_l: list, DEFAULT_CHUNKSIZE: int, username: str, dataset: str, vec_dim: int):
    print(f'sample_itemID_l {sample_itemID_l}')
    sample_itemID_l = np.sort(sample_itemID_l)
    item_chunkID_l = [_ // DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    item_chunk_offset_l = [_ % DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    chunkID2offset_m = {}
    for chunkID, chunk_offset in zip(item_chunkID_l, item_chunk_offset_l):
        if chunkID not in chunkID2offset_m:
            chunkID2offset_m[chunkID] = [chunk_offset]
        else:
            chunkID2offset_m[chunkID].append(chunk_offset)

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')

    item_n_vecs_l = np.load(os.path.join(embedding_dir, 'doclens.npy')).astype(np.uint64)
    item_n_vecs_offset_l = np.cumsum(item_n_vecs_l)
    item_n_vecs_offset_l = np.concatenate(([0], item_n_vecs_offset_l))
    n_item = len(item_n_vecs_l)

    print("load chunk data")
    sample_vecs_l = np.array([])
    sample_item_n_vec_l = np.array([], dtype=np.uint32)
    vecsID_l = np.array([])
    for chunkID, offset_itemID_l in chunkID2offset_m.items():
        print(f"chunkID {chunkID}, n_item in chunk {len(offset_itemID_l)}")
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        item_n_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        item_n_vecs_offset_chunk_l = np.cumsum(item_n_vecs_l_chunk)
        item_n_vecs_offset_chunk_l = np.concatenate(([0], item_n_vecs_offset_chunk_l))
        item_n_vecs_offset_chunk_l = np.array(item_n_vecs_offset_chunk_l, dtype=np.uint64)

        base_itemID = chunkID * DEFAULT_CHUNKSIZE
        vecsID_l_chunk = np.array([])
        item_n_vec_l_chunk = np.array([])

        for offset_itemID in offset_itemID_l:
            itemID = base_itemID + offset_itemID
            item_n_vecs = item_n_vecs_l[itemID]
            base_vecID_chunk = item_n_vecs_offset_chunk_l[offset_itemID]
            vecsID_l_chunk = np.concatenate(
                (vecsID_l_chunk, np.arange(base_vecID_chunk, base_vecID_chunk + item_n_vecs, 1))).astype(np.uint64)
            item_n_vec_l_chunk = np.concatenate((item_n_vec_l_chunk, [item_n_vecs]))

        sample_vecs_l_chunk = item_vecs_l_chunk[vecsID_l_chunk, :]
        sample_vecs_l = sample_vecs_l_chunk if len(sample_vecs_l) == 0 else np.concatenate(
            (sample_vecs_l, sample_vecs_l_chunk)).reshape(-1, vec_dim)

        vecsID_l_chunk = vecsID_l_chunk + item_n_vecs_offset_l[base_itemID]
        vecsID_l = np.concatenate(
            (vecsID_l, vecsID_l_chunk))

        sample_item_n_vec_l = np.concatenate((sample_item_n_vec_l, item_n_vec_l_chunk))

        print("finish load chunkID")

    sample_vecs_l = sample_vecs_l.reshape(-1, vec_dim)

    assert len(vecsID_l) == len(
        sample_vecs_l), f"len(vecsID_l) {len(vecsID_l)}, len(sample_vecs_l) {len(sample_vecs_l)}"
    vecsID_l = np.array(vecsID_l, dtype=np.uint64)

    sample_item_n_vec_l = sample_item_n_vec_l.astype(np.uint32)

    return sample_vecs_l, sample_item_n_vec_l, vecsID_l

def get_n_chunk(base_dir: str):
    filename_l = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)

"""
TODOs:

1. Notice we're using self.config.bsize.

2. Consider saving/using heldout_avg_residual as a vector --- that is, using 128 averages!

3. Consider the operations with .cuda() tensors. Are all of them good for OOM?
"""
