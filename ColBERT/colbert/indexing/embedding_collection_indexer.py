"""
Embedding-based collection indexer: builds a ColBERT index directly from
pre-computed document embeddings, bypassing the model encoding step.
"""

import os
import tqdm
import time
import ujson
import torch
import random

try:
    import faiss
except ImportError:
    print("WARNING: faiss must be imported for indexing")

import numpy as np

from colbert.infra.config.config import ColBERTConfig
from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats

from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.utils import optimize_ivf
from colbert.utils.utils import print_message

from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.collection_indexer import compute_faiss_kmeans


class EmbeddingCollectionIndexer:
    """
    Given pre-computed embeddings and doclens, compress and store them as a
    ColBERT index on disk.  No checkpoint / model is needed.

    The embeddings tensor has shape ``(total_tokens, dim)`` and ``doclens``
    is a list of ints whose sum equals ``total_tokens``.
    """

    def __init__(self, config: ColBERTConfig, embeddings: torch.Tensor,
                 doclens: list, verbose: int = 2):
        self.verbose = verbose
        self.config = config
        self.rank = 0  # single-process only
        self.nranks = 1

        self.use_gpu = self.config.total_visible_gpus > 0

        if self.verbose > 1:
            self.config.help()

        # ---- validate inputs ------------------------------------------------
        assert embeddings.dim() == 2, \
            f"embeddings must be 2-D (total_tokens, dim), got shape {embeddings.shape}"
        assert embeddings.shape[1] == self.config.dim, \
            (f"embeddings dim {embeddings.shape[1]} != config.dim {self.config.dim}")
        assert sum(doclens) == embeddings.shape[0], \
            (f"sum(doclens)={sum(doclens)} != embeddings.shape[0]={embeddings.shape[0]}")

        self.embeddings = embeddings
        self.doclens = list(doclens)
        self.num_documents = len(self.doclens)
        self.num_embeddings = embeddings.shape[0]
        self.avg_doclen = self.num_embeddings / self.num_documents

        self.saver = IndexSaver(config)

        print_memory_stats(f'EmbeddingCollectionIndexer')

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(self):
        with torch.inference_mode():
            self.setup()
            print_memory_stats('after setup')

            self.train()
            print_memory_stats('after train')

            self.index()
            print_memory_stats('after index')

            self.finalize()
            print_memory_stats('after finalize')

    # ------------------------------------------------------------------
    # setup – compute plan.json
    # ------------------------------------------------------------------
    def setup(self):
        self.chunksize = min(25_000, 1 + self.num_documents)
        self.num_chunks = int(np.ceil(self.num_documents / self.chunksize))

        # ---- sample embeddings for k-means training ----------------------
        # Use same heuristic as CollectionIndexer._sample_pids / _sample_embeddings
        typical_doclen = 120
        n_sample_pids = 16 * np.sqrt(typical_doclen * self.num_documents)
        n_sample_pids = min(1 + int(n_sample_pids), self.num_documents)
        sampled_pids = random.sample(range(self.num_documents), n_sample_pids)

        # Collect embeddings for sampled documents
        offsets = self._pid_offsets()
        sample_indices = []
        for pid in sampled_pids:
            start = offsets[pid]
            end = start + self.doclens[pid]
            sample_indices.append(torch.arange(start, end))
        sample_indices = torch.cat(sample_indices)
        self.sample_embs = self.embeddings[sample_indices]

        self.num_sample_embs = self.sample_embs.shape[0]
        self.num_embeddings_est = self.num_documents * self.avg_doclen

        self.num_partitions = int(
            2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est)))
        )

        if self.verbose > 0:
            Run().print_main(f'Creating {self.num_partitions:,} partitions.')
            Run().print_main(f'*Estimated* {int(self.num_embeddings_est):,} embeddings.')

        self._save_plan()

    def _pid_offsets(self):
        """Return a list where ``offsets[pid]`` is the index into self.embeddings
        for the first token of document ``pid``."""
        offsets = [0]
        for dl in self.doclens[:-1]:
            offsets.append(offsets[-1] + dl)
        return offsets

    # ------------------------------------------------------------------
    # train – k-means centroids + residual codec
    # ------------------------------------------------------------------
    def train(self):
        sample, heldout = self._split_sample()
        centroids = self._train_kmeans(sample)

        del sample
        print_memory_stats('after kmeans')

        bucket_cutoffs, bucket_weights, avg_residual = \
            self._compute_avg_residual(centroids, heldout)

        if self.verbose > 1:
            print_message(f'avg_residual = {avg_residual}')

        codec = ResidualCodec(
            config=self.config, centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )
        self.saver.save_codec(codec)

    def _split_sample(self):
        sample = self.sample_embs.half()
        sample = sample[torch.randperm(sample.size(0))]
        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, heldout = sample.split(
            [sample.size(0) - heldout_size, heldout_size], dim=0
        )
        return sample, heldout

    def _train_kmeans(self, sample):
        args_ = [self.config.dim, self.num_partitions, self.config.kmeans_niters]
        args_ = args_ + [[[sample]]]
        centroids = compute_faiss_kmeans(*args_)
        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        if self.use_gpu:
            centroids = centroids.half()
        else:
            centroids = centroids.float()
        return centroids

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodec(config=self.config, centroids=centroids, avg_residual=None)

        device = 'cuda' if self.use_gpu else 'cpu'
        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device=device)
        heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct, out_device=device)

        if self.use_gpu:
            heldout_avg_residual = heldout.cuda() - heldout_reconstruct
        else:
            heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()

        num_options = 2 ** self.config.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles = quantiles[1:]
        bucket_weights_quantiles = quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        if self.verbose > 2:
            print_message(
                f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} "
                f"and bucket_weights_quantiles = {bucket_weights_quantiles}"
            )
            print_message(
                f"#> Got bucket_cutoffs = {bucket_cutoffs} "
                f"and bucket_weights = {bucket_weights}"
            )

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

    # ------------------------------------------------------------------
    # index – compress and save all embeddings in chunks
    # ------------------------------------------------------------------
    def index(self):
        with self.saver.thread():
            offset_pid = 0     # document offset for this chunk
            emb_offset = 0     # embedding offset into self.embeddings

            for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
                # determine which documents belong to this chunk
                end_pid = min(offset_pid + self.chunksize, self.num_documents)
                chunk_doclens = self.doclens[offset_pid:end_pid]
                num_chunk_embs = sum(chunk_doclens)

                embs = self.embeddings[emb_offset:emb_offset + num_chunk_embs]

                if self.use_gpu:
                    assert embs.dtype == torch.float16 or embs.dtype == torch.float32
                embs = embs.half()

                if self.verbose > 1:
                    Run().print_main(
                        f"#> Saving chunk {chunk_idx}: \t {len(chunk_doclens):,} passages "
                        f"and {embs.size(0):,} embeddings. From #{offset_pid:,} onward."
                    )

                self.saver.save_chunk(chunk_idx, offset_pid, embs, chunk_doclens)

                offset_pid = end_pid
                emb_offset += num_chunk_embs

            assert emb_offset == self.num_embeddings, (emb_offset, self.num_embeddings)

    # ------------------------------------------------------------------
    # finalize – build IVF and metadata
    # ------------------------------------------------------------------
    def finalize(self):
        self._check_all_files_are_saved()
        self._collect_embedding_id_offset()
        self._build_ivf()
        self._update_metadata()

    def _check_all_files_are_saved(self):
        if self.verbose > 1:
            Run().print_main("#> Checking all files were saved...")
        success = True
        for chunk_idx in range(self.num_chunks):
            if not self.saver.check_chunk_exists(chunk_idx):
                success = False
                Run().print_main(f"#> ERROR: Could not find chunk {chunk_idx}!")
        if success and self.verbose > 1:
            Run().print_main("Found all files!")

    def _collect_embedding_id_offset(self):
        passage_offset = 0
        embedding_offset = 0
        self.embedding_offsets = []

        for chunk_idx in range(self.num_chunks):
            metadata_path = os.path.join(
                self.config.index_path_, f'{chunk_idx}.metadata.json'
            )
            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)
                chunk_metadata['embedding_offset'] = embedding_offset
                self.embedding_offsets.append(embedding_offset)

                assert chunk_metadata['passage_offset'] == passage_offset, \
                    (chunk_idx, passage_offset, chunk_metadata)

                passage_offset += chunk_metadata['num_passages']
                embedding_offset += chunk_metadata['num_embeddings']

            with open(metadata_path, 'w') as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + '\n')

        self.total_num_embeddings = embedding_offset
        assert len(self.embedding_offsets) == self.num_chunks

    def _build_ivf(self):
        if self.verbose > 1:
            Run().print_main("#> Building IVF...")

        codes = torch.zeros(self.total_num_embeddings,).long()

        if self.verbose > 1:
            Run().print_main("#> Loading codes...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(
                self.config.index_path_, chunk_idx
            )
            codes[offset:offset + chunk_codes.size(0)] = chunk_codes

        if self.verbose > 1:
            Run().print_main("Sorting codes...")

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        if self.verbose > 1:
            Run().print_main("Getting unique codes...")

        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        _, _ = optimize_ivf(ivf, ivf_lengths, self.config.index_path_)

    def _update_metadata(self):
        self.metadata_path = os.path.join(
            self.config.index_path_, 'metadata.json'
        )
        if self.verbose > 1:
            Run().print("#> Saving the indexing metadata to", self.metadata_path, "..")

        with open(self.metadata_path, 'w') as f:
            d = {'config': self.config.export()}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings'] = self.total_num_embeddings
            d['avg_doclen'] = self.total_num_embeddings / self.num_documents
            f.write(ujson.dumps(d, indent=4) + '\n')

    # ------------------------------------------------------------------
    # plan persistence
    # ------------------------------------------------------------------
    def _save_plan(self):
        self.plan_path = os.path.join(self.config.index_path_, 'plan.json')
        Run().print_main("#> Saving the indexing plan to", self.plan_path, "..")

        with open(self.plan_path, 'w') as f:
            d = {'config': self.config.export()}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings_est'] = self.num_embeddings_est
            d['avg_doclen_est'] = self.avg_doclen
            f.write(ujson.dumps(d, indent=4) + '\n')
