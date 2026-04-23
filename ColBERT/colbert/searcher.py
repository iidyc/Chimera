import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

import time

from colbert.search.profiling import get_active_profiler, profile_section

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, verbose:int = 3):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        # Resolve checkpoint: may be explicitly provided, found in index
        # config, or intentionally None (embedding-only mode).
        resolved_checkpoint = checkpoint or self.index_config.checkpoint

        if resolved_checkpoint is not None:
            # Standard path: load config from checkpoint, build full config,
            # and instantiate the ColBERT model.
            self.checkpoint_config = ColBERTConfig.load_from_checkpoint(resolved_checkpoint)
            self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

            self.collection = Collection.cast(collection or self.config.collection)
            self.configure(checkpoint=resolved_checkpoint, collection=self.collection)

            self.checkpoint = Checkpoint(resolved_checkpoint, colbert_config=self.config, verbose=self.verbose)
            use_gpu = self.config.total_visible_gpus > 0
            if use_gpu:
                self.checkpoint = self.checkpoint.cuda()
        else:
            # Embedding-only mode: no checkpoint model is loaded.
            self.checkpoint_config = None
            self.config = ColBERTConfig.from_existing(self.index_config, initial_config)

            if collection is not None:
                self.collection = Collection.cast(collection)
            elif self.config.collection is not None:
                self.collection = Collection.cast(self.config.collection)
            else:
                self.collection = None

            self.checkpoint = None   # no model

        use_gpu = self.config.total_visible_gpus > 0
        load_index_with_mmap = self.config.load_index_with_mmap
        compressed_embeddings_storage = self.config.compressed_embeddings_storage
        gpu_index_resident = self.config.gpu_index_resident
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        if compressed_embeddings_storage not in {"cpu", "gpu"}:
            raise ValueError(
                "compressed_embeddings_storage must be one of: cpu, gpu"
            )
        if compressed_embeddings_storage == "gpu" and not use_gpu:
            raise ValueError(
                "compressed_embeddings_storage=gpu requires CUDA-enabled search"
            )
        if gpu_index_resident and not use_gpu:
            raise ValueError("gpu_index_resident requires CUDA-enabled search")
        if gpu_index_resident and load_index_with_mmap:
            raise ValueError("gpu_index_resident is incompatible with memory-mapped index loading")
        if gpu_index_resident and compressed_embeddings_storage != "gpu":
            raise ValueError(
                "gpu_index_resident requires compressed_embeddings_storage=gpu"
            )
        self.ranker = IndexScorer(
            self.index,
            use_gpu,
            load_index_with_mmap,
            compressed_embeddings_storage=compressed_embeddings_storage,
            gpu_index_resident=gpu_index_resident,
        )

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def _assert_checkpoint(self):
        if self.checkpoint is None:
            raise RuntimeError(
                "No checkpoint loaded. Use search_from_embeddings() / "
                "search_all_from_embeddings() instead, or initialise the "
                "Searcher with a checkpoint."
            )

    def encode(self, text: TextQueries, full_length_search=False):
        self._assert_checkpoint()
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None):
        self._assert_checkpoint()
        Q = self.encode(text, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False, qid_to_pids=None):
        self._assert_checkpoint()
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids)

    # ------------------------------------------------------------------
    # Embedding-based search (no checkpoint required)
    # ------------------------------------------------------------------

    def search_from_embeddings(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        """Search with a pre-computed query embedding tensor.

        Parameters
        ----------
        Q : torch.Tensor
            Query embeddings of shape ``(1, query_maxlen, dim)`` or
            ``(query_maxlen, dim)``.  Will be reshaped to 3-D if needed.
        k : int
            Number of results to return.
        filter_fn : callable, optional
            Optional filter over candidate passage IDs.
        pids : list[int], optional
            If provided, only these passage IDs are scored.

        Returns
        -------
        tuple
            ``(pids, ranks, scores)`` – same format as :meth:`search`.
        """
        if Q.dim() == 2:
            Q = Q.unsqueeze(0)
        assert Q.dim() == 3, f"Expected Q of shape (1, query_maxlen, dim), got {Q.shape}"
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all_from_embeddings(self, Q: torch.Tensor, qids=None, k=10,
                                    filter_fn=None, qid_to_pids=None):
        """Search with pre-computed query embeddings for multiple queries.

        Parameters
        ----------
        Q : torch.Tensor
            Query embeddings of shape ``(num_queries, query_maxlen, dim)``.
        qids : list, optional
            Query IDs corresponding to each row of *Q*.  If *None*,
            ``range(len(Q))`` is used.
        k : int
            Number of results per query.
        filter_fn : callable, optional
            Optional filter over candidate passage IDs.
        qid_to_pids : dict, optional
            Mapping from ``qid`` to a list of candidate passage IDs.

        Returns
        -------
        Ranking
        """
        assert Q.dim() == 3, f"Expected Q of shape (num_queries, query_maxlen, dim), got {Q.shape}"
        num_queries = Q.shape[0]
        if qids is None:
            qids = list(range(num_queries))
        assert len(qids) == num_queries

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx:query_idx+1],
                        k, filter_fn=filter_fn,
                        pids=qid_to_pids.get(qid),
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        queries_dict = {qid: str(qid) for qid in qids}
        data = {qid: val for qid, val in zip(qids, all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all_from_embeddings'
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx:query_idx+1],
                        k, filter_fn=filter_fn,
                        pids=qid_to_pids[qid]
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None, force_cpu_scoring: bool = False):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        profiler = get_active_profiler()
        if profiler is not None:
            profiler.mark_query()

        with profile_section("search.total", cuda=self.config.total_visible_gpus > 0):
            pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids, force_cpu_scoring=force_cpu_scoring)

        return pids[:k], list(range(1, k+1)), scores[:k]
