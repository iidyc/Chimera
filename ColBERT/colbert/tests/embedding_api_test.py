"""
Tests for the embedding-based indexing and search APIs.

These tests verify that:
1. EmbeddingIndexer can build an index from pre-computed embeddings
2. Searcher can load that index without a checkpoint and search with embedding tensors
3. Text-based methods raise RuntimeError when no checkpoint is loaded
4. Input validation works (mismatched doclens, wrong shapes, etc.)
"""

import os
import sys
import shutil
import torch
import numpy as np
import pytest

# Ensure colbert package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from colbert import EmbeddingIndexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 128
NBITS = 2
NUM_DOCS = 200
AVG_DOCLEN = 30  # average tokens per doc
QUERY_MAXLEN = 32

TEST_ROOT = os.path.join(os.path.dirname(__file__), '..', 'experiments', '_embedding_test')
INDEX_NAME = 'test_emb_index'


def _generate_embeddings(num_docs=NUM_DOCS, avg_doclen=AVG_DOCLEN, dim=DIM):
    """Return (embeddings, doclens) with random normalised vectors."""
    rng = np.random.RandomState(42)
    doclens = rng.randint(max(1, avg_doclen - 10), avg_doclen + 10, size=num_docs).tolist()
    total_tokens = sum(doclens)
    embs = torch.randn(total_tokens, dim)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs, doclens


def _generate_queries(num_queries=5, query_maxlen=QUERY_MAXLEN, dim=DIM):
    """Return a random query tensor of shape (num_queries, query_maxlen, dim)."""
    Q = torch.randn(num_queries, query_maxlen, dim)
    Q = torch.nn.functional.normalize(Q, dim=-1)
    return Q


def _cleanup():
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingIndexer:

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_index_builds_successfully(self):
        """EmbeddingIndexer.index() should create a valid ColBERT index directory."""
        embs, doclens = _generate_embeddings()

        with Run().context(RunConfig(nranks=1, experiment='_embedding_test',
                                     root=os.path.join(os.path.dirname(__file__), '..', 'experiments'))):
            config = ColBERTConfig(dim=DIM, nbits=NBITS, kmeans_niters=4)
            indexer = EmbeddingIndexer(config=config, verbose=1)
            index_path = indexer.index(name=INDEX_NAME, embeddings=embs,
                                       doclens=doclens, overwrite=True)

        assert os.path.isdir(index_path), f"Index directory not created: {index_path}"

        # Check essential files
        for fname in ['plan.json', 'metadata.json', 'centroids.pt',
                       'avg_residual.pt', 'buckets.pt', 'ivf.pid.pt']:
            fpath = os.path.join(index_path, fname)
            assert os.path.exists(fpath), f"Missing index file: {fname}"

        # Check chunk files exist
        assert os.path.exists(os.path.join(index_path, '0.codes.pt'))
        assert os.path.exists(os.path.join(index_path, '0.residuals.pt'))
        assert os.path.exists(os.path.join(index_path, 'doclens.0.json'))

        print(f"[PASS] Index built at {index_path}")

    def test_doclens_mismatch_raises(self):
        """sum(doclens) != embeddings.shape[0] should raise AssertionError."""
        embs, doclens = _generate_embeddings()
        doclens_bad = doclens[:-5]  # intentionally wrong

        with Run().context(RunConfig(nranks=1, experiment='_embedding_test',
                                     root=os.path.join(os.path.dirname(__file__), '..', 'experiments'))):
            config = ColBERTConfig(dim=DIM, nbits=NBITS, kmeans_niters=4)
            indexer = EmbeddingIndexer(config=config, verbose=0)
            try:
                indexer.index(name='bad_index', embeddings=embs,
                              doclens=doclens_bad, overwrite=True)
                assert False, "Should have raised"
            except AssertionError:
                print("[PASS] Correctly raised AssertionError for doclens mismatch")

    def test_dim_mismatch_raises(self):
        """embeddings.shape[1] != config.dim should raise AssertionError."""
        embs, doclens = _generate_embeddings(dim=64)

        with Run().context(RunConfig(nranks=1, experiment='_embedding_test',
                                     root=os.path.join(os.path.dirname(__file__), '..', 'experiments'))):
            config = ColBERTConfig(dim=DIM, nbits=NBITS)
            indexer = EmbeddingIndexer(config=config, verbose=0)
            try:
                indexer.index(name='bad_dim_index', embeddings=embs,
                              doclens=doclens, overwrite=True)
                assert False, "Should have raised"
            except AssertionError:
                print("[PASS] Correctly raised AssertionError for dim mismatch")


class TestSearcherFromEmbeddings:

    @classmethod
    def setup_class(cls):
        """Build an index to search against."""
        _cleanup()
        cls.embs, cls.doclens = _generate_embeddings()

        cls.exp_root = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        with Run().context(RunConfig(nranks=1, experiment='_embedding_test',
                                     root=cls.exp_root)):
            config = ColBERTConfig(dim=DIM, nbits=NBITS, kmeans_niters=4)
            indexer = EmbeddingIndexer(config=config, verbose=1)
            cls.index_path = indexer.index(name=INDEX_NAME, embeddings=cls.embs,
                                            doclens=cls.doclens, overwrite=True)

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def _get_searcher(self):
        """Return a Searcher without a checkpoint."""
        with Run().context(RunConfig(nranks=1, experiment='_embedding_test',
                                     root=self.exp_root)):
            searcher = Searcher(index=INDEX_NAME, checkpoint=None, verbose=1)
        return searcher

    def test_search_from_embeddings_single(self):
        """search_from_embeddings with a single query should return results."""
        searcher = self._get_searcher()
        Q = _generate_queries(num_queries=1)
        pids, ranks, scores = searcher.search_from_embeddings(Q[0], k=5)

        assert len(pids) == 5, f"Expected 5 pids, got {len(pids)}"
        assert len(scores) == 5
        assert ranks == list(range(1, 6))
        assert all(isinstance(p, int) for p in pids)
        assert all(0 <= p < NUM_DOCS for p in pids)
        print(f"[PASS] search_from_embeddings returned pids={pids[:3]}... scores={scores[:3]}")

    def test_search_from_embeddings_3d(self):
        """search_from_embeddings should also accept (1, qlen, dim) tensors."""
        searcher = self._get_searcher()
        Q = _generate_queries(num_queries=1)  # (1, 32, 128)
        pids, ranks, scores = searcher.search_from_embeddings(Q, k=3)

        assert len(pids) == 3
        print(f"[PASS] 3-D input works: pids={pids}")

    def test_search_all_from_embeddings(self):
        """search_all_from_embeddings should return a Ranking object."""
        searcher = self._get_searcher()
        Q = _generate_queries(num_queries=3)
        qids = [100, 200, 300]
        ranking = searcher.search_all_from_embeddings(Q, qids=qids, k=5)

        assert hasattr(ranking, 'data')
        for qid in qids:
            assert qid in ranking.data, f"qid {qid} missing from ranking"
        print(f"[PASS] search_all_from_embeddings returned ranking for qids {qids}")

    def test_text_search_raises_without_checkpoint(self):
        """search() should raise RuntimeError when checkpoint is None."""
        searcher = self._get_searcher()
        try:
            searcher.search("some query text", k=5)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No checkpoint loaded" in str(e)
            print(f"[PASS] search() correctly raised: {e}")

    def test_encode_raises_without_checkpoint(self):
        """encode() should raise RuntimeError when checkpoint is None."""
        searcher = self._get_searcher()
        try:
            searcher.encode("some query text")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No checkpoint loaded" in str(e)
            print(f"[PASS] encode() correctly raised: {e}")


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Running EmbeddingIndexer tests")
    print("=" * 60)

    t = TestEmbeddingIndexer()
    t.setup_class()
    try:
        t.test_index_builds_successfully()
        t.test_doclens_mismatch_raises()
        t.test_dim_mismatch_raises()
    finally:
        t.teardown_class()

    print()
    print("=" * 60)
    print("Running Searcher-from-embeddings tests")
    print("=" * 60)

    t2 = TestSearcherFromEmbeddings()
    t2.setup_class()
    try:
        t2.test_search_from_embeddings_single()
        t2.test_search_from_embeddings_3d()
        t2.test_search_all_from_embeddings()
        t2.test_text_search_raises_without_checkpoint()
        t2.test_encode_raises_without_checkpoint()
    finally:
        t2.teardown_class()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
