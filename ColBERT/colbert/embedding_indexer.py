"""
Top-level API for building a ColBERT index from pre-computed embeddings.

Usage::

    from colbert import EmbeddingIndexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    embeddings = torch.load("doc_embeddings.pt")   # (total_tokens, dim)
    doclens    = torch.load("doclens.pt")           # list[int]

    with Run().context(RunConfig(experiment="my_exp")):
        config = ColBERTConfig(dim=128, nbits=2)
        indexer = EmbeddingIndexer(config=config)
        indexer.index(name="my_index", embeddings=embeddings, doclens=doclens)
"""

import os
import time
import torch

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig

from colbert.utils.utils import create_directory, print_message

from colbert.indexing.embedding_collection_indexer import EmbeddingCollectionIndexer


class EmbeddingIndexer:
    """Build a ColBERT index directly from pre-computed document embeddings.

    Unlike :class:`Indexer`, no ColBERT checkpoint is loaded – embeddings are
    assumed to already be L2-normalised ColBERT token vectors.

    Parameters
    ----------
    config : ColBERTConfig, optional
        Must specify at least ``dim`` and ``nbits``.
    verbose : int
        Verbosity level (0 = silent, 3 = very chatty).
    """

    def __init__(self, config: ColBERTConfig = None, verbose: int = 3):
        self.index_path = None
        self.verbose = verbose
        self.config = ColBERTConfig.from_existing(config, Run().config)

        # Validate essential config values that would normally come from
        # a checkpoint but must be provided explicitly here.
        assert self.config.dim is not None and self.config.dim > 0, \
            "config.dim must be set explicitly when using EmbeddingIndexer (no checkpoint to infer it from)"
        assert self.config.nbits is not None and self.config.nbits > 0, \
            "config.nbits must be set explicitly when using EmbeddingIndexer"

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        """Return the path to the most recently built index, or *None*."""
        return self.index_path

    def erase(self, force_silent: bool = False):
        """Delete artefacts in the current index directory."""
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)

            delete = filepath.endswith(".json")
            delete = delete and (
                'metadata' in filepath or 'doclen' in filepath or 'plan' in filepath
            )
            delete = delete or filepath.endswith(".pt")

            if delete:
                deleted.append(filepath)

        if len(deleted):
            if not force_silent:
                print_message(
                    f"#> Will delete {len(deleted)} files already at "
                    f"{directory} in 20 seconds..."
                )
                time.sleep(20)

            for filepath in deleted:
                os.remove(filepath)

        return deleted

    def index(self, name: str, embeddings: torch.Tensor, doclens,
              overwrite=True):
        """Build a ColBERT index from pre-computed embeddings.

        Parameters
        ----------
        name : str
            Index name (will be created under the configured index root).
        embeddings : torch.Tensor
            All document token embeddings stacked into a single 2-D tensor of
            shape ``(total_tokens, dim)``.  Should be L2-normalised and of the
            same dimension as ``config.dim``.
        doclens : list[int] | torch.Tensor
            Per-document token counts.  ``sum(doclens)`` must equal
            ``embeddings.shape[0]``.
        overwrite : bool | str
            Same semantics as :meth:`Indexer.index`.

        Returns
        -------
        str
            Path to the created index directory.
        """
        assert overwrite in [True, False, 'reuse', 'resume', 'force_silent_overwrite']

        if isinstance(doclens, torch.Tensor):
            doclens = doclens.tolist()

        self.configure(index_name=name)
        self.configure(bsize=64, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = not os.path.exists(self.config.index_path_)

        assert (overwrite in [True, 'reuse', 'resume', 'force_silent_overwrite']) or \
            index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite == 'force_silent_overwrite':
            self.erase(force_silent=True)
        elif overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            encoder = EmbeddingCollectionIndexer(
                config=self.config,
                embeddings=embeddings,
                doclens=doclens,
                verbose=self.verbose,
            )
            encoder.run()

        return self.index_path
