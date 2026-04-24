import os
import time

import torch.multiprocessing as mp

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import Launcher

from colbert.utils.utils import create_directory, print_message

from colbert.indexing.collection_indexer_generate import CollectionIndexerGenerate


class IndexerGenerate:
    def __init__(self):
        pass

    def index(self, username:str, dataset:str):
        encoder = CollectionIndexerGenerate(username=username, dataset=dataset)
        build_index_time, encode_passage_time = encoder.run()

        return build_index_time, encode_passage_time

