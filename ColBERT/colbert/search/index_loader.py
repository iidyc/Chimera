import os
import ujson
import torch
import numpy as np
import tqdm

from colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.utils import optimize_ivf
from colbert.search.strided_tensor import StridedTensor


class IndexLoader:
    def __init__(
        self,
        index_path,
        use_gpu=True,
        load_index_with_mmap=False,
        compressed_embeddings_storage="cpu",
        gpu_index_resident=False,
    ):
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap
        if compressed_embeddings_storage not in {"cpu", "gpu"}:
            raise ValueError(
                "compressed_embeddings_storage must be one of: cpu, gpu"
            )
        if compressed_embeddings_storage == "gpu" and not use_gpu:
            raise ValueError(
                "compressed_embeddings_storage=gpu requires use_gpu=True"
            )
        if gpu_index_resident and not use_gpu:
            raise ValueError("gpu_index_resident requires use_gpu=True")
        if gpu_index_resident and load_index_with_mmap:
            raise ValueError(
                "gpu_index_resident is incompatible with load_index_with_mmap=True"
            )
        if gpu_index_resident and compressed_embeddings_storage != "gpu":
            raise ValueError(
                "gpu_index_resident requires compressed_embeddings_storage=gpu"
            )
        self.compressed_embeddings_storage = compressed_embeddings_storage
        self.gpu_index_resident = gpu_index_resident

        self._load_codec()
        self._load_ivf()

        self._load_doclens()
        self._load_embeddings()

    def _load_codec(self):
        print_message(f"#> Loading codec...")
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        if False:
            ivf = ivf.tolist()
            ivf = [ivf[offset:endpos] for offset, endpos in lengths2offsets(ivf_lengths)]
        else:
            # ivf, ivf_lengths = ivf.cuda(), torch.LongTensor(ivf_lengths).cuda()  # FIXME: REMOVE THIS LINE!
            if self.gpu_index_resident:
                print_message("#> Moving IVF to GPU...")
                ivf = ivf.cuda()
                ivf_lengths = torch.as_tensor(ivf_lengths, dtype=torch.long, device=ivf.device)
            ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu, profile_name="ivf")

        self.ivf = ivf

    def _load_doclens(self):
        doclens = []

        print_message("#> Loading doclens...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            with open(os.path.join(self.index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)

        self.doclens_cpu = torch.tensor(doclens)
        self.doclens = self.doclens_cpu.cuda() if self.gpu_index_resident else self.doclens_cpu

    def _load_embeddings(self):
        self.embeddings = ResidualCodec.Embeddings.load_chunks(
                self.index_path,
                range(self.num_chunks),
                self.num_embeddings,
                self.load_index_with_mmap,
                compressed_embeddings_storage=self.compressed_embeddings_storage,
        )

    @property
    def metadata(self):
        try:
            self._metadata
        except:
            with open(os.path.join(self.index_path, 'metadata.json')) as f:
                self._metadata = ujson.load(f)

        return self._metadata

    @property
    def config(self):
        raise NotImplementedError()  # load from dict at metadata['config']

    @property
    def num_chunks(self):
        # EVENTUALLY: If num_chunks doesn't exist (i.e., old index), fall back to counting doclens.*.json files.
        return self.metadata['num_chunks']

    @property
    def num_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata['num_embeddings']
