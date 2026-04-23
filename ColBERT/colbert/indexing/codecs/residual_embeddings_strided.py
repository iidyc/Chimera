# from colbert.indexing.codecs.residual import ResidualCodec
import colbert.indexing.codecs.residual_embeddings as residual_embeddings

from colbert.search.strided_tensor import StridedTensor
from colbert.search.profiling import profile_section

class ResidualEmbeddingsStrided:
    def __init__(self, codec, embeddings, doclens):
        self.codec = codec
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals
        self.use_gpu = self.codec.use_gpu

        self.codes_strided = StridedTensor(self.codes, doclens, use_gpu=self.use_gpu, profile_name="codes")
        self.residuals_strided = StridedTensor(self.residuals, doclens, use_gpu=self.use_gpu, profile_name="residuals")

    def lookup_pids(self, passage_ids, out_device='cuda'):
        with profile_section("final.lookup_codes", cuda=self.use_gpu):
            codes_packed, codes_lengths = self.codes_strided.lookup(passage_ids)#.as_packed_tensor()
        with profile_section("final.lookup_residuals", cuda=self.use_gpu):
            residuals_packed, _ = self.residuals_strided.lookup(passage_ids)#.as_packed_tensor()

        with profile_section("final.decompress", cuda=self.use_gpu):
            embeddings_packed = self.codec.decompress(
                residual_embeddings.ResidualEmbeddings(
                    codes_packed,
                    residuals_packed,
                    compressed_embeddings_storage="gpu" if codes_packed.is_cuda else "cpu",
                    keep_cpu_copy=False,
                )
            )

        return embeddings_packed, codes_lengths

    def lookup_codes(self, passage_ids):
        return self.codes_strided.lookup(passage_ids)#.as_packed_tensor()
