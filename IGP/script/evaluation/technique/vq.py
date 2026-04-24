import numpy as np
import torch
import os
import tqdm
import sys
import random
import faiss

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.data import dataset_io, util


def sample_itemID4kmeans(n_item):
    # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
    # Keep in mind that, say, 15% still means at least 100k.
    # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
    # Then we subsample the vectors to 100 * num_partitions

    typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
    n_sample_pid = 8 * np.sqrt(typical_doclen * n_item)
    # n_sample_pid = np.sqrt(typical_doclen * n_item) / 2
    # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
    n_sample_pid = min(1 + int(n_sample_pid), n_item)

    random.seed(12345)
    sample_pid_l = random.sample(range(n_item), n_sample_pid)

    return sample_pid_l


def get_sample_vecs_l(sample_itemID_l: list, DEFAULT_CHUNKSIZE: int, username: str, dataset: str, vec_dim: int):
    del DEFAULT_CHUNKSIZE
    sample_itemID_l = np.sort(np.array(sample_itemID_l, dtype=np.uint64))
    item_n_vecs_l = dataset_io.load_doclens(username=username, dataset=dataset).astype(np.uint64)
    item_n_vecs_offset_l = np.empty(item_n_vecs_l.shape[0] + 1, dtype=np.uint64)
    item_n_vecs_offset_l[0] = 0
    np.cumsum(item_n_vecs_l, out=item_n_vecs_offset_l[1:])

    sample_item_n_vec_l = item_n_vecs_l[sample_itemID_l].astype(np.uint32, copy=True)
    total_sample_vec = int(np.sum(sample_item_n_vec_l, dtype=np.uint64))
    sample_vecs_l = np.empty((total_sample_vec, vec_dim), dtype=np.float32)
    vecsID_l = np.empty(total_sample_vec, dtype=np.uint64)

    if dataset_io.dataset_mode(dataset) == "legacy":
        data = dataset_io.load_all_embeddings(username=username, dataset=dataset)
        offset = 0
        for itemID, item_n_vec in zip(sample_itemID_l, sample_item_n_vec_l):
            start = int(item_n_vecs_offset_l[itemID])
            end = int(item_n_vecs_offset_l[itemID + 1])
            sample_vecs_l[offset:offset + item_n_vec] = data[start:end]
            vecsID_l[offset:offset + item_n_vec] = np.arange(start, end, dtype=np.uint64)
            offset += int(item_n_vec)
        return sample_vecs_l, sample_item_n_vec_l, vecsID_l

    data_mmap = dataset_io.load_embedding_memmap(username=username, dataset=dataset)
    output_offset = 0
    run_start = 0
    n_sample_item = int(sample_itemID_l.shape[0])
    while run_start < n_sample_item:
        run_end = run_start + 1
        while run_end < n_sample_item and sample_itemID_l[run_end] == sample_itemID_l[run_end - 1] + 1:
            run_end += 1

        start_item = int(sample_itemID_l[run_start])
        end_item = int(sample_itemID_l[run_end - 1])
        start_vec = int(item_n_vecs_offset_l[start_item])
        end_vec = int(item_n_vecs_offset_l[end_item + 1])
        run_n_vec = end_vec - start_vec

        sample_vecs_l[output_offset:output_offset + run_n_vec] = np.asarray(
            data_mmap[start_vec:end_vec], dtype=np.float32
        )
        vecsID_l[output_offset:output_offset + run_n_vec] = np.arange(start_vec, end_vec, dtype=np.uint64)
        output_offset += run_n_vec
        run_start = run_end

    assert output_offset == total_sample_vec, (
        f"sample copy mismatch: copied {output_offset} vectors, expected {total_sample_vec}"
    )
    assert len(vecsID_l) == len(sample_vecs_l), (
        f"len(vecsID_l) {len(vecsID_l)}, len(sample_vecs_l) {len(sample_vecs_l)}"
    )
    return sample_vecs_l, sample_item_n_vec_l, vecsID_l


def compress_into_codes(embs: np.ndarray, centroid_l: np.ndarray):
    codes = []

    centroid_l = torch.Tensor(centroid_l).to('cuda')
    # print(centroid_l)
    bsize = (1 << 29) // centroid_l.shape[0]
    embs = torch.from_numpy(embs)
    for batch in embs.split(bsize):
        indices = (centroid_l.to('cuda') @ batch.T.to('cuda').float()).max(dim=0).indices
        indices = indices.to('cpu')
        codes.append(indices)

    return torch.cat(codes)


def item_code_in_chunk(code_l: np.ndarray, itemlen_l: np.ndarray, itemID: int):
    vecs_start_idx = int(np.sum(itemlen_l[:itemID]))
    n_item_vecs = int(itemlen_l[itemID])
    item_code = code_l[vecs_start_idx: vecs_start_idx + n_item_vecs]
    return item_code


def sample_vector(username: str, dataset: str):
    vec_dim = dataset_io.embedding_dim(username=username, dataset=dataset)
    item_n_vec_l = dataset_io.load_doclens(username=username, dataset=dataset).astype(np.uint32)
    n_item = item_n_vec_l.shape[0]

    print("sample itemID for kmeans")
    sample_itemID_l = sample_itemID4kmeans(n_item=n_item)
    DEFAULT_CHUNKSIZE = 0

    print("read sample vector from disk")
    sample_vecs_l, sample_item_n_vec_l, _ = get_sample_vecs_l(sample_itemID_l=sample_itemID_l,
                                                              DEFAULT_CHUNKSIZE=DEFAULT_CHUNKSIZE,
                                                              username=username, dataset=dataset, vec_dim=vec_dim)
    return sample_vecs_l, sample_item_n_vec_l


def faiss_kmeans(sample_vecs_l: np.ndarray, n_centroid: int):
    print("build kmeans")
    vec_dim = sample_vecs_l.shape[1]
    kmeans = faiss.Kmeans(vec_dim, n_centroid, niter=20, gpu=True, verbose=True, seed=123)
    kmeans.train(sample_vecs_l)

    centroids = torch.from_numpy(kmeans.centroids)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    centroids = centroids.float()
    return centroids.numpy()


def compute_assignment(username: str, dataset: str, centroid_l: np.ndarray):
    n_centroid = len(centroid_l)
    code_l = torch.Tensor([])
    centroid2itemID_ivf = []
    for centroidID in range(n_centroid):
        centroid2itemID_ivf.append([])

    print("compute assignment of centroid vector")
    accu_itemID = 0
    for itemlen_l_chunk, item_vecs_l_chunk in tqdm.tqdm(
        dataset_io.iter_embedding_chunks(username=username, dataset=dataset)
    ):
        n_item_chunk = itemlen_l_chunk.shape[0]

        code_l_chunk = compress_into_codes(embs=item_vecs_l_chunk, centroid_l=centroid_l)
        for itemID_chunk in range(n_item_chunk):
            itemID = accu_itemID + itemID_chunk
            item_code_l = item_code_in_chunk(np.array(code_l_chunk), itemlen_l_chunk, itemID_chunk)
            for code in np.unique(item_code_l):
                assert 0 <= code < n_centroid
                centroid2itemID_ivf[code].append(itemID)
        code_l = torch.cat([code_l, code_l_chunk])
        accu_itemID += n_item_chunk

    cluster2itemID_l = np.array([itemID for itemID_l in centroid2itemID_ivf for itemID in itemID_l], dtype=np.uint32)
    cluster_n_item_l = np.array([len(itemID_l) for itemID_l in centroid2itemID_ivf], dtype=np.uint32)
    code_l = np.array(code_l, dtype=np.uint32)
    return code_l, cluster2itemID_l, cluster_n_item_l


def vq_ivf(username: str, dataset: str, n_centroid: int):
    sample_vecs_l, sample_item_n_vec_l = sample_vector(username=username, dataset=dataset)

    centroid_l = faiss_kmeans(sample_vecs_l=sample_vecs_l, n_centroid=n_centroid)

    code_l, cluster2itemID_l, cluster_n_item_l = compute_assignment(username=username, dataset=dataset,
                                                                    centroid_l=centroid_l)
    centroid_l = np.array(centroid_l, dtype=np.float32)

    return centroid_l, code_l, cluster2itemID_l, cluster_n_item_l
