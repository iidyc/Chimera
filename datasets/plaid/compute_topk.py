"""
Compute top-1000 documents per query using ColBERT MaxSim scoring.

Binary formats (from encode_dataset.py / encode_queries.py):
  msmarco_emb.bin:     [total_n:i32, d:i32] + float32[total_n, d]
  msmarco_doclens.bin: [num_docs:i32] + int32[num_docs]
  msmarco_queries.bin: [nq:i32, ql:i32, d:i32] + float32[nq, ql, d]

Usage:
  python compute_topk.py msmarco_emb.bin msmarco_doclens.bin msmarco_queries.bin output.tsv \
      [--doc_batch_size 50000] [--topk 1000]
"""

import argparse
import numpy as np
import torch
import time


def load_queries(path):
    with open(path, "rb") as f:
        nq = np.fromfile(f, dtype=np.int32, count=1)[0]
        ql = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        Q = np.fromfile(f, dtype=np.float32, count=nq * ql * d).reshape(nq, ql, d)
    print(f"Queries: {nq} x {ql} x {d}")
    return Q


def load_doclens(path):
    with open(path, "rb") as f:
        num_docs = np.fromfile(f, dtype=np.int32, count=1)[0]
        doclens = np.fromfile(f, dtype=np.int32, count=num_docs)
    print(f"Doclens: {num_docs} docs, total tokens = {doclens.sum()}")
    return doclens


def mmap_embeddings(path):
    with open(path, "rb") as f:
        total_n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
    header_bytes = 8  # 2 x int32
    emb = np.memmap(path, dtype=np.float32, mode="r",
                    offset=header_bytes, shape=(total_n, d))
    print(f"Embeddings: {total_n} tokens x {d} dims")
    return total_n, d, emb


def pad_and_mask(D_flat, batch_doclens, max_doclen, d, device):
    """Vectorized padding — no Python for-loop over documents."""
    batch_docs = len(batch_doclens)
    doclens_t = torch.from_numpy(batch_doclens.astype(np.int64)).to(device)

    # Build (doc_index, token_position) for every real token
    doc_indices = torch.repeat_interleave(
        torch.arange(batch_docs, device=device), doclens_t)
    tok_positions = torch.cat(
        [torch.arange(dl, device=device) for dl in doclens_t.tolist()])

    D_padded = torch.zeros(batch_docs, max_doclen, d,
                           dtype=torch.float32, device=device)
    D_padded[doc_indices, tok_positions] = D_flat

    mask = torch.arange(max_doclen, device=device).unsqueeze(0) < doclens_t.unsqueeze(1)
    return D_padded, mask


def maxsim_batch(Q_sub, D_padded, mask):
    """
    Q_sub:    (qb, ql, d) float32
    D_padded: (dc, max_dl, d) float32
    mask:     (dc, max_dl) bool
    Returns:  (qb, dc) float32 MaxSim scores
    """
    # (qb, dc, ql, max_dl) in float32
    scores = torch.matmul(
        Q_sub.unsqueeze(1),                          # (qb, 1, ql, d)
        D_padded.permute(0, 2, 1).unsqueeze(0)       # (1, dc, d, max_dl)
    )
    scores.masked_fill_(~mask.view(1, -1, 1, mask.shape[1]), -float("inf"))
    return scores.max(dim=-1).values.sum(dim=-1)


def compute_topk(emb_path, doclens_path, queries_path, output_path,
                 doc_batch_size=100000, topk=1000):
    Q_np = load_queries(queries_path)
    doclens = load_doclens(doclens_path)
    total_n, d, emb = mmap_embeddings(emb_path)

    num_queries, ql = Q_np.shape[0], Q_np.shape[1]
    num_docs = len(doclens)

    doc_offsets = np.zeros(num_docs + 1, dtype=np.int64)
    np.cumsum(doclens, out=doc_offsets[1:])
    assert doc_offsets[-1] == total_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Q_gpu = torch.from_numpy(Q_np).to(device=device)

    top_scores = torch.full((num_queries, topk), -float("inf"),
                            dtype=torch.float32, device=device)
    top_ids = torch.full((num_queries, topk), -1, dtype=torch.int32, device=device)

    # ---- Memory budget for the 4D matmul tensor (float32) ----
    # Target: ≤ 8 GB  =>  qb * dc * ql * max_dl ≤ 8GB / 4 = 2e9 elements
    max_dl_est = int(doclens.max())
    budget_elements = 2e9  # float32 elements that fit in 8 GB
    query_batch = min(32, num_queries)
    doc_chunk = max(1, int(budget_elements / (query_batch * ql * max_dl_est)))
    print(f"query_batch={query_batch}, doc_chunk={doc_chunk}, max_doclen={max_dl_est}")

    t0 = time.time()
    for batch_start in range(0, num_docs, doc_batch_size):
        batch_end = min(batch_start + doc_batch_size, num_docs)

        tok_start = int(doc_offsets[batch_start])
        tok_end = int(doc_offsets[batch_end])
        batch_doclens = doclens[batch_start:batch_end]
        max_doclen = int(batch_doclens.max())
        batch_docs = batch_end - batch_start

        # Load token embeddings: mmap → GPU float32
        batch_emb = np.array(emb[tok_start:tok_end])
        D_flat = torch.from_numpy(batch_emb).to(device=device)
        del batch_emb

        # Vectorized pad + mask
        D_padded, mask = pad_and_mask(D_flat, batch_doclens, max_doclen, d, device)
        del D_flat

        # Pre-transpose once: (bd, d, max_dl) for the matmul
        D_T = D_padded.permute(0, 2, 1).contiguous()
        del D_padded

        # Score all queries against this doc batch
        batch_all_scores = torch.empty(num_queries, batch_docs,
                                       dtype=torch.float32, device=device)

        for qi in range(0, num_queries, query_batch):
            qe = min(qi + query_batch, num_queries)
            Q_sub = Q_gpu[qi:qe]  # (qb, ql, d)

            for di in range(0, batch_docs, doc_chunk):
                de = min(di + doc_chunk, batch_docs)
                # (qb, dc, ql, max_dl)
                scores = torch.matmul(
                    Q_sub.unsqueeze(1),        # (qb, 1, ql, d)
                    D_T[di:de].unsqueeze(0)    # (1, dc, d, max_dl)
                )
                scores.masked_fill_(
                    ~mask[di:de].view(1, de - di, 1, max_doclen), -float("inf"))
                batch_all_scores[qi:qe, di:de] = \
                    scores.max(dim=-1).values.sum(dim=-1)
                del scores

        del D_T, mask

        # Merge with running top-k
        doc_ids_batch = torch.arange(batch_start, batch_end,
                                     dtype=torch.int32, device=device)
        combined_scores = torch.cat([top_scores, batch_all_scores], dim=1)
        combined_ids = torch.cat(
            [top_ids, doc_ids_batch.unsqueeze(0).expand(num_queries, -1)], dim=1)

        _, sel = combined_scores.topk(topk, dim=1)
        top_scores = combined_scores.gather(1, sel)
        top_ids = combined_ids.gather(1, sel)

        del batch_all_scores, combined_scores, combined_ids
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  docs {batch_start}-{batch_end} / {num_docs}  ({elapsed:.1f}s)")

    # Final sort by score descending
    sorted_idx = top_scores.argsort(dim=1, descending=True)
    top_ids = top_ids.gather(1, sorted_idx).cpu().numpy()

    with open(output_path, "w") as f:
        for qi in range(num_queries):
            for rank in range(topk):
                did = int(top_ids[qi, rank])
                if did < 0:
                    break
                f.write(f"{qi}\t{did}\t{rank + 1}\n")

    print(f"Wrote {output_path}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_path")
    parser.add_argument("doclens_path")
    parser.add_argument("queries_path")
    parser.add_argument("output_path")
    parser.add_argument("--doc_batch_size", type=int, default=50000,
                        help="Docs loaded from mmap per outer iteration")
    parser.add_argument("--topk", type=int, default=1000)
    args = parser.parse_args()

    compute_topk(args.emb_path, args.doclens_path, args.queries_path,
                 args.output_path, args.doc_batch_size, args.topk)
