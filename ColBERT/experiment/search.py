import os
import argparse
import csv
import time
import numpy as np
import torch

from colbert import EmbeddingIndexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT


def read_gt_tsv(num_queries, top_k, filename):
    ground_truth = [[-1] * top_k for _ in range(num_queries)]
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    print("Warning: Skipping malformed line.")
                    continue
                qID = int(parts[0])
                itemID = int(parts[1])
                rank = int(parts[2])
                ground_truth[qID][rank - 1] = itemID
    except FileNotFoundError:
        print(f"Error: Could not open file {filename}")
    return ground_truth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root-path", default="/home/yanqichen_umass_edu/work/ColBERT/experiments")
    p.add_argument("--experiment-name", default="4_bit_index")
    p.add_argument("--index-name", default="my_index")
    p.add_argument("--query-path", default="../gpu-mvr/build/query_embeddings.bin")
    p.add_argument("--ground-truth-path", default="lotte-groundtruth-top1000--.tsv")
    p.add_argument("--output-csv", default="search_results.csv")
    def pair(s):
        a, b = s.split(",")
        return (int(a), int(b))
    p.add_argument("--pairs", type=pair, nargs="+", required=True,
                   help="List of ncells,ndocs pairs, e.g. --pairs 1,1000 2,4000 4,8000")
    p.add_argument("--k", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.query_path, "rb") as base_file:
        n, q_doclen, d = torch.from_numpy(np.fromfile(base_file, dtype=np.int32, count=3))
        n, q_doclen, d = np.int64(n), np.int64(q_doclen), np.int64(d)
        print(f"Expecting to read {n} document embeddings of dimension {d} from {base_file.name}")
        query_emb = torch.from_numpy(
            np.fromfile(base_file, dtype=np.float32, count=n * q_doclen * d)
        ).reshape(n, q_doclen, d)
    print(f"Loaded document embeddings: {query_emb.shape}")

    ground_truth = read_gt_tsv(num_queries=n, top_k=1000, filename=args.ground_truth_path)

    with open(args.output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["avg_time", "recall"])

        with Run().context(RunConfig(experiment=args.experiment_name, root=args.root_path)):
            ColBERT.try_load_torch_extensions(False)
            for ncells, ndocs in args.pairs:
                config = ColBERTConfig(ncells=ncells, ndocs=ndocs)
                searcher = Searcher(index=args.index_name, checkpoint=None, config=config)
                num_runs = 3
                run_times = []
                results = []
                for run_idx in range(num_runs):
                    results = []
                    begin_time = time.time()
                    for qid in range(n):
                        pids, ranks, scores = searcher.search_from_embeddings(query_emb[qid], k=args.k)
                        results.append(pids)
                    run_times.append(time.time() - begin_time)
                avg_time = (sum(run_times) / num_runs) / n
                recall = sum(
                    sum(1 for pid in ground_truth[qid][:args.k] if pid in results[qid])
                    for qid in range(n)
                ) / n / args.k
                print(f"ncells={ncells} ndocs={ndocs} avg_time={avg_time:.4f}s recall@{args.k}={recall:.4f}")
                writer.writerow([f"{avg_time:.4f}", f"{recall:.4f}"])
                csv_file.flush()

if __name__ == "__main__":
    main()
