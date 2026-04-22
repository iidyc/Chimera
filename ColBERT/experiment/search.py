import os
import argparse
import csv
import time
import threading
import subprocess
import shutil
import numpy as np
import torch

from colbert import EmbeddingIndexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT


def bytes_to_mib(value):
    return float(value) / (1024.0 * 1024.0)


class ProcessGpuMemoryMonitor:
    def __init__(self, interval_s=0.2):
        self.pid = os.getpid()
        self.interval_s = interval_s
        self.enabled = shutil.which("nvidia-smi") is not None
        self.current_mib = 0.0
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        if not self.enabled:
            return 0.0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return self.current_mib

        current = 0.0
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used_mib = float(parts[1])
            except ValueError:
                continue
            if pid == self.pid:
                current += used_mib

        self.current_mib = current
        self.peak_mib = max(self.peak_mib, current)
        return current

    def start(self):
        if not self.enabled:
            return

        self._sample_once()

        def loop():
            while not self._stop.wait(self.interval_s):
                self._sample_once()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()


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
    p.add_argument("--warmup", type=int, default=5)
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
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    warmup_queries = min(int(args.warmup), int(n))
    run_queries = int(n)

    ground_truth = read_gt_tsv(num_queries=n, top_k=1000, filename=args.ground_truth_path)

    with open(args.output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "ncells",
            "ndocs",
            "avg_time_s",
            "qps",
            "recall",
            "gpu_mem_current_mib",
            "gpu_mem_peak_mib",
            "gpu_mem_total_mib",
            "torch_peak_allocated_mib",
            "torch_peak_reserved_mib",
        ])

        with Run().context(RunConfig(experiment=args.experiment_name, root=args.root_path)):
            ColBERT.try_load_torch_extensions(False)
            for ncells, ndocs in args.pairs:
                config = ColBERTConfig(ncells=ncells, ndocs=ndocs)
                searcher = Searcher(index=args.index_name, checkpoint=None, config=config)
                num_runs = 3
                run_times = []
                results = []
                gpu_monitor = ProcessGpuMemoryMonitor()
                gpu_mem_total_mib = 0.0
                torch_peak_allocated_mib = 0.0
                torch_peak_reserved_mib = 0.0
                print(
                    f"[RUN] ncells={ncells} ndocs={ndocs} "
                    f"warmup_queries={warmup_queries} eval_queries={run_queries}"
                )
                if torch.cuda.is_available():
                    _, total_bytes = torch.cuda.mem_get_info()
                    gpu_mem_total_mib = bytes_to_mib(total_bytes)
                gpu_monitor.start()
                for run_idx in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    for qid in range(warmup_queries):
                        searcher.search_from_embeddings(query_emb[qid], k=args.k)
                    results = []
                    begin_time = time.time()
                    for qid in range(run_queries):
                        pids, ranks, scores = searcher.search_from_embeddings(query_emb[qid], k=args.k)
                        results.append(pids)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch_peak_allocated_mib = max(
                            torch_peak_allocated_mib,
                            bytes_to_mib(torch.cuda.max_memory_allocated()),
                        )
                        torch_peak_reserved_mib = max(
                            torch_peak_reserved_mib,
                            bytes_to_mib(torch.cuda.max_memory_reserved()),
                        )
                    run_times.append(time.time() - begin_time)
                gpu_monitor.stop()
                avg_time = (sum(run_times) / num_runs) / run_queries
                qps = (1.0 / avg_time) if avg_time > 0.0 else 0.0
                recall = sum(
                    sum(1 for pid in ground_truth[qid][:args.k] if pid in results[qid])
                    for qid in range(run_queries)
                ) / run_queries / args.k
                print(
                    f"[GPU_MEM] current={gpu_monitor.current_mib:.2f} MiB, "
                    f"peak={gpu_monitor.peak_mib:.2f} MiB, "
                    f"total={gpu_mem_total_mib:.2f} MiB, "
                    f"torch_peak_allocated={torch_peak_allocated_mib:.2f} MiB, "
                    f"torch_peak_reserved={torch_peak_reserved_mib:.2f} MiB"
                )
                print(
                    f"ncells={ncells} ndocs={ndocs} avg_time={avg_time:.4f}s "
                    f"recall@{args.k}={recall:.4f} qps={qps:.4f} "
                    f"gpu_mem_current_mib={gpu_monitor.current_mib:.2f} "
                    f"gpu_mem_peak_mib={gpu_monitor.peak_mib:.2f} "
                    f"gpu_mem_total_mib={gpu_mem_total_mib:.2f} "
                    f"torch_peak_allocated_mib={torch_peak_allocated_mib:.2f} "
                    f"torch_peak_reserved_mib={torch_peak_reserved_mib:.2f}"
                )
                writer.writerow([
                    ncells,
                    ndocs,
                    f"{avg_time:.6f}",
                    f"{qps:.6f}",
                    f"{recall:.6f}",
                    f"{gpu_monitor.current_mib:.2f}",
                    f"{gpu_monitor.peak_mib:.2f}",
                    f"{gpu_mem_total_mib:.2f}",
                    f"{torch_peak_allocated_mib:.2f}",
                    f"{torch_peak_reserved_mib:.2f}",
                ])
                csv_file.flush()

if __name__ == "__main__":
    main()
