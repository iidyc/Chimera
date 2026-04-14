#include <omp.h>
#include <fstream>

#include "gpu_index_layout.hpp"
#include "index.hpp"
#include "io.hpp"

#include "utils.hpp"

int main() {
    int k = 100;
    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> Q = load_query(q_doclen, num_q, d);
    std::vector<int> doclens = load_doclens();
    cpu_mvr_index index(gpu_index_layout::kQuantizedDataFilename);
    index.set_doc_mapping(doclens);

    int nq = 100;
    int nprobe = 128;

    std::ofstream csv("gather_recall.csv");
    csv << "method,param,avg_cand_size,recall\n";

    int k_ranks[] = {400, 800, 1600, 3200, 6400, 12800};
    for (int k_rank : k_ranks) {
        std::vector<std::vector<size_t>> results(nq);
        for (size_t i = 0; i < nq; ++i) {
            index.gather_ids_rank_dists(&Q[i * q_doclen * d], q_doclen, nprobe, results[i], k_rank);
        }
        auto ground_truth = read_gt_tsv(num_q, 1000);
        int avg_cand_size = 0;
        for (const auto& res : results) {
            avg_cand_size += res.size();
        }
        avg_cand_size /= nq;
        double recall = compute_recall(ground_truth, results, k);
        std::cout << "Cand size: " << avg_cand_size << " for k = " << k_rank << "\n";
        std::cout << "Recall@" << k << ": " << recall << "\n";
        csv << "rank_dists," << k_rank << "," << avg_cand_size << "," << recall << "\n";
    }

    int nprobes[] = {4, 8, 16, 32, 64, 128};
    for (int nprobe : nprobes) {
        std::vector<std::vector<size_t>> results(nq);
        for (size_t i = 0; i < nq; ++i) {
            index.gather_ids(&Q[i * q_doclen * d], q_doclen, nprobe, results[i]);
        }
        auto ground_truth = read_gt_tsv(num_q, 1000);
        int avg_cand_size = 0;
        for (const auto& res : results) {
            avg_cand_size += res.size();
        }
        avg_cand_size /= nq;
        double recall = compute_recall(ground_truth, results, k);
        std::cout << "Cand size: " << avg_cand_size << " for nprobe = " << nprobe << "\n";
        std::cout << "Recall@" << k << ": " << recall << "\n";
        csv << "gather_ids," << nprobe << "," << avg_cand_size << "," << recall << "\n";
    }

    csv.close();
    std::cout << "Wrote gather_recall.csv" << std::endl;

    return 0;
}
