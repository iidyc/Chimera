#include <omp.h>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "index.hpp"
#include "io.hpp"

#include "utils.hpp"

int main() {
    int k = 100;
    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> dataset = load_data(num_d, d);
    std::vector<float> Q = load_query(q_doclen, num_q, d);
    std::vector<int> doclens = load_doclens();
    cpu_mvr_index index("2097152_4_new.index");
    index.set_doc_mapping(doclens);

    int nq = 100;
    int nprobe = 128;
    std::vector<std::vector<size_t>> results(nq);
    std::vector<std::vector<float>> est_dists(nq);
    std::vector<std::vector<float>> true_dists(nq);
    for (size_t i = 0; i < nq; ++i) {
        index.gather_dists(dataset.data(), &Q[i * q_doclen * d], q_doclen, nprobe, results[i], est_dists[i]);
        size_t n_docs = results[i].size();
        true_dists[i].resize(n_docs);
#pragma omp parallel for schedule(dynamic)
        for (size_t di = 0; di < n_docs; ++di) {
            size_t doc_id = results[i][di];
            float doc_score = 0.0F;
            for (size_t j = 0; j < q_doclen; ++j) {
                float max_dot = -std::numeric_limits<float>::infinity();
                for (size_t kk = 0; kk < index.doc_len(doc_id); ++kk) {
                    size_t idx = index.doc_ptrs_[doc_id] + kk;
                    float dot = rabitqlib::dot_product(&dataset[idx * d], &Q[i * q_doclen * d + j * d], d);
                    max_dot = std::max(max_dot, dot);
                }
                doc_score += max_dot;
            }
            true_dists[i][di] = doc_score;
        }
        std::cout << "Query " << i << ": " << n_docs << " docs found" << std::endl;
    }

    // Write top-1000 docs (ranked by true distance, descending) per query
    int top_k_out = 1000;
    std::ofstream csv("dist_decomp.csv");
    csv << "query_id,rank,doc_id,est_dist,true_dist\n";
    for (size_t i = 0; i < nq; ++i) {
        size_t n_docs = results[i].size();
        // Build index array and sort by true_dist descending
        std::vector<size_t> order(n_docs);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return true_dists[i][a] > true_dists[i][b];
        });
        size_t limit = std::min((size_t)top_k_out, n_docs);
        for (size_t rank = 0; rank < limit; ++rank) {
            size_t di = order[rank];
            csv << i << "," << rank + 1 << "," << results[i][di] << ","
                << est_dists[i][di] << "," << true_dists[i][di] << "\n";
        }
    }
    csv.close();
    std::cout << "Wrote dist_decomp.csv (top-" << top_k_out << " per query)" << std::endl;

    return 0;
}