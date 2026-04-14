#include "gpu_index_baseline.cuh"
#include "gpu_index_layout.hpp"
#include "io.hpp"
#include "utils.hpp"

int main() {
    int k = 100;
    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d);
    std::vector<int> doclens = load_doclens();
    auto ground_truth = read_gt_tsv(num_q, 1000);

    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        return 1;
    }

    gpu_mvr_index_baseline index(gpu_index_layout::kQuantizedDataFilename, doclens);

    // warmup
    for (size_t i = 0; i < 5; ++i) {
        index.search(&Q[i * Q_DOCLEN * d], k);
    }
    Timer timer;
    timer.tick();
    int nq = 5;
    std::vector<std::vector<size_t>> results(nq);
    for (size_t i = 0; i < nq; ++i) {
        results[i] = index.search(&Q[i * Q_DOCLEN * d], k);
    }
    timer.tuck("Baseline GPU search time for " + std::to_string(nq) + " queries.");

    compute_recall(ground_truth, results, k);

    return 0;
}
