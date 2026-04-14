#include "gpu_index.cuh"
#include "gpu_index_layout.hpp"
#include "io.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    int k = 100;
    std::string query_file = "query_embeddings.bin";
    std::string doclens_file = "doclens.bin";
    std::string gt_file = "lotte-groundtruth-top1000--.tsv";
    std::string index_file = gpu_index_layout::kQuantizedDataFilename;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--query" || arg == "-q") && i + 1 < argc) {
            query_file = argv[++i];
        } else if ((arg == "--doclens" || arg == "-d") && i + 1 < argc) {
            doclens_file = argv[++i];
        } else if ((arg == "--gt" || arg == "-g") && i + 1 < argc) {
            gt_file = argv[++i];
        } else if ((arg == "--index" || arg == "-i") && i + 1 < argc) {
            index_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--query <file>] [--doclens <file>] [--gt <file>] [--index <index_dir|quantized_data.bin>]" << std::endl;
            return 0;
        }
    }

    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, query_file);
    std::vector<int> doclens = load_doclens(doclens_file);
    auto ground_truth = read_gt_tsv(num_q, 1000, gt_file);

    // Validate that query file matches compiled Q_DOCLEN
    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        std::cerr << "Please recompile with matching Q_DOCLEN in gpu_config.cuh" << std::endl;
        return 1;
    }

    gpu_mvr_index index(index_file, doclens);

    // warmup with 5 queries to stabilize GPU performance
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
    timer.tuck("GPU search time for " + std::to_string(nq) + " queries.");

    compute_recall(ground_truth, results, k);

    return 0;
}
