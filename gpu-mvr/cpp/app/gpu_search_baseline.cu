#include "arg_utils.hpp"
#include "gpu_index_baseline.cuh"
#include "io.hpp"
#include "utils.hpp"

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --query <query_embeddings.bin>\n"
        << "  --doclens <doclens.bin>\n"
        << "  --gt <groundtruth.tsv>\n"
        << "  --index <index_dir|quantized_data.bin>\n"
        << "  [--k <top_k>]\n"
        << "  [--nq <num_queries_to_run>]\n"
        << "  [--warmup <num_warmup_queries>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int k = 100;
    int nq = 5;
    int warmup = 5;
    std::string query_file;
    std::string doclens_file;
    std::string gt_file;
    std::string index_file;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--query" || arg == "-q") {
                query_file = require_value(argc, argv, i, arg);
            } else if (arg == "--doclens" || arg == "-d") {
                doclens_file = require_value(argc, argv, i, arg);
            } else if (arg == "--gt" || arg == "-g") {
                gt_file = require_value(argc, argv, i, arg);
            } else if (arg == "--index" || arg == "-i") {
                index_file = require_value(argc, argv, i, arg);
            } else if (arg == "--k") {
                k = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--nq") {
                nq = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--warmup") {
                warmup = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--help" || arg == "-h") {
                print_input_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    if (query_file.empty() || doclens_file.empty() || gt_file.empty() || index_file.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, query_file);
    std::vector<int> doclens = load_doclens(doclens_file);
    auto ground_truth = read_gt_tsv(static_cast<int>(num_q), 1000, gt_file);

    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        return 1;
    }

    gpu_mvr_index_baseline index(index_file, doclens);

    const int warmup_queries = std::min<int>(warmup, static_cast<int>(num_q));
    for (int i = 0; i < warmup_queries; ++i) {
        index.search(&Q[i * Q_DOCLEN * d], k);
    }
    Timer timer;
    timer.tick();
    const int run_queries = std::min<int>(nq, static_cast<int>(num_q));
    std::vector<std::vector<size_t>> results(run_queries);
    for (int i = 0; i < run_queries; ++i) {
        results[i] = index.search(&Q[i * Q_DOCLEN * d], k);
    }
    timer.tuck("Baseline GPU search time for " + std::to_string(run_queries) + " queries.");

    compute_recall(ground_truth, results, k);

    return 0;
}
