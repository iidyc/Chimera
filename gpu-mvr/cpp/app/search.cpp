#include "arg_utils.hpp"
#include <omp.h>

#include "index.hpp"
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
        << "  [--nprobe <num_probes>]\n"
        << "  [--nq <num_queries_to_run>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int k = 100;
    int nprobe = 128;
    int nq = 5;
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
            } else if (arg == "--nprobe") {
                nprobe = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--nq") {
                nq = std::stoi(require_value(argc, argv, i, arg));
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

    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> Q = load_query(q_doclen, num_q, d, query_file);
    std::vector<int> doclens = load_doclens(doclens_file);
    cpu_mvr_index index(index_file);
    index.set_doc_mapping(doclens);

    // std::vector<float> rotated_q(d);
    // index.rotator_->rotate(&Q[0], rotated_q.data());
    // query_object query_obj(rotated_q.data(), index.padded_dim_, index.ex_bits);
    // float one_bit_dist = distance_one_bit(&query_obj, index.one_bit_code_.data(), index.one_bit_factor_[0], index.padded_dim_);
    // float ex_dist = distance_ex_bits(&query_obj, index.ex_code_.data(), index.ex_bits, index.ip_func_, one_bit_dist, index.one_bit_factor_[0], index.ex_factor_[0], index.padded_dim_);
    // std::cout << "Distance check: " << ex_dist << "\n" << std::flush;

    // return 0;

    Timer timer;
    timer.tick();
    const int run_queries = std::min<int>(nq, static_cast<int>(num_q));
    std::vector<std::vector<size_t>> results(run_queries);
// #pragma omp parallel for
    for (int i = 0; i < run_queries; ++i) {
        results[i] = index.search(&Q[i * q_doclen * d], q_doclen, k, nprobe);
    }
    timer.tuck("Search time for " + std::to_string(run_queries) + " queries.");
    auto ground_truth = read_gt_tsv(static_cast<int>(num_q), 1000, gt_file);
    compute_recall(ground_truth, results, k);

    return 0;
}
