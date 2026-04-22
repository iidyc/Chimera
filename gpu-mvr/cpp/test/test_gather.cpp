#include "arg_utils.hpp"
#include <omp.h>
#include <fstream>

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
        << "  --index <index_dir|doc_1bit.bin>\n"
        << "  --output <gather_recall.csv>\n"
        << "  [--k <top_k>]\n"
        << "  [--nq <num_queries_to_run>]\n"
        << "  [--nprobe <num_probes>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int k = 100;
    int nq = 100;
    int nprobe = 128;
    std::string query_file;
    std::string doclens_file;
    std::string gt_file;
    std::string index_file;
    std::string output_file;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--query" || arg == "-q") {
                query_file = require_value(argc, argv, i, arg);
            } else if (arg == "--doclens" || arg == "-d") {
                doclens_file = require_value(argc, argv, i, arg);
            } else if (arg == "--gt" || arg == "-g") {
                gt_file = require_value(argc, argv, i, arg);
            } else if (arg == "--index" || arg == "-i") {
                index_file = require_value(argc, argv, i, arg);
            } else if (arg == "--output" || arg == "-o") {
                output_file = require_value(argc, argv, i, arg);
            } else if (arg == "--k") {
                k = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--nq") {
                nq = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--nprobe") {
                nprobe = std::stoi(require_value(argc, argv, i, arg));
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

    if (query_file.empty() || doclens_file.empty() || gt_file.empty() ||
        index_file.empty() || output_file.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> Q = load_query(q_doclen, num_q, d, query_file);
    std::vector<int> doclens = load_doclens(doclens_file);
    cpu_mvr_index index(index_file);
    index.set_doc_mapping(doclens);

    const int run_queries = std::min<int>(nq, static_cast<int>(num_q));
    std::ofstream csv(output_file);
    csv << "method,param,avg_cand_size,recall\n";
    auto ground_truth = read_gt_tsv(static_cast<int>(num_q), 1000, gt_file);

    int k_ranks[] = {400, 800, 1600, 3200, 6400, 12800};
    for (int k_rank : k_ranks) {
        std::vector<std::vector<size_t>> results(run_queries);
        for (int i = 0; i < run_queries; ++i) {
            index.gather_ids_rank_dists(&Q[i * q_doclen * d], q_doclen, nprobe, results[i], k_rank);
        }
        int avg_cand_size = 0;
        for (const auto& res : results) {
            avg_cand_size += static_cast<int>(res.size());
        }
        avg_cand_size /= run_queries;
        double recall = compute_recall(ground_truth, results, k);
        std::cout << "Cand size: " << avg_cand_size << " for k = " << k_rank << "\n";
        std::cout << "Recall@" << k << ": " << recall << "\n";
        csv << "rank_dists," << k_rank << "," << avg_cand_size << "," << recall << "\n";
    }

    int nprobes[] = {4, 8, 16, 32, 64, 128};
    for (int nprobe : nprobes) {
        std::vector<std::vector<size_t>> results(run_queries);
        for (int i = 0; i < run_queries; ++i) {
            index.gather_ids(&Q[i * q_doclen * d], q_doclen, nprobe, results[i]);
        }
        int avg_cand_size = 0;
        for (const auto& res : results) {
            avg_cand_size += static_cast<int>(res.size());
        }
        avg_cand_size /= run_queries;
        double recall = compute_recall(ground_truth, results, k);
        std::cout << "Cand size: " << avg_cand_size << " for nprobe = " << nprobe << "\n";
        std::cout << "Recall@" << k << ": " << recall << "\n";
        csv << "gather_ids," << nprobe << "," << avg_cand_size << "," << recall << "\n";
    }

    csv.close();
    std::cout << "Wrote " << output_file << std::endl;

    return 0;
}
