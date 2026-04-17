#include "arg_utils.hpp"
#include <omp.h>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "index.hpp"
#include "io.hpp"

#include "utils.hpp"

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --data <embeddings.bin>\n"
        << "  --query <query_embeddings.bin>\n"
        << "  --doclens <doclens.bin>\n"
        << "  --index <index_dir|doc_1bit.bin>\n"
        << "  --output <dist_decomp.csv>\n"
        << "  [--nq <num_queries_to_run>]\n"
        << "  [--nprobe <num_probes>]\n"
        << "  [--top_k_out <num_rows_per_query>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int k = 100;
    int nq = 100;
    int nprobe = 128;
    int top_k_out = 1000;
    std::string data_file;
    std::string query_file;
    std::string doclens_file;
    std::string index_file;
    std::string output_file;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--data") {
                data_file = require_value(argc, argv, i, arg);
            } else if (arg == "--query" || arg == "-q") {
                query_file = require_value(argc, argv, i, arg);
            } else if (arg == "--doclens" || arg == "-d") {
                doclens_file = require_value(argc, argv, i, arg);
            } else if (arg == "--index" || arg == "-i") {
                index_file = require_value(argc, argv, i, arg);
            } else if (arg == "--output" || arg == "-o") {
                output_file = require_value(argc, argv, i, arg);
            } else if (arg == "--nq") {
                nq = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--nprobe") {
                nprobe = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--top_k_out") {
                top_k_out = std::stoi(require_value(argc, argv, i, arg));
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

    if (data_file.empty() || query_file.empty() || doclens_file.empty() ||
        index_file.empty() || output_file.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> dataset = load_data(num_d, d, data_file);
    std::vector<float> Q = load_query(q_doclen, num_q, d, query_file);
    std::vector<int> doclens = load_doclens(doclens_file);
    cpu_mvr_index index(index_file);
    index.set_doc_mapping(doclens);

    const int run_queries = std::min<int>(nq, static_cast<int>(num_q));
    std::vector<std::vector<size_t>> results(run_queries);
    std::vector<std::vector<float>> est_dists(run_queries);
    std::vector<std::vector<float>> true_dists(run_queries);
    for (int i = 0; i < run_queries; ++i) {
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
    std::ofstream csv(output_file);
    csv << "query_id,rank,doc_id,est_dist,true_dist\n";
    for (int i = 0; i < run_queries; ++i) {
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
    std::cout << "Wrote " << output_file
              << " (top-" << top_k_out << " per query)" << std::endl;

    return 0;
}
