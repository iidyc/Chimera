#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace Chimera {

namespace {

std::string require_value(
    int argc,
    char* argv[],
    int& index,
    const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++index];
}

void validate_search_options(const SearchOptions& options) {
    if (options.nprobe <= 0) {
        throw std::runtime_error("--nprobe must be > 0");
    }
    if (options.k_refine <= 0) {
        throw std::runtime_error("--k-refine must be > 0");
    }
    if (options.k_full_bit <= 0) {
        throw std::runtime_error("--k-full-bit must be > 0");
    }
    if (options.cagra_itopk_size <= 0) {
        throw std::runtime_error("--cagra-itopk-size must be > 0");
    }
    if (options.num_chunks <= 0) {
        throw std::runtime_error("--num-chunks must be > 0");
    }
    if (options.k_full_bit > options.k_refine) {
        throw std::runtime_error("--k-full-bit must not exceed --k-refine");
    }
}

}  // namespace

BuildCliArgs parse_build_args(int argc, char** argv) {
    if (argc == 1) {
        throw HelpRequested {};
    }

    BuildCliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            throw HelpRequested {};
        }
        if (arg == "--index" || arg == "-i") {
            args.index_dir = require_value(argc, argv, i, arg);
        } else if (arg == "--doclens" || arg == "-d") {
            args.doclens_file = require_value(argc, argv, i, arg);
        } else if (arg == "--data") {
            args.data_file = require_value(argc, argv, i, arg);
        } else if (arg == "--n-clusters") {
            args.n_clusters = std::stoull(require_value(argc, argv, i, arg));
        } else if (arg == "--ex-bits") {
            args.ex_bits = std::stoull(require_value(argc, argv, i, arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.index_dir.empty() || args.doclens_file.empty() ||
        args.data_file.empty() || args.n_clusters == 0 || args.ex_bits == 0) {
        throw std::runtime_error("Missing required arguments.");
    }
    return args;
}

SearchCliArgs parse_search_args(int argc, char** argv) {
    if (argc == 1) {
        throw HelpRequested {};
    }

    SearchCliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            throw HelpRequested {};
        }
        if (arg == "--query" || arg == "-q") {
            args.query_file = require_value(argc, argv, i, arg);
        } else if (arg == "--gt" || arg == "-g") {
            args.gt_file = require_value(argc, argv, i, arg);
        } else if (arg == "--index" || arg == "-i") {
            args.index_dir = require_value(argc, argv, i, arg);
        } else if (arg == "--k") {
            args.k = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--nq") {
            args.nq = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--nprobe") {
            args.options.nprobe = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--k-refine") {
            args.options.k_refine = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--k-full-bit") {
            args.options.k_full_bit = std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--cagra-itopk-size") {
            args.options.cagra_itopk_size =
                std::stoi(require_value(argc, argv, i, arg));
        } else if (arg == "--num-chunks") {
            args.options.num_chunks =
                std::stoi(require_value(argc, argv, i, arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.query_file.empty() || args.gt_file.empty() ||
        args.index_dir.empty()) {
        throw std::runtime_error("Missing required arguments.");
    }
    if (!std::filesystem::is_directory(args.index_dir)) {
        throw std::runtime_error(
            "--index must point to a Chimera index directory.");
    }
    if (!std::filesystem::exists(
            std::filesystem::path(args.index_dir) / "doclens.bin")) {
        throw std::runtime_error(
            "Missing doclens.bin in --index.");
    }
    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    validate_search_options(args.options);
    return args;
}

void print_build_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " --index <index_dir> --doclens <doclens> --data <data>"
        << " --n-clusters <count> --ex-bits <count>\n\n"
        << "Arguments:\n"
        << "  --index, -i      Output index directory.\n"
        << "  --doclens, -d    Input document-length file.\n"
        << "  --data           Input token-embedding file.\n"
        << "  --n-clusters     Number of centroids in the CAGRA graph.\n"
        << "  --ex-bits        Residual quantization bits per dimension (1-7).\n";
}

void print_search_help(const char* program) {
    const SearchCliArgs defaults;
    std::cout
        << "Usage: " << program << "\n"
        << "  --query <query_embeddings.bin>\n"
        << "  --gt <groundtruth.tsv>\n"
        << "  --index <index_dir>\n"
        << "  [--k <top_k>]                   Final documents returned. Default: "
        << defaults.k << "\n"
        << "  [--nq <num_queries>]            Evaluation queries; -1 means all. Default: "
        << defaults.nq << "\n"
        << "  [--warmup <num_queries>]        Warmup queries. Default: "
        << defaults.warmup << "\n"
        << "  [--nprobe <num_probes>]         IVF clusters probed per query token. Default: "
        << defaults.options.nprobe << "\n"
        << "  [--k-refine <count>]            Candidates retained after refinement. Default: "
        << defaults.options.k_refine << "\n"
        << "  [--k-full-bit <count>]          Candidates scored with full-bit codes. Default: "
        << defaults.options.k_full_bit << "\n"
        << "  [--cagra-itopk-size <count>]    CAGRA intermediate top-k. Default: "
        << defaults.options.cagra_itopk_size << "\n"
        << "  [--num-chunks <count>]          Collaborative-scoring chunks. Default: "
        << defaults.options.num_chunks << "\n";
}

double percentile_ms(const std::vector<double>& sorted_ms, double pct) {
    if (sorted_ms.empty()) {
        return 0.0;
    }
    const double clamped = std::clamp(pct, 0.0, 100.0);
    const double pos =
        (clamped / 100.0) * static_cast<double>(sorted_ms.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double weight = pos - static_cast<double>(lo);
    return sorted_ms[lo] * (1.0 - weight) + sorted_ms[hi] * weight;
}

QueryLatencySummary summarize_query_latencies_ms(
    const std::vector<double>& latencies_ms) {
    QueryLatencySummary summary;
    if (latencies_ms.empty()) {
        return summary;
    }

    std::vector<double> sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    summary.mean_ms = sum / static_cast<double>(sorted.size());

    double sq_sum = 0.0;
    for (double latency_ms : sorted) {
        const double delta = latency_ms - summary.mean_ms;
        sq_sum += delta * delta;
    }
    summary.stddev_ms = std::sqrt(sq_sum / static_cast<double>(sorted.size()));
    summary.min_ms = sorted.front();
    summary.p50_ms = percentile_ms(sorted, 50.0);
    summary.p90_ms = percentile_ms(sorted, 90.0);
    summary.p95_ms = percentile_ms(sorted, 95.0);
    summary.p99_ms = percentile_ms(sorted, 99.0);
    summary.max_ms = sorted.back();
    return summary;
}

std::vector<std::vector<size_t>> read_gt_tsv(
    int num_queries,
    int top_k,
    const std::string& gt_filename) {
    std::vector<std::vector<size_t>> ground_truth(
        num_queries,
        std::vector<size_t>(top_k, -1));
    std::ifstream file(gt_filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << gt_filename << std::endl;
        return ground_truth;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream stream(line);
        std::string segment;
        std::vector<std::string> fields;
        while (std::getline(stream, segment, '\t')) {
            fields.push_back(segment);
        }
        if (fields.size() < 3) {
            std::cerr << "Warning: Skipping malformed line." << std::endl;
            continue;
        }

        const int query_id = std::stoi(fields[0]);
        const size_t item_id = std::stoull(fields[1]);
        const int rank = std::stoi(fields[2]);
        if (rank - 1 >= top_k || rank - 1 < 0) {
            std::cerr << "Warning: Rank " << rank << " exceeds top_k "
                      << top_k << ". Skipping." << std::endl;
            std::exit(0);
        }
        ground_truth[query_id][rank - 1] = item_id;
    }
    return ground_truth;
}

double compute_recall(
    const std::vector<std::vector<size_t>>& ground_truth,
    const std::vector<std::vector<size_t>>& retrieved,
    int top_k) {
    if (top_k <= 0) {
        return 0.0;
    }

    const int num_queries = static_cast<int>(retrieved.size());
    int total_recall = 0;
    for (int i = 0; i < num_queries; ++i) {
        const auto& gt = ground_truth[i];
        const auto& result = retrieved[i];
        const size_t retrieved_cutoff =
            std::min(result.size(), static_cast<size_t>(top_k));
        const std::unordered_set<size_t> result_set(
            result.begin(),
            result.begin() + retrieved_cutoff);
        const int gt_cutoff =
            std::min<int>(top_k, static_cast<int>(gt.size()));
        for (int j = 0; j < gt_cutoff; ++j) {
            total_recall += result_set.count(gt[j]) != 0;
        }
    }
    return static_cast<double>(total_recall) / (num_queries * top_k);
}

}  // namespace Chimera
