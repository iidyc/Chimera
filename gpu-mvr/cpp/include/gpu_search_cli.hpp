#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include "gpu_search_options.hpp"

struct gpu_search_cli_args {
    int k = 100;
    int nq = -1;
    int warmup = 5;
    std::string query_file;
    std::string doclens_file;
    std::string gt_file;
    std::string index_file;
    gpu_search_runtime_options runtime;
};

inline std::string require_gpu_search_arg_value(int argc, char* argv[], int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++i];
}

inline void print_gpu_search_help(const char* program, const gpu_search_cli_args& defaults) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --query <query_embeddings.bin>\n"
        << "  --doclens <doclens.bin>\n"
        << "  --gt <groundtruth.tsv>\n"
        << "  --index <index_dir|cpu_index.bin>\n"
        << "  [--k <top_k>]                   Final number of documents returned. Default: " << defaults.k << "\n"
        << "  [--nq <num_queries_to_run>]     Number of evaluation queries after warmup; -1 means all. Default: " << defaults.nq << "\n"
        << "  [--warmup <num_warmup_queries>] Warmup queries run before timing/recall. Default: " << defaults.warmup << "\n"
        << "  [--nprobe <num_probes>]         Number of IVF clusters probed per query token. Default: " << defaults.runtime.nprobe << "\n"
        << "  [--k-rank-cluster <count>]      Stage-1 document candidates kept after cluster scoring. Default: " << defaults.runtime.k_rank_cluster << "\n"
        << "  [--k-rank-all-tokens <count>]   Stage-2 candidates kept before final CPU/GPU rerank. Default: " << defaults.runtime.k_rank_all_tokens << "\n"
        << "  [--itopk-size <count>]          Internal CAGRA intermediate top-k during centroid search. Default: " << defaults.runtime.itopk_size << "\n"
        << "  [--overlap-chunks <count>]      Number of overlap chunks in persistent stage-2/3 pipeline; ignored by v0. Default: " << defaults.runtime.overlap_chunks << "\n";
}

inline gpu_search_cli_args parse_gpu_search_args(
    int argc,
    char** argv,
    const gpu_search_cli_args& defaults)
{
    gpu_search_cli_args args = defaults;

    if (argc == 1) {
        throw std::runtime_error("help_requested");
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--query" || arg == "-q") {
            args.query_file = require_gpu_search_arg_value(argc, argv, i, arg);
        } else if (arg == "--doclens" || arg == "-d") {
            args.doclens_file = require_gpu_search_arg_value(argc, argv, i, arg);
        } else if (arg == "--gt" || arg == "-g") {
            args.gt_file = require_gpu_search_arg_value(argc, argv, i, arg);
        } else if (arg == "--index" || arg == "-i") {
            args.index_file = require_gpu_search_arg_value(argc, argv, i, arg);
        } else if (arg == "--k") {
            args.k = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--nq") {
            args.nq = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--nprobe") {
            args.runtime.nprobe = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--k-rank-cluster") {
            args.runtime.k_rank_cluster = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--k-rank-all-tokens") {
            args.runtime.k_rank_all_tokens = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--itopk-size") {
            args.runtime.itopk_size = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--overlap-chunks") {
            args.runtime.overlap_chunks = std::stoi(require_gpu_search_arg_value(argc, argv, i, arg));
        } else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("help_requested");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.query_file.empty() || args.doclens_file.empty() ||
        args.gt_file.empty() || args.index_file.empty()) {
        throw std::runtime_error("Missing required arguments.");
    }
    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    if (args.runtime.nprobe <= 0) {
        throw std::runtime_error("--nprobe must be > 0");
    }
    if (args.runtime.k_rank_cluster <= 0) {
        throw std::runtime_error("--k-rank-cluster must be > 0");
    }
    if (args.runtime.k_rank_all_tokens <= 0) {
        throw std::runtime_error("--k-rank-all-tokens must be > 0");
    }
    if (args.runtime.itopk_size <= 0) {
        throw std::runtime_error("--itopk-size must be > 0");
    }
    if (args.runtime.overlap_chunks <= 0) {
        throw std::runtime_error("--overlap-chunks must be > 0");
    }

    return args;
}
