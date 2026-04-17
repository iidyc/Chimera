#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#include "gpu_index_layout.hpp"
#include "gpu_search_options.hpp"

struct gpu_search_named_runtime_config {
    std::string label;
    gpu_search_runtime_options runtime;
};

struct gpu_search_cli_args {
    int k = 100;
    int nq = -1;
    int warmup = 5;
    std::string query_file;
    std::string doclens_file;
    std::string gt_file;
    std::string index_file;
    std::string config_file;
    gpu_search_runtime_options runtime;
};

inline std::string require_gpu_search_arg_value(int argc, char* argv[], int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++i];
}

inline std::string trim_gpu_search_csv_field(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

inline std::vector<std::string> split_gpu_search_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim_gpu_search_csv_field(field));
    }
    return fields;
}

inline void validate_gpu_search_runtime_options(const gpu_search_runtime_options& runtime) {
    if (runtime.nprobe <= 0) {
        throw std::runtime_error("--nprobe must be > 0");
    }
    if (runtime.k_rank_cluster <= 0) {
        throw std::runtime_error("--k-rank-cluster must be > 0");
    }
    if (runtime.k_rank_all_tokens <= 0) {
        throw std::runtime_error("--k-rank-all-tokens must be > 0");
    }
    if (runtime.itopk_size <= 0) {
        throw std::runtime_error("--itopk-size must be > 0");
    }
    if (runtime.overlap_chunks <= 0) {
        throw std::runtime_error("--overlap-chunks must be > 0");
    }
}

inline std::vector<gpu_search_named_runtime_config> load_gpu_search_runtime_configs(const gpu_search_cli_args& args) {
    if (args.config_file.empty()) {
        return {gpu_search_named_runtime_config{"single", args.runtime}};
    }

    std::ifstream file(args.config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config CSV: " + args.config_file);
    }

    std::vector<gpu_search_named_runtime_config> configs;
    std::string line;
    while (std::getline(file, line)) {
        const auto fields = split_gpu_search_csv_line(line);
        if (fields.empty()) {
            continue;
        }
        if (fields[0].empty() || fields[0] == "label" || fields[0][0] == '#') {
            continue;
        }
        if (fields.size() != 6) {
            throw std::runtime_error(
                "Expected 6 CSV columns in " + args.config_file +
                " (label,nprobe,k_rank_cluster,k_rank_all_tokens,itopk_size,overlap_chunks)");
        }

        gpu_search_named_runtime_config config;
        config.label = fields[0];
        config.runtime.nprobe = std::stoi(fields[1]);
        config.runtime.k_rank_cluster = std::stoi(fields[2]);
        config.runtime.k_rank_all_tokens = std::stoi(fields[3]);
        config.runtime.itopk_size = std::stoi(fields[4]);
        config.runtime.overlap_chunks = std::stoi(fields[5]);
        validate_gpu_search_runtime_options(config.runtime);
        configs.push_back(config);
    }

    if (configs.empty()) {
        throw std::runtime_error("No runtime configurations loaded from " + args.config_file);
    }

    return configs;
}

inline gpu_search_runtime_options max_gpu_search_runtime_options(
    const std::vector<gpu_search_named_runtime_config>& configs)
{
    if (configs.empty()) {
        throw std::runtime_error("No runtime configurations available");
    }

    gpu_search_runtime_options max_runtime = configs.front().runtime;
    for (const auto& config : configs) {
        max_runtime.nprobe = std::max(max_runtime.nprobe, config.runtime.nprobe);
        max_runtime.k_rank_cluster = std::max(max_runtime.k_rank_cluster, config.runtime.k_rank_cluster);
        max_runtime.k_rank_all_tokens = std::max(max_runtime.k_rank_all_tokens, config.runtime.k_rank_all_tokens);
        max_runtime.itopk_size = std::max(max_runtime.itopk_size, config.runtime.itopk_size);
        max_runtime.overlap_chunks = std::max(max_runtime.overlap_chunks, config.runtime.overlap_chunks);
    }
    return max_runtime;
}

inline void print_gpu_search_runtime_config_banner(const gpu_search_named_runtime_config& config) {
    std::cout
        << "[CONFIG] label=" << config.label
        << " nprobe=" << config.runtime.nprobe
        << " k_rank_cluster=" << config.runtime.k_rank_cluster
        << " k_rank_all_tokens=" << config.runtime.k_rank_all_tokens
        << " itopk_size=" << config.runtime.itopk_size
        << " overlap_chunks=" << config.runtime.overlap_chunks
        << std::endl;
}

inline void print_gpu_search_help(const char* program, const gpu_search_cli_args& defaults) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --query <query_embeddings.bin>\n"
        << "  [--doclens <doclens.bin>]       Optional when <index>/doclens.bin exists.\n"
        << "  --gt <groundtruth.tsv>\n"
        << "  --index <index_dir|doc_1bit.bin>\n"
        << "  [--config-file <gpu_mvr_config.csv>] Benchmark multiple runtime configs in one process.\n"
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
        } else if (arg == "--config-file") {
            args.config_file = require_gpu_search_arg_value(argc, argv, i, arg);
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

    if (args.query_file.empty() || args.gt_file.empty() || args.index_file.empty()) {
        throw std::runtime_error("Missing required arguments.");
    }
    if (args.doclens_file.empty()) {
        args.doclens_file = gpu_index_layout::doclens_path_for_index(args.index_file);
    }
    if (!std::filesystem::exists(args.doclens_file)) {
        throw std::runtime_error(
            "Missing doclens file. Provide --doclens or ensure " +
            std::string(gpu_index_layout::kDoclensFilename) + " exists next to --index.");
    }
    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    validate_gpu_search_runtime_options(args.runtime);

    return args;
}
