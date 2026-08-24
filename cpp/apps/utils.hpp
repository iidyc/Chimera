#pragma once

#include "chimera/chimera_index.cuh"

#include <cstddef>
#include <exception>
#include <string>
#include <vector>

namespace Chimera {

class HelpRequested final : public std::exception {};

struct BuildCliArgs {
    std::string index_dir;
    std::string data_file;
    std::string doclens_file;
    size_t n_clusters = 0;
    size_t ex_bits = 0;
};

struct SearchCliArgs {
    int k = 100;
    int nq = -1;
    int warmup = 5;
    std::string query_file;
    std::string gt_file;
    std::string index_dir;
    SearchOptions options;
};

struct QueryLatencySummary {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

BuildCliArgs parse_build_args(int argc, char** argv);
SearchCliArgs parse_search_args(int argc, char** argv);
void print_build_help(const char* program);
void print_search_help(const char* program);
double percentile_ms(const std::vector<double>& sorted_ms, double pct);
QueryLatencySummary summarize_query_latencies_ms(
    const std::vector<double>& latencies_ms);
std::vector<std::vector<size_t>> read_gt_tsv(
    int num_queries,
    int top_k,
    const std::string& gt_filename = "lotte-groundtruth-top1000--.tsv");
double compute_recall(
    const std::vector<std::vector<size_t>>& ground_truth,
    const std::vector<std::vector<size_t>>& retrieved,
    int top_k);

}  // namespace Chimera
