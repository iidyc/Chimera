#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace Chimera {

struct Timer {
    std::chrono::_V2::system_clock::time_point s;
    std::chrono::_V2::system_clock::time_point e;
    std::chrono::duration<double> diff;

    void tick();
    double tuck(std::string message, bool print = true);
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

double percentile_ms(const std::vector<double>& sorted_ms, double pct);
QueryLatencySummary summarize_query_latencies_ms(const std::vector<double>& latencies_ms);
void print_query_latency_summary(
    const std::vector<double>& latencies_ms,
    double end_to_end_seconds);
std::vector<std::vector<size_t>> read_gt_tsv(
    int num_queries,
    int top_k,
    const std::string& gt_filename = "lotte-groundtruth-top1000--.tsv");
double compute_recall(
    const std::vector<std::vector<size_t>>& ground_truth,
    const std::vector<std::vector<size_t>>& retrieved,
    int top_k);


}  // namespace Chimera
