#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace Chimera {

void Timer::tick() {
    s = std::chrono::high_resolution_clock::now();
}

double Timer::tuck(std::string message, bool print) {
    e = std::chrono::high_resolution_clock::now();
    diff = e - s;
    if (print) {
        std::cout << "[" << diff.count() << " s] " << message << std::endl;
    }
    return diff.count();
}

double percentile_ms(const std::vector<double>& sorted_ms, double pct) {
    if (sorted_ms.empty()) {
        return 0.0;
    }
    const double clamped = std::clamp(pct, 0.0, 100.0);
    const double pos = (clamped / 100.0) * static_cast<double>(sorted_ms.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double weight = pos - static_cast<double>(lo);
    return sorted_ms[lo] * (1.0 - weight) + sorted_ms[hi] * weight;
}

QueryLatencySummary summarize_query_latencies_ms(const std::vector<double>& latencies_ms) {
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

void print_query_latency_summary(
    const std::vector<double>& latencies_ms,
    double end_to_end_seconds)
{
    const auto summary = summarize_query_latencies_ms(latencies_ms);
    const double avg_qps =
        (end_to_end_seconds > 0.0)
            ? static_cast<double>(latencies_ms.size()) / end_to_end_seconds
            : 0.0;

    auto old_flags = std::cout.flags();
    auto old_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[SEARCH] End-to-end measured time: "
              << end_to_end_seconds * 1000.0 << " ms\n";
    std::cout << "[SEARCH] Average latency per query: "
              << summary.mean_ms << " ms\n";
    std::cout << "[SEARCH] Throughput: " << avg_qps << " qps\n";
    std::cout << "[SEARCH] Query latency distribution (ms): "
              << "min=" << summary.min_ms
              << ", p50=" << summary.p50_ms
              << ", p90=" << summary.p90_ms
              << ", p95=" << summary.p95_ms
              << ", p99=" << summary.p99_ms
              << ", max=" << summary.max_ms
              << ", stddev=" << summary.stddev_ms
              << "\n";
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

std::vector<std::vector<size_t>> read_gt_tsv(
    int num_queries,
    int top_k,
    const std::string& gt_filename)
{
    std::vector<std::vector<size_t>> ground_truth(num_queries, std::vector<size_t>(top_k, -1));
    std::ifstream file(gt_filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << gt_filename << std::endl;
        return ground_truth;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> arr;
        while (std::getline(ss, segment, '\t')) {
            arr.push_back(segment);
        }
        if (arr.size() < 3) {
            std::cerr << "Warning: Skipping malformed line." << std::endl;
            continue;
        }
        int qID = std::stoi(arr[0]);
        size_t itemID = std::stoull(arr[1]);
        int rank = std::stoi(arr[2]);
        if (rank - 1 >= top_k || rank - 1 < 0) {
            std::cerr << "Warning: Rank " << rank << " exceeds top_k " << top_k << ". Skipping." << std::endl;
            std::exit(0);
        }
        ground_truth[qID][rank - 1] = itemID;
    }
    return ground_truth;
}

double compute_recall(
    const std::vector<std::vector<size_t>>& ground_truth,
    const std::vector<std::vector<size_t>>& retrieved,
    int top_k)
{
    if (top_k <= 0) {
        std::cout << "Recall@" + std::to_string(top_k) + ": 0" << std::endl;
        return 0.0;
    }

    int num_queries = retrieved.size();
    int total_recall = 0;
    for (int i = 0; i < num_queries; ++i) {
        const auto& gt = ground_truth[i];
        const auto& ret = retrieved[i];
        const size_t retrieved_cutoff = std::min(ret.size(), static_cast<size_t>(top_k));
        std::unordered_set<size_t> ret_set(ret.begin(), ret.begin() + retrieved_cutoff);
        int correct = 0;
        const int gt_cutoff = std::min<int>(top_k, static_cast<int>(gt.size()));
        for (int j = 0; j < gt_cutoff; ++j) {
            if (ret_set.find(gt[j]) != ret_set.end()) {
                correct++;
            }
        }
        total_recall += correct;
    }
    double recall = static_cast<double>(total_recall) / (num_queries * top_k);
    std::cout << "Recall@" + std::to_string(top_k) + ": " << recall << std::endl;
    return recall;
}


}  // namespace Chimera
