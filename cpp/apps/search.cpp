#include "utils.hpp"
#include "chimera/config.cuh"
#include "chimera/io.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Chimera;

#ifndef CHIMERA_IMPL_NAME
#define CHIMERA_IMPL_NAME "v8"
#endif

namespace {

void report_recalls(
    const std::vector<std::vector<size_t>>& eval_ground_truth,
    const std::vector<std::vector<size_t>>& results,
    int result_k) {
    std::cout << "[SEARCH] recall@" << result_k << '='
              << compute_recall(eval_ground_truth, results, result_k)
              << '\n';
    if (result_k == 100) {
        std::cout << "[SEARCH] recall@10="
                  << compute_recall(eval_ground_truth, results, 10)
                  << '\n';
    }
}

void report_latency(
    const std::vector<double>& query_latencies_ms) {
    const auto latency = summarize_query_latencies_ms(query_latencies_ms);
    std::cout << std::fixed << std::setprecision(3)
              << "[SEARCH] latency_ms"
              << " mean=" << latency.mean_ms
              << " min=" << latency.min_ms
              << " p50=" << latency.p50_ms
              << " p90=" << latency.p90_ms
              << " p95=" << latency.p95_ms
              << " p99=" << latency.p99_ms
              << " max=" << latency.max_ms
              << " stddev=" << latency.stddev_ms
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    SearchCliArgs args;

    try {
        args = parse_search_args(argc, argv);
    } catch (const HelpRequested&) {
        print_search_help(argv[0]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_search_help(argv[0]);
        return 1;
    }

    std::cout << "[SEARCH] implementation=" << CHIMERA_IMPL_NAME
              << " concurrent_queries=1\n";

    size_t num_q = 0;
    size_t d = 0;
    size_t query_length = 0;
    std::vector<float> queries =
        load_query(query_length, num_q, d, args.query_file);
    auto ground_truth = read_gt_tsv(num_q, 1000, args.gt_file);

    if (query_length != Q_DOCLEN) {
        std::cerr << "Query length " << query_length
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << '\n';
        return 1;
    }

    const int warmup_queries =
        std::min<int>(args.warmup, static_cast<int>(num_q));
    const int run_queries =
        (args.nq < 0)
            ? static_cast<int>(num_q)
            : std::min<int>(args.nq, static_cast<int>(num_q));
    if (run_queries == 0) {
        std::cerr << "No evaluation queries selected.\n";
        return 1;
    }

    std::vector<std::vector<size_t>> eval_ground_truth(
        ground_truth.begin(), ground_truth.begin() + run_queries);

    chimera_index index;
    index.load(args.index_dir, args.options);

    for (int i = 0; i < warmup_queries; ++i) {
        index.search(&queries[i * Q_DOCLEN * d], Q_DOCLEN * d, args.k);
    }

    std::vector<std::vector<size_t>> results(run_queries);
    std::vector<double> query_latencies_ms(run_queries);
    for (int i = 0; i < run_queries; ++i) {
        const auto query_start = std::chrono::high_resolution_clock::now();
        results[i] =
            index.search(&queries[i * Q_DOCLEN * d], Q_DOCLEN * d, args.k);
        const auto query_end = std::chrono::high_resolution_clock::now();
        query_latencies_ms[i] =
            std::chrono::duration<double, std::milli>(
                query_end - query_start).count();
    }

    report_latency(query_latencies_ms);
    report_recalls(eval_ground_truth, results, args.k);

    return 0;
}
