#include "search_cli.hpp"
#include "gpu_memory_tracker.hpp"
#include "gpu_index_v3.cuh"
#include "io.hpp"
#include "startup_profile.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    gpu_search_cli_args args;
    args.runtime.k_rank_cluster = 3000;

    try {
        args = parse_gpu_search_args(argc, argv, args);
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "help_requested") {
            std::cerr << "Argument error: " << e.what() << "\n\n";
        }
        print_gpu_search_help(argv[0], args);
        return std::string(e.what()) == "help_requested" ? 0 : 1;
    }

    gpu_mvr::StartupProfile startup("app");

    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, args.query_file);
    startup.mark("load_query");
    std::vector<int> doclens = load_doclens(args.doclens_file);
    startup.mark("load_doclens");
    auto ground_truth = read_gt_tsv(num_q, 1000, args.gt_file);
    startup.mark("read_gt_tsv");

    // Validate that query file matches compiled Q_DOCLEN
    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        std::cerr << "Please recompile with matching Q_DOCLEN in gpu_config.cuh" << std::endl;
        return 1;
    }

    const auto runtime_configs = load_gpu_search_runtime_configs(args);
    const auto allocation_runtime = max_gpu_search_runtime_options(runtime_configs);
    const int warmup_queries = std::min<int>(args.warmup, static_cast<int>(num_q));
    const int run_queries =
        (args.nq < 0) ? static_cast<int>(num_q) : std::min<int>(args.nq, static_cast<int>(num_q));
    if (run_queries == 0) {
        std::cerr << "No evaluation queries selected." << std::endl;
        return 1;
    }

    std::vector<std::vector<size_t>> eval_ground_truth(
        ground_truth.begin(),
        ground_truth.begin() + run_queries);

    gpu_mvr_index index(args.index_file, doclens, allocation_runtime);
    startup.mark("construct_index");
    gpu_mvr::GpuMemoryTracker gpu_memory;
    gpu_memory.sample("after_index_construct");

    for (const auto& runtime_config : runtime_configs) {
        print_gpu_search_runtime_config_banner(runtime_config);
        index.nprobe = runtime_config.runtime.nprobe;
        index.k_rank_cluster = runtime_config.runtime.k_rank_cluster;
        index.k_rank_all_tokens = runtime_config.runtime.k_rank_all_tokens;
        index.itopk_size = runtime_config.runtime.itopk_size;
        index.overlap_chunks = runtime_config.runtime.overlap_chunks;
        gpu_memory.sample("config_begin:" + runtime_config.label);

        for (int i = 0; i < warmup_queries; ++i) {
            const float* query_ptr = &Q[i * Q_DOCLEN * d];
            if (i + 1 == warmup_queries) {
                index.search_profiled(query_ptr, args.k);
            } else {
                index.search(query_ptr, args.k);
            }
            gpu_memory.sample_query_if_needed(
                "warmup:" + runtime_config.label,
                static_cast<size_t>(i),
                static_cast<size_t>(warmup_queries));
        }

        Timer timer;
        timer.tick();
        std::vector<std::vector<size_t>> results(run_queries);
        std::vector<double> query_latencies_ms;
        query_latencies_ms.reserve(run_queries);
        for (int i = 0; i < run_queries; ++i) {
            const auto query_start = std::chrono::high_resolution_clock::now();
            results[i] = index.search(&Q[i * Q_DOCLEN * d], args.k);
            const auto query_end = std::chrono::high_resolution_clock::now();
            query_latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(query_end - query_start).count());
            gpu_memory.sample_query_if_needed(
                "eval:" + runtime_config.label,
                static_cast<size_t>(i),
                static_cast<size_t>(run_queries));
        }
        gpu_memory.sample("config_end:" + runtime_config.label);
        const double total_seconds = timer.tuck(
            "label=" + runtime_config.label + " GPU search time for " +
            std::to_string(run_queries) + " queries.");
        print_query_latency_summary(query_latencies_ms, total_seconds);
        compute_recall(eval_ground_truth, results, args.k);
    }

    gpu_memory.print_summary();

    return 0;
}
