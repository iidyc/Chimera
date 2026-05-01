#include "search_cli.hpp"
#include "gpu_memory_tracker.hpp"
#include "gpu_index_v4.cuh"
#include "io.hpp"
#include "startup_profile.hpp"
#include "utils.hpp"

using namespace Chimera;

namespace {

void maybe_print_phase_progress(
    const std::string& label,
    const char* phase,
    int completed,
    int total,
    std::chrono::steady_clock::time_point phase_start,
    std::chrono::steady_clock::time_point& last_log_time)
{
    if (total <= 0 || completed <= 0) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const bool phase_done = completed >= total;
    const bool count_checkpoint = (completed == 1) || (completed % 250 == 0);
    const bool time_checkpoint =
        std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5;

    if (!phase_done && !count_checkpoint && !time_checkpoint) {
        return;
    }

    const double elapsed_s =
        std::chrono::duration<double>(now - phase_start).count();
    const double qps =
        (elapsed_s > 0.0) ? static_cast<double>(completed) / elapsed_s : 0.0;
    const double eta_s =
        (!phase_done && qps > 0.0) ? static_cast<double>(total - completed) / qps : 0.0;

    std::cout
        << "[RUN] Progress label=" << label
        << " phase=" << phase
        << " completed=" << completed << "/" << total
        << " elapsed_s=" << elapsed_s
        << " qps=" << qps;
    if (!phase_done && qps > 0.0) {
        std::cout << " eta_s=" << eta_s;
    }
    std::cout << std::endl;

    last_log_time = now;
}

}  // namespace

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

    Chimera::StartupProfile startup("app");

    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, args.query_file);
    startup.mark("load_query");
    std::vector<int> doclens = load_doclens(args.doclens_file);
    startup.mark("load_doclens");
    auto ground_truth = read_gt_tsv(num_q, 1000, args.gt_file);
    startup.mark("read_gt_tsv");

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

    std::cout
        << "[RUN] Constructing GPU index from " << args.index_file
        << " using max runtime config: "
        << "nprobe=" << allocation_runtime.nprobe
        << " k_rank_cluster=" << allocation_runtime.k_rank_cluster
        << " k_rank_all_tokens=" << allocation_runtime.k_rank_all_tokens
        << " itopk_size=" << allocation_runtime.itopk_size
        << " overlap_chunks=" << allocation_runtime.overlap_chunks
        << std::endl;
    chimera_index index(args.index_file, doclens, allocation_runtime);
    startup.mark("construct_index");
    std::cout
        << "[RUN] Index ready. Benchmarking " << runtime_configs.size()
        << " configuration(s) with warmup_queries=" << warmup_queries
        << " eval_queries=" << run_queries
        << " k=" << args.k
        << std::endl;
    Chimera::GpuMemoryTracker gpu_memory;
    gpu_memory.sample("after_index_construct");

    for (const auto& runtime_config : runtime_configs) {
        std::cout
            << "[RUN] Starting config label=" << runtime_config.label
            << " warmup_queries=" << warmup_queries
            << " eval_queries=" << run_queries
            << " k=" << args.k
            << std::endl;
        print_gpu_search_runtime_config_banner(runtime_config);
        index.nprobe = runtime_config.runtime.nprobe;
        index.k_rank_cluster = runtime_config.runtime.k_rank_cluster;
        index.k_rank_all_tokens = runtime_config.runtime.k_rank_all_tokens;
        index.itopk_size = runtime_config.runtime.itopk_size;
        index.overlap_chunks = runtime_config.runtime.overlap_chunks;
        gpu_memory.sample("config_begin:" + runtime_config.label);

        const auto warmup_start = std::chrono::steady_clock::now();
        auto warmup_last_log = warmup_start;
        for (int i = 0; i < warmup_queries; ++i) {
            const float* query_ptr = &Q[i * Q_DOCLEN * d];
            if (i + 1 == warmup_queries) {
                index.search_profiled(query_ptr, args.k);
            } else {
                index.search(query_ptr, args.k);
            }
            maybe_print_phase_progress(
                runtime_config.label,
                "warmup",
                i + 1,
                warmup_queries,
                warmup_start,
                warmup_last_log);
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
        const auto eval_start = std::chrono::steady_clock::now();
        auto eval_last_log = eval_start;
        for (int i = 0; i < run_queries; ++i) {
            const auto query_start = std::chrono::high_resolution_clock::now();
            results[i] = index.search(&Q[i * Q_DOCLEN * d], args.k);
            const auto query_end = std::chrono::high_resolution_clock::now();
            query_latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(query_end - query_start).count());
            maybe_print_phase_progress(
                runtime_config.label,
                "eval",
                i + 1,
                run_queries,
                eval_start,
                eval_last_log);
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
