#include "search_cli.hpp"
#include "cpu_kernel_v2.hpp"
#include "index.hpp"
#include "io.hpp"
#include "utils.hpp"

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
    const bool count_checkpoint = (completed == 1) || (completed % 100 == 0);
    const bool time_checkpoint =
        std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5;

    if (!phase_done && !count_checkpoint && !time_checkpoint) {
        return;
    }

    const double elapsed_s = std::chrono::duration<double>(now - phase_start).count();
    const double qps = (elapsed_s > 0.0) ? static_cast<double>(completed) / elapsed_s : 0.0;
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
    args.runtime.k_rank_cluster = 1800;
    args.runtime.k_rank_all_tokens = 300;

    try {
        args = parse_gpu_search_args(argc, argv, args);
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "help_requested") {
            std::cerr << "Argument error: " << e.what() << "\n\n";
        }
        print_gpu_search_help(argv[0], args);
        return std::string(e.what()) == "help_requested" ? 0 : 1;
    }

    size_t num_q = 0;
    size_t d = 0;
    size_t q_doclen_file = 0;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, args.query_file);
    std::vector<int> doclens = load_doclens(args.doclens_file);
    auto ground_truth = read_gt_tsv(static_cast<int>(num_q), 1000, args.gt_file);

    cpu_mvr_index index(args.index_file);
    index.set_doc_mapping(doclens);
    cpu_kernel_v2::clustered_stage1_cache stage1_cache(args.index_file, index);
    cpu_kernel_v2::packed_token_stream_cache stage2_cache(index);

    const auto runtime_configs = load_gpu_search_runtime_configs(args);
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
        << "[RUN] Constructed CPU index from " << args.index_file
        << " with " << runtime_configs.size()
        << " configuration(s), warmup_queries=" << warmup_queries
        << " eval_queries=" << run_queries
        << " k=" << args.k
        << std::endl;
    std::cout
        << "[RUN] cpu_search_v2 uses clustered fastscan Stage 1 from cluster_1bit.bin; "
        << "Stage 2 and Stage 3 reuse cpu_search_v1 kernels."
        << std::endl;
    std::cout
        << "[RUN] stage1_prepack="
        << (cpu_kernel_v2::kStage1PrepackEnabled ? "enabled" : "disabled")
        << " packed_stage1_cache original_code_mib="
        << (static_cast<double>(stage1_cache.original_code_bytes()) / (1024.0 * 1024.0))
        << " packed_code_mib="
        << (static_cast<double>(stage1_cache.packed_code_bytes()) / (1024.0 * 1024.0))
        << " packed_factor_mib="
        << (static_cast<double>(stage1_cache.packed_factor_bytes()) / (1024.0 * 1024.0))
        << " packed_doc_id_mib="
        << (static_cast<double>(stage1_cache.packed_doc_id_bytes()) / (1024.0 * 1024.0))
        << " packed_total_mib="
        << (static_cast<double>(stage1_cache.packed_total_bytes()) / (1024.0 * 1024.0))
        << std::endl;
    std::cout
        << "[RUN] packed_stage2_cache original_code_mib="
        << (static_cast<double>(stage2_cache.original_code_bytes()) / (1024.0 * 1024.0))
        << " packed_code_mib="
        << (static_cast<double>(stage2_cache.packed_code_bytes()) / (1024.0 * 1024.0))
        << " packed_factor_mib="
        << (static_cast<double>(stage2_cache.packed_factor_bytes()) / (1024.0 * 1024.0))
        << " packed_total_mib="
        << (static_cast<double>(stage2_cache.packed_total_bytes()) / (1024.0 * 1024.0))
        << std::endl;
    std::cout
        << "[RUN] cpu_search_v2 uses nprobe, k_rank_cluster, and k_rank_all_tokens; "
        << "itopk_size and overlap_chunks are accepted for config compatibility and ignored."
        << std::endl;

    for (const auto& runtime_config : runtime_configs) {
        std::cout
            << "[RUN] Starting config label=" << runtime_config.label
            << " warmup_queries=" << warmup_queries
            << " eval_queries=" << run_queries
            << " k=" << args.k
            << std::endl;
        print_gpu_search_runtime_config_banner(runtime_config);

        const auto warmup_start = std::chrono::steady_clock::now();
        auto warmup_last_log = warmup_start;
        for (int i = 0; i < warmup_queries; ++i) {
            const float* query_ptr = &Q[static_cast<size_t>(i) * q_doclen_file * d];
            if (i + 1 == warmup_queries) {
                cpu_kernel_v2::search_profiled(
                    index,
                    stage1_cache,
                    stage2_cache,
                    query_ptr,
                    q_doclen_file,
                    static_cast<size_t>(args.k),
                    runtime_config.runtime);
            } else {
                cpu_kernel_v2::search(
                    index,
                    stage1_cache,
                    stage2_cache,
                    query_ptr,
                    q_doclen_file,
                    static_cast<size_t>(args.k),
                    runtime_config.runtime);
            }
            maybe_print_phase_progress(
                runtime_config.label,
                "warmup",
                i + 1,
                warmup_queries,
                warmup_start,
                warmup_last_log);
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
            results[i] = cpu_kernel_v2::search(
                index,
                stage1_cache,
                stage2_cache,
                &Q[static_cast<size_t>(i) * q_doclen_file * d],
                q_doclen_file,
                static_cast<size_t>(args.k),
                runtime_config.runtime);
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
        }

        const double total_seconds = timer.tuck(
            "label=" + runtime_config.label + " CPU search time for " +
            std::to_string(run_queries) + " queries.");
        print_query_latency_summary(query_latencies_ms, total_seconds);
        compute_recall(eval_ground_truth, results, args.k);
    }

    return 0;
}
