#include "search_cli.hpp"
#include "gpu_memory_tracker.hpp"
#include "gpu_index.cuh"
#include "io.hpp"
#include "startup_profile.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

using namespace Chimera;

namespace {

void apply_runtime_config(chimera_index& index, const gpu_search_runtime_options& runtime) {
    index.nprobe = runtime.nprobe;
    index.k_rank_cluster = runtime.k_rank_cluster;
    index.k_rank_all_tokens = runtime.k_rank_all_tokens;
    index.itopk_size = runtime.itopk_size;
    index.overlap_chunks = runtime.overlap_chunks;
}

void report_recalls(
    const std::vector<std::vector<size_t>>& eval_ground_truth,
    const std::vector<std::vector<size_t>>& results,
    int result_k) {
    compute_recall(eval_ground_truth, results, result_k);
    if (result_k == 100) {
        compute_recall(eval_ground_truth, results, 10);
    }
}

#ifdef CHIMERA_PROFILE_APP
struct QueryLatencyRecord {
    std::string runtime_label;
    int query_id = -1;
    int slot = -1;
    int assignment_order = -1;
    int completion_order = -1;
    double start_ms = 0.0;
    double end_ms = 0.0;
    double latency_ms = 0.0;
};

struct ProfileArgs {
    std::string latency_csv = "gpu_search_profile_latency.csv";
};

double elapsed_ms_since(
    const std::chrono::steady_clock::time_point& base,
    const std::chrono::steady_clock::time_point& now) {
    return std::chrono::duration<double, std::milli>(now - base).count();
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

void write_latency_csv(
    const std::string& path,
    const std::vector<QueryLatencyRecord>& records) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open latency CSV for writing: " + path);
    }
    out << "runtime_label,query_id,slot,assignment_order,completion_order,"
        << "start_ms,end_ms,latency_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& record : records) {
        out << csv_escape(record.runtime_label) << ','
            << record.query_id << ','
            << record.slot << ','
            << record.assignment_order << ','
            << record.completion_order << ','
            << record.start_ms << ','
            << record.end_ms << ','
            << record.latency_ms << '\n';
    }
}

std::vector<std::string> strip_profile_args(
    int argc,
    char** argv,
    ProfileArgs& profile_args) {
    std::vector<std::string> forwarded;
    forwarded.reserve(static_cast<size_t>(argc));
    forwarded.emplace_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--latency-csv") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --latency-csv");
            }
            profile_args.latency_csv = argv[++i];
        } else {
            forwarded.push_back(arg);
        }
    }
    return forwarded;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    gpu_search_cli_args args;
    args.runtime.k_rank_cluster = 3000;
    args.concurrent_queries = 1;

#ifdef CHIMERA_PROFILE_APP
    ProfileArgs profile_args;
    std::vector<std::string> forwarded_args;
    std::vector<char*> forwarded_argv;
    try {
        forwarded_args = strip_profile_args(argc, argv, profile_args);
        forwarded_argv.reserve(forwarded_args.size());
        for (auto& arg : forwarded_args) {
            forwarded_argv.push_back(arg.data());
        }
        argc = static_cast<int>(forwarded_argv.size());
        argv = forwarded_argv.data();
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_gpu_search_help(argv[0], args);
        std::cout << "  [--latency-csv <path>] Write per-query latency records. Default: "
                  << profile_args.latency_csv << "\n";
        return 1;
    }
#endif

    try {
        args = parse_gpu_search_args(argc, argv, args);
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "help_requested") {
            std::cerr << "Argument error: " << e.what() << "\n\n";
        }
        print_gpu_search_help(argv[0], args);
#ifdef CHIMERA_PROFILE_APP
        std::cout << "  [--latency-csv <path>] Write per-query latency records. Default: "
                  << profile_args.latency_csv << "\n";
#endif
        return std::string(e.what()) == "help_requested" ? 0 : 1;
    }

    int cuda_device = 0;
    CUDA_CHECK(cudaGetDevice(&cuda_device));

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

    const int query_slots = std::max(1, std::min(args.concurrent_queries, std::max(1, run_queries)));
    const int effective_stage3_threads =
        (args.stage3_threads > 0)
            ? args.stage3_threads
            : std::max(1, omp_get_max_threads() / query_slots);
    std::cout
        << "[RUN] Constructing shared gpu_search index with " << query_slots
        << " query slot(s) from " << args.index_file
        << " using max runtime config: "
        << "nprobe=" << allocation_runtime.nprobe
        << " k_rank_cluster=" << allocation_runtime.k_rank_cluster
        << " k_rank_all_tokens=" << allocation_runtime.k_rank_all_tokens
        << " itopk_size=" << allocation_runtime.itopk_size
        << " overlap_chunks=" << allocation_runtime.overlap_chunks
        << std::endl;

    auto owner_index = std::make_unique<chimera_index>(
        args.index_file,
        doclens,
        allocation_runtime);
    std::vector<std::unique_ptr<chimera_index>> extra_slots;
    extra_slots.reserve(std::max(0, query_slots - 1));
    std::vector<chimera_index*> indices;
    indices.reserve(query_slots);
    indices.push_back(owner_index.get());
    for (int slot = 1; slot < query_slots; ++slot) {
        extra_slots.push_back(std::make_unique<chimera_index>(
            *owner_index,
            allocation_runtime));
        indices.push_back(extra_slots.back().get());
    }
    startup.mark("construct_indices");

    Chimera::GpuMemoryTracker gpu_memory;
    gpu_memory.sample("after_index_construct");

    if (args.profile_eval_all_queries && query_slots > 1) {
        std::cout
            << "[RUN] profile-eval-all-queries requested; running evaluation sequentially "
            << "because per-query profile output is not slot-aggregated.\n";
    }

#ifdef CHIMERA_PROFILE_APP
    std::vector<QueryLatencyRecord> all_latency_records;
    all_latency_records.reserve(static_cast<size_t>(run_queries) * runtime_configs.size());
#endif

    for (const auto& runtime_config : runtime_configs) {
        print_gpu_search_runtime_config_banner(runtime_config);
        std::cout
            << "[CONFIG] concurrent_queries=" << query_slots
            << " effective_concurrent_queries="
            << (args.profile_eval_all_queries ? 1 : query_slots)
            << " stage3_threads=" << effective_stage3_threads
            << std::endl;
        for (auto* index : indices) {
            apply_runtime_config(*index, runtime_config.runtime);
        }
        gpu_memory.sample("config_begin:" + runtime_config.label);

        for (int i = 0; i < warmup_queries; ++i) {
            omp_set_num_threads(effective_stage3_threads);
            const int slot = i % query_slots;
            const float* query_ptr = &Q[i * Q_DOCLEN * d];
            if (!args.profile_eval_all_queries && i + 1 == warmup_queries) {
                indices[slot]->search_profiled(query_ptr, args.k);
            } else {
                indices[slot]->search(query_ptr, args.k);
            }
            gpu_memory.sample_query_if_needed(
                "warmup:" + runtime_config.label,
                static_cast<size_t>(i),
                static_cast<size_t>(warmup_queries));
        }

        Timer timer;
        timer.tick();
        std::vector<std::vector<size_t>> results(run_queries);
        std::vector<double> query_latencies_ms(run_queries, 0.0);
#ifdef CHIMERA_PROFILE_APP
        std::vector<QueryLatencyRecord> latency_records(run_queries);
        std::atomic<int> assignment_order{0};
        std::atomic<int> completion_order{0};
        const auto eval_start = std::chrono::steady_clock::now();
#endif

        if (args.profile_eval_all_queries) {
            for (int i = 0; i < run_queries; ++i) {
                omp_set_num_threads(effective_stage3_threads);
#ifdef CHIMERA_PROFILE_APP
                const int assigned = assignment_order.fetch_add(1, std::memory_order_relaxed);
                const auto query_start = std::chrono::steady_clock::now();
#else
                const auto query_start = std::chrono::high_resolution_clock::now();
#endif
                results[i] = indices[0]->search_profiled(&Q[i * Q_DOCLEN * d], args.k);
                const auto query_end =
#ifdef CHIMERA_PROFILE_APP
                    std::chrono::steady_clock::now();
#else
                    std::chrono::high_resolution_clock::now();
#endif
                query_latencies_ms[i] =
                    std::chrono::duration<double, std::milli>(query_end - query_start).count();
#ifdef CHIMERA_PROFILE_APP
                latency_records[i] = QueryLatencyRecord{
                    runtime_config.label,
                    i,
                    0,
                    assigned,
                    completion_order.fetch_add(1, std::memory_order_relaxed),
                    elapsed_ms_since(eval_start, query_start),
                    elapsed_ms_since(eval_start, query_end),
                    query_latencies_ms[i]};
#endif
                gpu_memory.sample_query_if_needed(
                    "eval:" + runtime_config.label,
                    static_cast<size_t>(i),
                    static_cast<size_t>(run_queries));
            }
        } else {
            std::atomic<int> next_query{0};
            std::vector<std::thread> workers;
            workers.reserve(query_slots);
            for (int slot = 0; slot < query_slots; ++slot) {
                workers.emplace_back([&, slot]() {
                    CUDA_CHECK(cudaSetDevice(cuda_device));
                    omp_set_num_threads(effective_stage3_threads);
                    chimera_index* index = indices[slot];
                    while (true) {
                        const int query_idx = next_query.fetch_add(1, std::memory_order_relaxed);
                        if (query_idx >= run_queries) {
                            break;
                        }

                        const float* query_ptr = &Q[query_idx * Q_DOCLEN * d];
#ifdef CHIMERA_PROFILE_APP
                        const int assigned =
                            assignment_order.fetch_add(1, std::memory_order_relaxed);
                        const auto query_start = std::chrono::steady_clock::now();
#else
                        const auto query_start = std::chrono::high_resolution_clock::now();
#endif
                        results[query_idx] = index->search(query_ptr, args.k);
                        const auto query_end =
#ifdef CHIMERA_PROFILE_APP
                            std::chrono::steady_clock::now();
#else
                            std::chrono::high_resolution_clock::now();
#endif
                        query_latencies_ms[query_idx] =
                            std::chrono::duration<double, std::milli>(
                                query_end - query_start).count();
#ifdef CHIMERA_PROFILE_APP
                        latency_records[query_idx] = QueryLatencyRecord{
                            runtime_config.label,
                            query_idx,
                            slot,
                            assigned,
                            completion_order.fetch_add(1, std::memory_order_relaxed),
                            elapsed_ms_since(eval_start, query_start),
                            elapsed_ms_since(eval_start, query_end),
                            query_latencies_ms[query_idx]};
#endif
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            gpu_memory.sample("eval_end:" + runtime_config.label);
        }

        gpu_memory.sample("config_end:" + runtime_config.label);
        const double total_seconds = timer.tuck(
            "label=" + runtime_config.label + " GPU search time for " +
            std::to_string(run_queries) + " queries.");
        print_query_latency_summary(query_latencies_ms, total_seconds);
        report_recalls(eval_ground_truth, results, args.k);
#ifdef CHIMERA_PROFILE_APP
        all_latency_records.insert(
            all_latency_records.end(),
            latency_records.begin(),
            latency_records.end());
#endif
    }

#ifdef CHIMERA_PROFILE_APP
    write_latency_csv(profile_args.latency_csv, all_latency_records);
    std::cout << "[PROFILE] Wrote per-query latency CSV: "
              << profile_args.latency_csv << " (rows="
              << all_latency_records.size() << ")\n";
#endif

    gpu_memory.print_summary();

    return 0;
}
