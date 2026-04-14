#include "gpu_search_cli.hpp"
#include "gpu_index_baseline.cuh"
#include "io.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    gpu_search_cli_args args;
    args.runtime.k_rank_cluster = 5000;

    try {
        args = parse_gpu_search_args(argc, argv, args);
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "help_requested") {
            std::cerr << "Argument error: " << e.what() << "\n\n";
        }
        print_gpu_search_help(argv[0], args);
        return std::string(e.what()) == "help_requested" ? 0 : 1;
    }

    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d, args.query_file);
    std::vector<int> doclens = load_doclens(args.doclens_file);
    auto ground_truth = read_gt_tsv(static_cast<int>(num_q), 1000, args.gt_file);

    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        return 1;
    }

    gpu_mvr_index_baseline index(args.index_file, doclens, args.runtime);

    const int warmup_queries = std::min<int>(args.warmup, static_cast<int>(num_q));
    for (int i = 0; i < warmup_queries; ++i) {
        index.search(&Q[i * Q_DOCLEN * d], args.k);
    }

    const int remaining_queries = std::max<int>(0, static_cast<int>(num_q) - warmup_queries);
    const int run_queries =
        (args.nq < 0) ? remaining_queries : std::min<int>(args.nq, remaining_queries);
    if (run_queries == 0) {
        std::cerr << "No evaluation queries remain after warmup." << std::endl;
        return 1;
    }

    Timer timer;
    timer.tick();
    std::vector<std::vector<size_t>> results(run_queries);
    for (int i = 0; i < run_queries; ++i) {
        const int query_idx = warmup_queries + i;
        results[i] = index.search(&Q[query_idx * Q_DOCLEN * d], args.k);
    }
    timer.tuck("v0 GPU search time for " + std::to_string(run_queries) + " queries.");

    std::vector<std::vector<size_t>> eval_ground_truth(
        ground_truth.begin() + warmup_queries,
        ground_truth.begin() + warmup_queries + run_queries);
    compute_recall(eval_ground_truth, results, args.k);

    return 0;
}
