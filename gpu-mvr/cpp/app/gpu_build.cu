#include "arg_utils.hpp"
#include "build_gpu_index.hpp"
#include "io.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " --index_dir <index_dir> --doclens <doclens> [--data <data> --n_clusters <n_clusters> | --source_index <index_dir> --clustered_only]\n\n"
        << "Arguments:\n"
        << "  --index_dir   Output index directory.\n"
        << "                The builder writes three files into this directory:\n"
        << "                ivf.bin, cpu_index.bin, centroids.carga, and gpu_index.bin.\n"
        << "  --doclens     Input doclens file.\n"
        << "                Each value is the length of one document, i.e. the number of\n"
        << "                token embeddings belonging to that document.\n"
        << "                This is required to recover document embeddings from token embeddings.\n"
        << "  --data        Input token embedding file.\n"
        << "                This contains the token embeddings used to build the index.\n"
        << "  --source_index Existing split index directory to clone before generating\n"
        << "                gpu_index.bin only.\n"
        << "  --clustered_only Generate only gpu_index.bin from an existing index.\n"
        << "  --n_clusters  Number of randomly sampled centroids used to build the CAGRA graph.\n\n"
        << "Summary:\n"
        << "  Full build mode needs doclens, data, and n_clusters.\n"
        << "  Fast sidecar mode needs doclens, source_index, and clustered_only.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string index_dir;
    std::string data_filename;
    std::string doclens_filename;
    std::string source_index_dir;
    size_t n_clusters = 0;
    bool clustered_only = false;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_input_help(argv[0]);
                return 0;
            }
            if (arg == "--index_dir") {
                index_dir = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--doclens") {
                doclens_filename = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--data") {
                data_filename = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--source_index") {
                source_index_dir = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--clustered_only") {
                clustered_only = true;
                continue;
            }
            if (arg == "--n_clusters") {
                n_clusters = std::stoull(require_value(argc, argv, i, arg));
                continue;
            }

            throw std::runtime_error("Unknown argument: " + arg);
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    if (index_dir.empty() || doclens_filename.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    auto doc_lens = load_doclens(doclens_filename);

    if (clustered_only) {
        if (source_index_dir.empty()) {
            std::cerr << "Missing required arguments for clustered-only mode.\n\n";
            print_input_help(argv[0]);
            return 1;
        }
        build_clustered_stage1_sidecar(doc_lens, source_index_dir, index_dir);
        return 0;
    }

    if (data_filename.empty() || n_clusters == 0) {
        std::cerr << "Missing required arguments for full build mode.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t ex_bits = 3;
    auto data = load_data_mmap(data_filename);
    build_index(data, n_clusters, ex_bits, doc_lens, index_dir);
    return 0;
}
