#include "arg_utils.hpp"
#include "build_index.hpp"
#include "gpu_index_layout.hpp"
#include "io.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace Chimera;

namespace {

void copy_doclens_into_index_dir(
    const std::string& index_dir,
    const std::string& doclens_filename)
{
    std::filesystem::create_directories(index_dir);
    const auto dest_doclens_path = gpu_index_layout::doclens_path(index_dir);
    const auto source_path = std::filesystem::path(doclens_filename);
    const auto dest_path = std::filesystem::path(dest_doclens_path);
    if (std::filesystem::exists(source_path) &&
        std::filesystem::exists(dest_path) &&
        std::filesystem::equivalent(source_path, dest_path)) {
        return;
    }
    std::filesystem::copy_file(
        source_path,
        dest_path,
        std::filesystem::copy_options::overwrite_existing);
}

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " --index_dir <index_dir> --doclens <doclens> --data <data> --n_clusters <n_clusters>\n\n"
        << "Arguments:\n"
        << "  --index_dir   Output index directory.\n"
        << "                The builder writes the split index artifacts into this directory:\n"
        << "                ivf.bin, doc_1bit.bin, doc_4bit.bin, doc_4bit_ex.bin,\n"
        << "                cluster_1bit.bin, index_metadata.json, centroids.carga,\n"
        << "                and centroids.hnsw.\n"
        << "  --doclens     Input doclens file.\n"
        << "                Each value is the length of one document, i.e. the number of\n"
        << "                token embeddings belonging to that document.\n"
        << "                This is required to recover document embeddings from token embeddings.\n"
        << "                The file is copied into <index_dir>/doclens.bin.\n"
        << "  --data        Input token embedding file.\n"
        << "                This contains the token embeddings used to build the index.\n"
        << "  --n_clusters  Number of randomly sampled centroids used to build the CAGRA graph.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string index_dir;
    std::string data_filename;
    std::string doclens_filename;
    size_t n_clusters = 0;

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
    copy_doclens_into_index_dir(index_dir, doclens_filename);

    if (data_filename.empty() || n_clusters == 0) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t ex_bits = 3;
    auto data = load_data_mmap(data_filename);
    build_index(data, n_clusters, ex_bits, doc_lens, index_dir);
    return 0;
}
