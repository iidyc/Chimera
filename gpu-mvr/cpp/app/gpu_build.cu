#include "arg_utils.hpp"
#include "build_gpu_index.hpp"
#include "io.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " --index_dir <index_dir> --doclens <doclens> --data <data> --n_clusters <n_clusters>\n\n"
        << "Arguments:\n"
        << "  --index_dir   Output index directory.\n"
        << "                The builder writes three files into this directory:\n"
        << "                ivf.bin, quantized_data.bin, and centroids.carga.\n"
        << "  --doclens     Input doclens file.\n"
        << "                Each value is the length of one document, i.e. the number of\n"
        << "                token embeddings belonging to that document.\n"
        << "                This is required to recover document embeddings from token embeddings.\n"
        << "  --data        Input token embedding file.\n"
        << "                This contains the token embeddings used to build the index.\n"
        << "  --n_clusters  Number of randomly sampled centroids used to build the CAGRA graph.\n\n"
        << "Summary:\n"
        << "  index_dir specifies the output directory.\n"
        << "  doclens, data, and n_clusters are inputs to the build.\n";
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

    if (index_dir.empty() || doclens_filename.empty() || data_filename.empty() || n_clusters == 0) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t ex_bits = 3;
    auto data = load_data_mmap(data_filename);
    auto doc_lens = load_doclens(doclens_filename);
    build_index(data, n_clusters, ex_bits, doc_lens, index_dir);
    return 0;
}
