#include "arg_utils.hpp"
#include "index.hpp"
#include "io.hpp"

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --data <embeddings.bin>\n"
        << "  --output <index_file>\n"
        << "  --n_clusters <num_centroids>\n"
        << "  [--ex_bits <ex_bits>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_file;
    std::string output_file;
    size_t n_clusters = 0;
    size_t ex_bits = 4;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--data") {
                data_file = require_value(argc, argv, i, arg);
            } else if (arg == "--output") {
                output_file = require_value(argc, argv, i, arg);
            } else if (arg == "--n_clusters") {
                n_clusters = std::stoull(require_value(argc, argv, i, arg));
            } else if (arg == "--ex_bits") {
                ex_bits = std::stoull(require_value(argc, argv, i, arg));
            } else if (arg == "--help" || arg == "-h") {
                print_input_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    if (data_file.empty() || output_file.empty() || n_clusters == 0) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    size_t n, d;
    std::vector<float> data = load_data(n, d, data_file);
    cpu_mvr_index idx(n, d, n_clusters, ex_bits);
    idx.build_index(data.data());
    idx.save(output_file);
    return 0;
}
