// Build only the cuvs CAGRA index from centroids.bin (already rotated)
#include "arg_utils.hpp"
#include "ivf_pg.hpp"

namespace {

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program << "\n"
        << "  --centroids <centroids.bin>\n"
        << "  --output <graph_prefix>\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string centroids_file;
    std::string output_prefix;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--centroids") {
                centroids_file = require_value(argc, argv, i, arg);
            } else if (arg == "--output") {
                output_prefix = require_value(argc, argv, i, arg);
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

    if (centroids_file.empty() || output_prefix.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    int n, d;
    std::ifstream emb_file(centroids_file, std::ios::binary);
    if (!emb_file.is_open()) {
        std::cerr << "Cannot open " << centroids_file << std::endl;
        return 1;
    }
    emb_file.read(reinterpret_cast<char*>(&n), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    size_t num_centroids = static_cast<size_t>(n);
    std::vector<float> centroids(num_centroids * d);
    emb_file.read(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
    emb_file.close();
    std::cout << "Loaded " << num_centroids << " centroids, d=" << d << std::endl;

    PG_CAGRA pg_cagra(num_centroids, d);
    pg_cagra.build_index(centroids.data());
    std::cout << "CAGRA index built." << std::endl;

    pg_cagra.save(output_prefix);
    std::cout << "Saved to " << output_prefix << ".cagra" << std::endl;

    return 0;
}
