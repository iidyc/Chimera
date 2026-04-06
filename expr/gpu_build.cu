#include "build_gpu_index.hpp"
#include "io.hpp"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <index_name> <data_filename> <doclens_filename> <n_clusters>" << std::endl;
        return 1;
    }
    std::string index_name = argv[1];
    std::string data_filename = argv[2];
    std::string doclens_filename = argv[3];
    int n_clusters = std::stoi(argv[4]);
    size_t ex_bits = 4;
    size_t d = 128;  // embedding dimension

    // if (argc > 1 && std::string(argv[1]) == "--patch-cagra") {
    //     // Fast path: only rebuild CAGRA with rotated centroids (no data load needed)
    //     rebuild_cagra_rotated(n_clusters, d, filename);
    // } else {
        size_t n;
        std::vector<float> data = load_data(n, d, data_filename);
        auto doc_lens = load_doclens(doclens_filename);
        build_index(data.data(), n, d, n_clusters, ex_bits, doc_lens, index_name);
    // }
    return 0;
}
