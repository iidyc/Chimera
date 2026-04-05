#include "build_gpu_index.hpp"
#include "io.hpp"

int main(int argc, char* argv[]) {
    size_t n_clusters = 524288;
    size_t ex_bits = 4;
    size_t d = 128;  // embedding dimension
    std::string filename = "524288_4_new.index";

    if (argc > 1 && std::string(argv[1]) == "--patch-cagra") {
        // Fast path: only rebuild CAGRA with rotated centroids (no data load needed)
        rebuild_cagra_rotated(n_clusters, d, filename);
    } else {
        size_t n;
        std::vector<float> data = load_data(n, d);
        auto doc_lens = load_doclens();
        build_index(data.data(), n, d, n_clusters, ex_bits, doc_lens, filename);
    }
    return 0;
}
