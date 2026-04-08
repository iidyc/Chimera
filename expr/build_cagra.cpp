// Build only the cuvs CAGRA index from centroids.bin (already rotated)
#include "ivf_pg.hpp"

int main() {
    int n, d;
    std::ifstream emb_file("centroids.bin", std::ios::binary);
    if (!emb_file.is_open()) {
        std::cerr << "Cannot open centroids.bin" << std::endl;
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

    pg_cagra.save("index.ivf");
    std::cout << "Saved to index.ivf.cagra" << std::endl;

    return 0;
}
