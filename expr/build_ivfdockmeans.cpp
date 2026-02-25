#include "ivf_dockmeans.hpp"
#include "io.hpp"

int main() {
    size_t n, d;
    std::vector<float> data = load_data(n, d);
    auto doclens = load_doclens();
    std::vector<int> doc_ptrs(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        doc_ptrs[i + 1] = doc_ptrs[i] + doclens[i];
    }
    IVF_DocKMeans idx(20480, d);
    idx.build(data.data(), doc_ptrs.data(), doclens.size(), n, 10);
    idx.save("2097152_4_new.index");
    return 0;
}