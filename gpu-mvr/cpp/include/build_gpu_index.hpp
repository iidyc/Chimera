#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "io.hpp"

// ---------------------------------------------------------------------------
// GPU-accelerated index build pipeline
// ---------------------------------------------------------------------------
// 1. Randomly sample centroids
// 2. Build a temporary non-rotated CAGRA graph for fast centroid assignment
// 3. Build the persisted CAGRA graph on rotated centroids
// 4. Assemble IVF_PG, quantize data, and serialize everything into `index_dir`
//    as `ivf.bin`, `quantized_data.bin`, and `centroids.carga`
//
// ---------------------------------------------------------------------------
void build_index(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& index_dir);
