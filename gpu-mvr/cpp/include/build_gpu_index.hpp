#pragma once

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GPU-accelerated index build pipeline
// ---------------------------------------------------------------------------
// 1. Randomly sample centroids
// 2. Build a temporary non-rotated CAGRA graph for fast centroid assignment
// 3. Build the persisted CAGRA graph on rotated centroids
// 4. Assemble IVF_PG, quantize data, and serialize everything to disk
//
// Steps 1 and 2 can optionally be bootstrapped from precomputed files
// produced by expr/build_index_step12.py by passing non-empty
// `bootstrap_centroids` and `bootstrap_list_nos` paths.
// ---------------------------------------------------------------------------
void build_index(
    const float* data,
    size_t n,
    size_t d,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& filename,
    const std::string& bootstrap_centroids = "",
    const std::string& bootstrap_list_nos = "");
