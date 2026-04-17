#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "io.hpp"

void build_index_fast(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& index_dir);

void build_clustered_stage1_sidecar_fast(
    const std::vector<int>& doc_lens,
    const std::string& source_index_dir,
    const std::string& index_dir);
