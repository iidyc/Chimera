#pragma once

#include <cstddef>
#include <vector>

#include "gpu_search_options.hpp"
#include "index.hpp"

namespace cpu_kernel_v1 {

struct search_profile {
    double stage1_cluster_ms = 0.0;
    double stage1_probe_ms = 0.0;
    double stage1_prepare_ms = 0.0;
    double stage1_scan_ms = 0.0;
    double stage1_reduce_ms = 0.0;
    double stage2_1bit_ms = 0.0;
    double stage3_exbits_ms = 0.0;
    double total_ms = 0.0;
};

void rank_cluster_dists(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    size_t nprobe,
    size_t k,
    std::vector<size_t>& output_ids);

void rank_all_tokens_1bit(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    const std::vector<size_t>& input_ids,
    size_t k,
    std::vector<size_t>& output_ids,
    std::vector<float>& one_bit_dists);

void rank_all_tokens_exbits(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    const std::vector<size_t>& input_ids,
    const std::vector<float>& one_bit_dists,
    size_t k,
    std::vector<size_t>& output_ids);

std::vector<size_t> search(
    const cpu_mvr_index& index,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime);

std::vector<size_t> search_profiled(
    const cpu_mvr_index& index,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime,
    search_profile* profile = nullptr);

void print_search_profile(const search_profile& profile);

}  // namespace cpu_kernel_v1
