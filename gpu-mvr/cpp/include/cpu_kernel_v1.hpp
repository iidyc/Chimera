#pragma once

#include <cstddef>
#include <vector>

#include "gpu_search_options.hpp"
#include "index.hpp"

namespace cpu_kernel_v1 {

struct search_profile {
    double query_setup_ms = 0.0;
    double stage1_cluster_ms = 0.0;
    double stage1_probe_ms = 0.0;
    double stage1_prepare_ms = 0.0;
    double stage1_scan_ms = 0.0;
    double stage1_reduce_ms = 0.0;
    double stage1_reduce_merge_ms = 0.0;
    double stage1_reduce_within_query_merge_ms = 0.0;
    double stage1_reduce_cross_query_merge_ms = 0.0;
    double stage1_reduce_sort_ms = 0.0;
    double stage1_reduce_serial_setup_ms = 0.0;
    double stage1_reduce_partition_accumulate_work_ms = 0.0;
    double stage1_reduce_partition_sort_work_ms = 0.0;
    double stage1_cleanup_ms = 0.0;
    double stage2_1bit_ms = 0.0;
    double stage2_lut_ms = 0.0;
    double stage2_score_docs_ms = 0.0;
    double stage2_select_topk_ms = 0.0;
    double stage2_materialize_ms = 0.0;
    double stage3_exbits_ms = 0.0;
    double stage3_prepare_ms = 0.0;
    double stage3_score_docs_ms = 0.0;
    double stage3_select_topk_ms = 0.0;
    double total_ms = 0.0;
    double end_to_end_ms = 0.0;
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
    std::vector<size_t>& output_ids,
    search_profile* profile = nullptr);

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
    search_profile* profile = nullptr,
    bool print_profile = true);

void accumulate_search_profile(search_profile& accum, const search_profile& sample);
search_profile average_search_profile(const search_profile& total, size_t count);
void print_search_profile(const search_profile& profile);
void print_search_profile_summary(const search_profile& total, size_t count);

}  // namespace cpu_kernel_v1
