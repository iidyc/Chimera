#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cpu_aligned_allocator.hpp"
#include "gpu_search_options.hpp"
#include "index.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"

namespace Chimera {

namespace cpu_kernel {

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

enum class stage1_doc_accum_mode {
    doc_run_hash = 0,
    sorted_doc_merge_fast = 1,
    sorted_doc_merge_parallel = 2,
    sorted_doc_merge_hybrid = 3,
};

inline constexpr bool kStage1PrepackEnabled = true;
#ifndef CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE
#define CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE 1
#endif
#ifndef CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE
#define CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE 1
#endif
inline constexpr stage1_doc_accum_mode kStage1DocAccumMode =
    static_cast<stage1_doc_accum_mode>(CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE);
inline constexpr bool kStage3DynamicSchedule =
    static_cast<bool>(CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE);

constexpr const char* stage1_doc_accum_mode_name(stage1_doc_accum_mode mode)
{
    switch (mode) {
        case stage1_doc_accum_mode::doc_run_hash:
            return "doc_run_hash";
        case stage1_doc_accum_mode::sorted_doc_merge_fast:
            return "sorted_doc_merge_fast";
        case stage1_doc_accum_mode::sorted_doc_merge_parallel:
            return "sorted_doc_merge_parallel";
        case stage1_doc_accum_mode::sorted_doc_merge_hybrid:
            return "sorted_doc_merge_hybrid";
    }
    return "unknown";
}

inline constexpr const char* kStage1DocAccumModeName =
    stage1_doc_accum_mode_name(kStage1DocAccumMode);

inline constexpr const char* kStage3ScheduleName =
    kStage3DynamicSchedule ? "dynamic_1" : "balanced_static_buckets";

struct repacked_cluster {
    aligned_vector_64<uint8_t> packed_codes;
    aligned_vector_64<float> packed_factors;
    aligned_vector_64<int> packed_doc_ids;
};

struct packed_cluster_pos {
    size_t start_token = 0;
    size_t token_count = 0;
};

struct packed_token_stream_cache {
    packed_token_stream_cache() = default;
    explicit packed_token_stream_cache(const cpu_mvr_index& index);

    [[nodiscard]] const uint8_t* batch_codes(size_t batch_idx) const;
    [[nodiscard]] const float* factors() const { return packed_factors.data(); }
    [[nodiscard]] size_t batch_count() const { return packed_factors.size() / rabitqlib::fastscan::kBatchSize; }
    [[nodiscard]] size_t original_code_bytes() const { return original_code_bytes_; }
    [[nodiscard]] size_t packed_code_bytes() const { return packed_codes.size(); }
    [[nodiscard]] size_t packed_factor_bytes() const { return packed_factors.size() * sizeof(float); }
    [[nodiscard]] size_t packed_total_bytes() const { return packed_code_bytes() + packed_factor_bytes(); }

   private:
    size_t code_bytes_per_vector_ = 0;
    size_t padded_dim_ = 0;
    size_t original_code_bytes_ = 0;
    aligned_vector_64<uint8_t> packed_codes;
    aligned_vector_64<float> packed_factors;
};

class clustered_stage1_cache {
   public:
    clustered_stage1_cache(const std::string& index_path, const cpu_mvr_index& index);
    ~clustered_stage1_cache();

    clustered_stage1_cache(const clustered_stage1_cache&) = delete;
    clustered_stage1_cache& operator=(const clustered_stage1_cache&) = delete;
    clustered_stage1_cache(clustered_stage1_cache&&) noexcept;
    clustered_stage1_cache& operator=(clustered_stage1_cache&&) noexcept;

    [[nodiscard]] size_t code_bytes_per_vector() const { return code_bytes_per_vector_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    [[nodiscard]] const char* code_data() const { return code_data_; }
    [[nodiscard]] const float* factor_data() const { return factor_data_; }
    [[nodiscard]] const int* doc_id_data() const { return doc_id_data_; }
    [[nodiscard]] const packed_cluster_pos& packed_cluster_range(size_t cluster_id) const {
        return packed_cluster_pos_[cluster_id];
    }
    [[nodiscard]] size_t packed_cluster_token_count(size_t cluster_id) const {
        return packed_cluster_pos_[cluster_id].token_count;
    }
    [[nodiscard]] size_t packed_cluster_padded_tokens(size_t cluster_id) const {
        const size_t next_start = (cluster_id + 1 < packed_cluster_pos_.size())
                                      ? packed_cluster_pos_[cluster_id + 1].start_token
                                      : packed_factors_.size();
        return next_start - packed_cluster_pos_[cluster_id].start_token;
    }
    [[nodiscard]] const uint8_t* packed_cluster_codes(size_t cluster_id) const {
        return packed_codes_.data() +
               packed_cluster_pos_[cluster_id].start_token * code_bytes_per_vector_;
    }
    [[nodiscard]] const float* packed_cluster_factors(size_t cluster_id) const {
        return packed_factors_.data() + packed_cluster_pos_[cluster_id].start_token;
    }
    [[nodiscard]] const int* packed_cluster_doc_ids(size_t cluster_id) const {
        return packed_doc_ids_.data() + packed_cluster_pos_[cluster_id].start_token;
    }
    [[nodiscard]] size_t original_code_bytes() const { return original_code_bytes_; }
    [[nodiscard]] size_t packed_code_bytes() const { return packed_code_bytes_; }
    [[nodiscard]] size_t packed_factor_bytes() const { return packed_factor_bytes_; }
    [[nodiscard]] size_t packed_doc_id_bytes() const { return packed_doc_id_bytes_; }
    [[nodiscard]] size_t packed_total_bytes() const {
        return packed_code_bytes_ + packed_factor_bytes_ + packed_doc_id_bytes_;
    }
    [[nodiscard]] bool has_prepacked_clusters() const { return !packed_cluster_pos_.empty(); }

   private:
    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t mapping_bytes_ = 0;

    size_t n_entries_ = 0;
    size_t code_bytes_per_vector_ = 0;
    size_t padded_dim_ = 0;
    size_t original_code_bytes_ = 0;
    size_t packed_code_bytes_ = 0;
    size_t packed_factor_bytes_ = 0;
    size_t packed_doc_id_bytes_ = 0;

    const char* code_data_ = nullptr;
    const float* factor_data_ = nullptr;
    const int* doc_id_data_ = nullptr;
    aligned_vector_64<char> raw_codes_;
    aligned_vector_64<float> raw_factors_;
    aligned_vector_64<int> raw_doc_ids_;
    std::vector<packed_cluster_pos> packed_cluster_pos_;
    aligned_vector_64<uint8_t> packed_codes_;
    aligned_vector_64<float> packed_factors_;
    aligned_vector_64<int> packed_doc_ids_;

    friend void rank_cluster_dists(
        const cpu_mvr_index& index,
        const clustered_stage1_cache& stage1_cache,
        query_object* queries,
        size_t q_doclen,
        size_t nprobe,
        size_t k,
        std::vector<size_t>& output_ids);
};

void rank_cluster_dists(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    query_object* queries,
    size_t q_doclen,
    size_t nprobe,
    size_t k,
    std::vector<size_t>& output_ids,
    search_profile* profile = nullptr);

std::vector<size_t> search(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    const packed_token_stream_cache& stage2_cache,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime);

std::vector<size_t> search_profiled(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    const packed_token_stream_cache& stage2_cache,
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

}  // namespace cpu_kernel


}  // namespace Chimera
