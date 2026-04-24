#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cpu_aligned_allocator.hpp"
#include "cpu_kernel_v1.hpp"
#include "gpu_search_options.hpp"
#include "index.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"

namespace cpu_kernel_v3 {

using search_profile = cpu_kernel_v1::search_profile;

enum class stage1_doc_accum_mode {
    sparse_flat_hash,
    sorted_doc_merge_fast,
};

enum class stage1_cluster_scan_mode {
    per_posting_hash,
    doc_run_hash,
};

inline constexpr bool kStage1PrepackEnabled = true;
inline constexpr bool kStage1DenseDocBufferEnabled = false;
inline constexpr stage1_doc_accum_mode kStage1DocAccumMode =
    stage1_doc_accum_mode::sorted_doc_merge_fast;

constexpr const char* stage1_doc_accum_mode_name(stage1_doc_accum_mode mode)
{
    switch (mode) {
        case stage1_doc_accum_mode::sparse_flat_hash:
            return "sparse_flat_hash";
        case stage1_doc_accum_mode::sorted_doc_merge_fast:
            return "sorted_doc_merge_fast";
    }
    return "unknown";
}

inline constexpr const char* kStage1DocAccumModeName =
    stage1_doc_accum_mode_name(kStage1DocAccumMode);

inline constexpr stage1_cluster_scan_mode kStage1ClusterScanMode =
    stage1_cluster_scan_mode::per_posting_hash;

constexpr const char* stage1_cluster_scan_mode_name(stage1_cluster_scan_mode mode)
{
    switch (mode) {
        case stage1_cluster_scan_mode::per_posting_hash:
            return "per_posting_hash";
        case stage1_cluster_scan_mode::doc_run_hash:
            return "doc_run_hash";
    }
    return "unknown";
}

inline constexpr const char* kStage1ClusterScanModeName =
    stage1_cluster_scan_mode_name(kStage1ClusterScanMode);

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

using cpu_kernel_v1::accumulate_search_profile;
using cpu_kernel_v1::average_search_profile;
void print_search_profile(const search_profile& profile);
using cpu_kernel_v1::print_search_profile_summary;

}  // namespace cpu_kernel_v3
