#include "cpu_kernel_v2.hpp"

#include <cerrno>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clustered_stage1_file_format.hpp"
#include "gpu_index_layout.hpp"
#include "query.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/index/lut.hpp"
#include "rabitqlib/index/query.hpp"

namespace cpu_kernel_v2 {

namespace {

uint8_t reverse_nibble_bits(uint8_t nibble) {
    return static_cast<uint8_t>(((nibble & 0x1u) << 3) |
                                ((nibble & 0x2u) << 1) |
                                ((nibble & 0x4u) >> 1) |
                                ((nibble & 0x8u) >> 3));
}

uint8_t swap_and_reverse_nibbles(uint8_t value) {
    const uint8_t lo = reverse_nibble_bits(static_cast<uint8_t>(value & 0x0Fu));
    const uint8_t hi = reverse_nibble_bits(static_cast<uint8_t>((value >> 4) & 0x0Fu));
    return static_cast<uint8_t>(hi | (lo << 4));
}

size_t cluster_batch_code_bytes(size_t code_bytes_per_vector) {
    return code_bytes_per_vector * rabitqlib::fastscan::kBatchSize;
}

size_t token_stream_batch_code_bytes(size_t code_bytes_per_vector) {
    return code_bytes_per_vector * rabitqlib::fastscan::kBatchSize;
}

size_t padded_cluster_tokens(size_t cluster_size)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    return ((cluster_size + batch_size - 1) / batch_size) * batch_size;
}

std::vector<uint8_t>& repack_source_codes_scratch(size_t code_bytes_per_vector)
{
    thread_local std::vector<uint8_t> scratch;
    const size_t needed = rabitqlib::fastscan::kBatchSize * code_bytes_per_vector;
    if (scratch.size() != needed) {
        scratch.resize(needed);
    }
    return scratch;
}

repacked_cluster repack_cluster(
    const clustered_stage1_cache& stage1_cache,
    size_t cluster_start,
    size_t cluster_size)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    const size_t num_batches = (cluster_size + batch_size - 1) / batch_size;
    const size_t bytes_per_batch =
        cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());
    const size_t padded_tokens = num_batches * batch_size;

    repacked_cluster out;
    out.packed_codes.resize(num_batches * bytes_per_batch);
    out.packed_factors.resize(padded_tokens, 0.0f);
    out.packed_doc_ids.resize(padded_tokens, -1);

    auto& source_codes = repack_source_codes_scratch(stage1_cache.code_bytes_per_vector());

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const size_t token_base = cluster_start + batch_idx * batch_size;
        const size_t valid = std::min(batch_size, cluster_size - batch_idx * batch_size);

        std::fill(source_codes.begin(), source_codes.end(), 0);
        for (size_t lane = 0; lane < valid; ++lane) {
            const size_t token_idx = token_base + lane;
            uint8_t* dst = source_codes.data() + lane * stage1_cache.code_bytes_per_vector();
            const uint8_t* src = reinterpret_cast<const uint8_t*>(
                stage1_cache.code_data() + token_idx * stage1_cache.code_bytes_per_vector());
            out.packed_factors[batch_idx * batch_size + lane] = stage1_cache.factor_data()[token_idx];
            out.packed_doc_ids[batch_idx * batch_size + lane] = stage1_cache.doc_id_data()[token_idx];
            for (size_t b = 0; b < stage1_cache.code_bytes_per_vector(); ++b) {
                // cluster_1bit.bin stores each byte in natural packed bit order
                // (dims 0..3 in the low nibble, dims 4..7 in the high nibble).
                // fastscan::pack_codes() consumes the two nibbles in the opposite
                // order, and the nibble value itself is interpreted with reversed
                // bit significance. Repack accordingly before handing the batch to
                // the AVX fastscan kernel.
                dst[b] = swap_and_reverse_nibbles(src[b]);
            }
        }

        rabitqlib::fastscan::pack_codes(
            stage1_cache.padded_dim(),
            source_codes.data(),
            valid,
            out.packed_codes.data() + batch_idx * bytes_per_batch);
    }

    return out;
}

void repack_cluster_into(
    const clustered_stage1_cache& stage1_cache,
    size_t cluster_start,
    size_t cluster_size,
    uint8_t* packed_codes,
    float* packed_factors,
    int* packed_doc_ids)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    const size_t num_batches = (cluster_size + batch_size - 1) / batch_size;
    const size_t bytes_per_batch =
        cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());

    std::fill(packed_factors, packed_factors + num_batches * batch_size, 0.0f);
    std::fill(packed_doc_ids, packed_doc_ids + num_batches * batch_size, -1);

    auto& source_codes = repack_source_codes_scratch(stage1_cache.code_bytes_per_vector());

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const size_t token_base = cluster_start + batch_idx * batch_size;
        const size_t valid = std::min(batch_size, cluster_size - batch_idx * batch_size);

        std::fill(source_codes.begin(), source_codes.end(), 0);
        for (size_t lane = 0; lane < valid; ++lane) {
            const size_t token_idx = token_base + lane;
            uint8_t* dst = source_codes.data() + lane * stage1_cache.code_bytes_per_vector();
            const uint8_t* src = reinterpret_cast<const uint8_t*>(
                stage1_cache.code_data() + token_idx * stage1_cache.code_bytes_per_vector());
            packed_factors[batch_idx * batch_size + lane] = stage1_cache.factor_data()[token_idx];
            packed_doc_ids[batch_idx * batch_size + lane] = stage1_cache.doc_id_data()[token_idx];
            for (size_t b = 0; b < stage1_cache.code_bytes_per_vector(); ++b) {
                dst[b] = swap_and_reverse_nibbles(src[b]);
            }
        }

        rabitqlib::fastscan::pack_codes(
            stage1_cache.padded_dim(),
            source_codes.data(),
            valid,
            packed_codes + batch_idx * bytes_per_batch);
    }
}

float stage2_score_doc_fastscan(
    const cpu_mvr_index& index,
    const packed_token_stream_cache& stage2_cache,
    const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
    query_object* queries,
    size_t q_doclen,
    size_t doc_id)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    const float* factors = stage2_cache.factors();
    const size_t doc_start = static_cast<size_t>(index.doc_ptrs_[doc_id]);
    const size_t doc_length = index.doc_len(doc_id);
    const size_t batch_begin = doc_start / batch_size;
    const size_t batch_end = (doc_start + doc_length - 1) / batch_size;
    float doc_score = 0.0f;
    std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

    for (size_t j = 0; j < q_doclen; ++j) {
        const auto& lut = *luts[j];
        float max_token_score = -std::numeric_limits<float>::infinity();
        for (size_t batch_idx = batch_begin; batch_idx <= batch_end; ++batch_idx) {
            rabitqlib::fastscan::accumulate_hacc(
                stage2_cache.batch_codes(batch_idx),
                lut.lut(),
                accum.data(),
                index.padded_dim_);

            const size_t batch_token_start = batch_idx * batch_size;
            const size_t overlap_begin = std::max(doc_start, batch_token_start);
            const size_t overlap_end = std::min(doc_start + doc_length, batch_token_start + batch_size);

            for (size_t tid = overlap_begin; tid < overlap_end; ++tid) {
                const size_t lane = tid - batch_token_start;
                const float ip =
                    lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
                const float dist =
                    (ip - queries[j].cb1_sumq) * factors[batch_token_start + lane];
                max_token_score = std::max(max_token_score, dist);
            }
        }
        doc_score += max_token_score;
    }

    return doc_score;
}

void stage2_materialize_doc_dists_fastscan(
    const cpu_mvr_index& index,
    const packed_token_stream_cache& stage2_cache,
    const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
    query_object* queries,
    size_t q_doclen,
    size_t doc_id,
    float* out_doc_token_dists)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    const float* factors = stage2_cache.factors();
    const size_t doc_start = static_cast<size_t>(index.doc_ptrs_[doc_id]);
    const size_t doc_length = index.doc_len(doc_id);
    const size_t batch_begin = doc_start / batch_size;
    const size_t batch_end = (doc_start + doc_length - 1) / batch_size;
    std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

    for (size_t j = 0; j < q_doclen; ++j) {
        const auto& lut = *luts[j];
        for (size_t batch_idx = batch_begin; batch_idx <= batch_end; ++batch_idx) {
            rabitqlib::fastscan::accumulate_hacc(
                stage2_cache.batch_codes(batch_idx),
                lut.lut(),
                accum.data(),
                index.padded_dim_);

            const size_t batch_token_start = batch_idx * batch_size;
            const size_t overlap_begin = std::max(doc_start, batch_token_start);
            const size_t overlap_end = std::min(doc_start + doc_length, batch_token_start + batch_size);

            for (size_t tid = overlap_begin; tid < overlap_end; ++tid) {
                const size_t lane = tid - batch_token_start;
                const size_t local_token_idx = tid - doc_start;
                const float ip =
                    lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
                out_doc_token_dists[local_token_idx * q_doclen + j] =
                    (ip - queries[j].cb1_sumq) * factors[batch_token_start + lane];
            }
        }
    }
}

void rank_all_tokens_1bit_fastscan(
    const cpu_mvr_index& index,
    const packed_token_stream_cache& stage2_cache,
    query_object* queries,
    size_t q_doclen,
    const std::vector<size_t>& input_ids,
    size_t k,
    std::vector<size_t>& output_ids,
    std::vector<float>& one_bit_dists)
{
    output_ids.clear();
    one_bit_dists.clear();

    std::vector<std::unique_ptr<rabitqlib::Lut<float>>> luts;
    luts.reserve(q_doclen);
    for (size_t j = 0; j < q_doclen; ++j) {
        luts.emplace_back(std::make_unique<rabitqlib::Lut<float>>(queries[j].rotated_query, index.padded_dim_, true));
    }

    std::vector<float> doc_scores(input_ids.size(), 0.0f);
#pragma omp parallel for
    for (size_t idx = 0; idx < input_ids.size(); ++idx) {
        doc_scores[idx] = stage2_score_doc_fastscan(
            index,
            stage2_cache,
            luts,
            queries,
            q_doclen,
            input_ids[idx]);
    }

    const size_t topk_count = std::min(k, input_ids.size());
    std::vector<size_t> selected_doc_indices(input_ids.size());
    std::iota(selected_doc_indices.begin(), selected_doc_indices.end(), 0);
    auto score_order = [&doc_scores](size_t a, size_t b) {
        if (doc_scores[a] != doc_scores[b]) return doc_scores[a] > doc_scores[b];
        return a < b;
    };
    std::partial_sort(
        selected_doc_indices.begin(),
        selected_doc_indices.begin() + topk_count,
        selected_doc_indices.end(),
        score_order);
    selected_doc_indices.resize(topk_count);

    output_ids.resize(topk_count);
    std::vector<size_t> selected_doc_ptrs(topk_count + 1, 0);
    for (size_t i = 0; i < topk_count; ++i) {
        output_ids[i] = input_ids[selected_doc_indices[i]];
        selected_doc_ptrs[i + 1] = selected_doc_ptrs[i] + index.doc_len(output_ids[i]);
    }

    one_bit_dists.resize(selected_doc_ptrs.back() * q_doclen);
#pragma omp parallel for
    for (size_t idx = 0; idx < topk_count; ++idx) {
        const size_t doc_id = output_ids[idx];
        stage2_materialize_doc_dists_fastscan(
            index,
            stage2_cache,
            luts,
            queries,
            q_doclen,
            doc_id,
            one_bit_dists.data() + selected_doc_ptrs[idx] * q_doclen);
    }
}

void build_query_objects(
    const cpu_mvr_index& index,
    const float* queries,
    size_t q_doclen,
    std::vector<float>& rotated_queries,
    std::vector<query_object>& query_objs)
{
    rotated_queries.resize(q_doclen * index.padded_dim_);
    for (size_t i = 0; i < q_doclen; ++i) {
        index.rotator_->rotate(&queries[i * index.d], &rotated_queries[i * index.padded_dim_]);
    }

    query_objs.resize(q_doclen);
    for (size_t i = 0; i < q_doclen; ++i) {
        query_objs[i] = query_object(&rotated_queries[i * index.padded_dim_], index.padded_dim_, index.ex_bits);
    }
}

using steady_clock = std::chrono::steady_clock;

inline double ms_between(steady_clock::time_point a, steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<size_t> search_impl(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    const packed_token_stream_cache& stage2_cache,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime,
    search_profile* profile)
{
    std::vector<float> rotated_queries;
    std::vector<query_object> query_objs;
    build_query_objects(index, queries, q_doclen, rotated_queries, query_objs);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<size_t> rank_cluster_doc_ids;
    rank_cluster_dists(
        index,
        stage1_cache,
        query_objs.data(),
        q_doclen,
        static_cast<size_t>(runtime.nprobe),
        static_cast<size_t>(runtime.k_rank_cluster),
        rank_cluster_doc_ids,
        profile);
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<size_t> rank_all_tokens_ids;
    std::vector<float> one_bit_dists;
    rank_all_tokens_1bit_fastscan(
        index,
        stage2_cache,
        query_objs.data(),
        q_doclen,
        rank_cluster_doc_ids,
        static_cast<size_t>(runtime.k_rank_all_tokens),
        rank_all_tokens_ids,
        one_bit_dists);
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<size_t> result;
    cpu_kernel_v1::rank_all_tokens_exbits(
        index,
        query_objs.data(),
        q_doclen,
        rank_all_tokens_ids,
        one_bit_dists,
        k,
        result);
    const auto t3 = std::chrono::high_resolution_clock::now();

    if (profile != nullptr) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        profile->stage1_cluster_ms = ms(t0, t1);
        profile->stage2_1bit_ms = ms(t1, t2);
        profile->stage3_exbits_ms = ms(t2, t3);
        profile->total_ms = ms(t0, t3);
    }

    return result;
}

}  // namespace

clustered_stage1_cache::clustered_stage1_cache(
    const std::string& index_path,
    const cpu_mvr_index& index)
{
    const auto actual_cluster_path = gpu_index_layout::cluster_1bit_path(
        gpu_index_layout::index_dir_from_index_path(index_path));

    std::ifstream input(actual_cluster_path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open clustered stage-1 sidecar: " + actual_cluster_path);
    }

    const auto header = clustered_stage1_file_format::read_header(input, actual_cluster_path);
    n_entries_ = header.n_entries;
    code_bytes_per_vector_ = header.code_bytes_per_vector;
    padded_dim_ = header.padded_dim == 0 ? index.padded_dim_ : header.padded_dim;

    if (n_entries_ != index.n) {
        throw std::runtime_error(
            "cluster_1bit.bin n_entries mismatch: expected " +
            std::to_string(index.n) + ", got " + std::to_string(n_entries_));
    }
    if (padded_dim_ != index.padded_dim_) {
        throw std::runtime_error(
            "cluster_1bit.bin padded_dim mismatch: expected " +
            std::to_string(index.padded_dim_) + ", got " + std::to_string(padded_dim_));
    }
    if (code_bytes_per_vector_ != index.padded_dim_ / 8) {
        throw std::runtime_error(
            "cluster_1bit.bin code_bytes_per_vector mismatch: expected " +
            std::to_string(index.padded_dim_ / 8) + ", got " +
            std::to_string(code_bytes_per_vector_));
    }

    input.close();

    mapping_bytes_ = std::filesystem::file_size(actual_cluster_path);
    fd_ = open(actual_cluster_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open clustered stage-1 sidecar fd: " + actual_cluster_path);
    }
    mapping_ = mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
        const auto err = errno;
        close(fd_);
        fd_ = -1;
        mapping_ = nullptr;
        throw std::runtime_error(
            "Failed to mmap clustered stage-1 sidecar " + actual_cluster_path + ": " +
            std::system_category().message(err));
    }

    const size_t prefix = clustered_stage1_file_format::prefix_bytes(header);
    const size_t code_bytes = n_entries_ * code_bytes_per_vector_;
    const size_t factor_offset = prefix + code_bytes;
    const size_t doc_id_offset = factor_offset + n_entries_ * sizeof(float);
    const size_t required_bytes = doc_id_offset + n_entries_ * sizeof(int);
    if (mapping_bytes_ < required_bytes) {
        throw std::runtime_error("cluster_1bit.bin is truncated: " + actual_cluster_path);
    }

    const char* base = static_cast<const char*>(mapping_);
    original_code_bytes_ = code_bytes;

#if defined(GPU_MVR_CPU_V2_STAGE1_PREPACK) && GPU_MVR_CPU_V2_STAGE1_PREPACK
    code_data_ = base + prefix;
    factor_data_ = reinterpret_cast<const float*>(base + factor_offset);
    doc_id_data_ = reinterpret_cast<const int*>(base + doc_id_offset);
    const size_t n_clusters = index.ivf->cluster_pos.size() > 0 ? index.ivf->cluster_pos.size() - 1 : 0;
    packed_cluster_pos_.resize(n_clusters);
    size_t total_padded_tokens = 0;
    for (size_t cid = 0; cid < n_clusters; ++cid) {
        const size_t cluster_start = index.ivf->cluster_pos[cid];
        const size_t cluster_size = index.ivf->cluster_pos[cid + 1] - cluster_start;
        const size_t padded_tokens = padded_cluster_tokens(cluster_size);
        packed_cluster_pos_[cid].start_token = total_padded_tokens;
        packed_cluster_pos_[cid].token_count = cluster_size;
        total_padded_tokens += padded_tokens;
    }

    packed_code_bytes_ = total_padded_tokens * code_bytes_per_vector_;
    packed_factor_bytes_ = total_padded_tokens * sizeof(float);
    packed_doc_id_bytes_ = total_padded_tokens * sizeof(int);
    packed_codes_.resize(packed_code_bytes_);
    packed_factors_.resize(total_padded_tokens, 0.0f);
    packed_doc_ids_.resize(total_padded_tokens, -1);

#pragma omp parallel for
    for (size_t cid = 0; cid < n_clusters; ++cid) {
        const size_t cluster_start = index.ivf->cluster_pos[cid];
        const size_t cluster_size = index.ivf->cluster_pos[cid + 1] - cluster_start;
        const size_t packed_token_start = packed_cluster_pos_[cid].start_token;
        repack_cluster_into(
            *this,
            cluster_start,
            cluster_size,
            packed_codes_.data() + packed_token_start * code_bytes_per_vector_,
            packed_factors_.data() + packed_token_start,
            packed_doc_ids_.data() + packed_token_start);
    }

    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_bytes_);
        mapping_ = nullptr;
        close(fd_);
        fd_ = -1;
        code_data_ = nullptr;
        factor_data_ = nullptr;
        doc_id_data_ = nullptr;
    }
#else
    raw_codes_.resize(code_bytes);
    raw_factors_.resize(n_entries_);
    raw_doc_ids_.resize(n_entries_);
    std::memcpy(raw_codes_.data(), base + prefix, code_bytes);
    std::memcpy(raw_factors_.data(), base + factor_offset, n_entries_ * sizeof(float));
    std::memcpy(raw_doc_ids_.data(), base + doc_id_offset, n_entries_ * sizeof(int));
    code_data_ = raw_codes_.data();
    factor_data_ = raw_factors_.data();
    doc_id_data_ = raw_doc_ids_.data();

    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_bytes_);
        mapping_ = nullptr;
        close(fd_);
        fd_ = -1;
    }
#endif
}

packed_token_stream_cache::packed_token_stream_cache(const cpu_mvr_index& index)
{
    const size_t batch_size = rabitqlib::fastscan::kBatchSize;
    code_bytes_per_vector_ = index.padded_dim_ / 8;
    padded_dim_ = index.padded_dim_;
    original_code_bytes_ = index.n * code_bytes_per_vector_;
    const size_t num_batches = (index.n + batch_size - 1) / batch_size;
    const size_t bytes_per_batch = token_stream_batch_code_bytes(code_bytes_per_vector_);
    const size_t padded_tokens = num_batches * batch_size;
    packed_codes.resize(num_batches * bytes_per_batch);
    packed_factors.resize(padded_tokens, 0.0f);

    std::vector<uint8_t> source_codes(batch_size * code_bytes_per_vector_);
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const size_t token_base = batch_idx * batch_size;
        const size_t valid = std::min(batch_size, index.n - batch_idx * batch_size);
        std::fill(source_codes.begin(), source_codes.end(), 0);
        for (size_t lane = 0; lane < valid; ++lane) {
            const size_t token_idx = token_base + lane;
            uint8_t* dst = source_codes.data() + lane * code_bytes_per_vector_;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(
                index.one_bit_code_.data() + token_idx * code_bytes_per_vector_);
            packed_factors[batch_idx * batch_size + lane] = index.one_bit_factor_[token_idx];
            for (size_t b = 0; b < code_bytes_per_vector_; ++b) {
                dst[b] = swap_and_reverse_nibbles(src[b]);
            }
        }
        rabitqlib::fastscan::pack_codes(
            index.padded_dim_,
            source_codes.data(),
            valid,
            packed_codes.data() + batch_idx * bytes_per_batch);
    }
}

const uint8_t* packed_token_stream_cache::batch_codes(size_t batch_idx) const
{
    return packed_codes.data() + batch_idx * token_stream_batch_code_bytes(code_bytes_per_vector_);
}

clustered_stage1_cache::~clustered_stage1_cache()
{
    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_bytes_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

clustered_stage1_cache::clustered_stage1_cache(clustered_stage1_cache&& other) noexcept
    : fd_(other.fd_),
      mapping_(other.mapping_),
      mapping_bytes_(other.mapping_bytes_),
      n_entries_(other.n_entries_),
      code_bytes_per_vector_(other.code_bytes_per_vector_),
      padded_dim_(other.padded_dim_),
      original_code_bytes_(other.original_code_bytes_),
      packed_code_bytes_(other.packed_code_bytes_),
      packed_factor_bytes_(other.packed_factor_bytes_),
      packed_doc_id_bytes_(other.packed_doc_id_bytes_),
      code_data_(other.code_data_),
      factor_data_(other.factor_data_),
      doc_id_data_(other.doc_id_data_),
      raw_codes_(std::move(other.raw_codes_)),
      raw_factors_(std::move(other.raw_factors_)),
      raw_doc_ids_(std::move(other.raw_doc_ids_)),
      packed_cluster_pos_(std::move(other.packed_cluster_pos_)),
      packed_codes_(std::move(other.packed_codes_)),
      packed_factors_(std::move(other.packed_factors_)),
      packed_doc_ids_(std::move(other.packed_doc_ids_))
{
    if (!raw_codes_.empty()) {
        code_data_ = raw_codes_.data();
        factor_data_ = raw_factors_.data();
        doc_id_data_ = raw_doc_ids_.data();
    }
    other.fd_ = -1;
    other.mapping_ = nullptr;
    other.mapping_bytes_ = 0;
    other.original_code_bytes_ = 0;
    other.packed_code_bytes_ = 0;
    other.packed_factor_bytes_ = 0;
    other.packed_doc_id_bytes_ = 0;
    other.code_data_ = nullptr;
    other.factor_data_ = nullptr;
    other.doc_id_data_ = nullptr;
    other.raw_codes_.clear();
    other.raw_factors_.clear();
    other.raw_doc_ids_.clear();
    other.packed_cluster_pos_.clear();
    other.packed_codes_.clear();
    other.packed_factors_.clear();
    other.packed_doc_ids_.clear();
}

clustered_stage1_cache& clustered_stage1_cache::operator=(clustered_stage1_cache&& other) noexcept
{
    if (this == &other) return *this;
    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_bytes_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }

    fd_ = other.fd_;
    mapping_ = other.mapping_;
    mapping_bytes_ = other.mapping_bytes_;
    n_entries_ = other.n_entries_;
    code_bytes_per_vector_ = other.code_bytes_per_vector_;
    padded_dim_ = other.padded_dim_;
    original_code_bytes_ = other.original_code_bytes_;
    packed_code_bytes_ = other.packed_code_bytes_;
    packed_factor_bytes_ = other.packed_factor_bytes_;
    packed_doc_id_bytes_ = other.packed_doc_id_bytes_;
    code_data_ = other.code_data_;
    factor_data_ = other.factor_data_;
    doc_id_data_ = other.doc_id_data_;
    raw_codes_ = std::move(other.raw_codes_);
    raw_factors_ = std::move(other.raw_factors_);
    raw_doc_ids_ = std::move(other.raw_doc_ids_);
    packed_cluster_pos_ = std::move(other.packed_cluster_pos_);
    packed_codes_ = std::move(other.packed_codes_);
    packed_factors_ = std::move(other.packed_factors_);
    packed_doc_ids_ = std::move(other.packed_doc_ids_);
    if (!raw_codes_.empty()) {
        code_data_ = raw_codes_.data();
        factor_data_ = raw_factors_.data();
        doc_id_data_ = raw_doc_ids_.data();
    }

    other.fd_ = -1;
    other.mapping_ = nullptr;
    other.mapping_bytes_ = 0;
    other.original_code_bytes_ = 0;
    other.packed_code_bytes_ = 0;
    other.packed_factor_bytes_ = 0;
    other.packed_doc_id_bytes_ = 0;
    other.code_data_ = nullptr;
    other.factor_data_ = nullptr;
    other.doc_id_data_ = nullptr;
    other.raw_codes_.clear();
    other.raw_factors_.clear();
    other.raw_doc_ids_.clear();
    other.packed_cluster_pos_.clear();
    other.packed_codes_.clear();
    other.packed_factors_.clear();
    other.packed_doc_ids_.clear();
    return *this;
}

void rank_cluster_dists(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    query_object* queries,
    size_t q_doclen,
    size_t nprobe,
    size_t k,
    std::vector<size_t>& output_ids,
    search_profile* profile)
{
    output_ids.clear();

    const auto t_stage1_start = steady_clock::now();
    std::vector<std::vector<size_t>> clusters_per_query(q_doclen);
    if (auto* hnsw = dynamic_cast<PG_HNSW*>(index.ivf->pg_index)) {
        hnsw->hnsw_index->setEf(PG_HNSW::search_ef_for_k(nprobe));
#pragma omp parallel for
        for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
            auto result = hnsw->hnsw_index->searchKnn(queries[j].rotated_query, nprobe);
            auto& clusters = clusters_per_query[static_cast<size_t>(j)];
            while (!result.empty()) {
                clusters.push_back(static_cast<size_t>(result.top().second));
                result.pop();
            }
        }
    } else {
        for (size_t j = 0; j < q_doclen; ++j) {
            index.ivf->pg_index->search(queries[j].rotated_query, nprobe, clusters_per_query[j]);
        }
    }
    const auto t_probe_done = steady_clock::now();

    std::vector<std::unordered_map<size_t, repacked_cluster>> local_repacked_by_query;
#if !defined(GPU_MVR_CPU_V2_STAGE1_PREPACK) || !GPU_MVR_CPU_V2_STAGE1_PREPACK
    local_repacked_by_query.resize(q_doclen);
    for (size_t j = 0; j < q_doclen; ++j) {
        auto& cache = local_repacked_by_query[j];
        cache.reserve(clusters_per_query[j].size());
        for (size_t cid : clusters_per_query[j]) {
            if (cache.find(cid) != cache.end()) continue;
            const size_t cluster_start = index.ivf->cluster_pos[cid];
            const size_t cluster_size = index.ivf->cluster_pos[cid + 1] - cluster_start;
            cache.emplace(cid, repack_cluster(stage1_cache, cluster_start, cluster_size));
        }
    }
#endif
    const auto t_prepare_done = steady_clock::now();

    std::vector<std::unordered_map<int, float>> query_doc_max(q_doclen);
#pragma omp parallel for
    for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
        const rabitqlib::Lut<float> lut(
            queries[j].rotated_query,
            index.padded_dim_,
            true);

        auto& doc_max = query_doc_max[static_cast<size_t>(j)];
        doc_max.reserve(clusters_per_query[static_cast<size_t>(j)].size() * 64);

        std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

        for (size_t cid : clusters_per_query[j]) {
            size_t cluster_size = 0;
#if defined(GPU_MVR_CPU_V2_STAGE1_PREPACK) && GPU_MVR_CPU_V2_STAGE1_PREPACK
            const uint8_t* packed_codes = stage1_cache.packed_cluster_codes(cid);
            const float* packed_factors = stage1_cache.packed_cluster_factors(cid);
            const int* packed_doc_ids = stage1_cache.packed_cluster_doc_ids(cid);
            cluster_size = stage1_cache.packed_cluster_token_count(cid);
#else
            const auto& repacked = local_repacked_by_query[static_cast<size_t>(j)].at(cid);
            const uint8_t* packed_codes = repacked.packed_codes.data();
            const float* packed_factors = repacked.packed_factors.data();
            const int* packed_doc_ids = repacked.packed_doc_ids.data();
            const size_t cluster_start = index.ivf->cluster_pos[cid];
            cluster_size = index.ivf->cluster_pos[cid + 1] - cluster_start;
#endif
            const size_t num_batches =
                (cluster_size + rabitqlib::fastscan::kBatchSize - 1) /
                rabitqlib::fastscan::kBatchSize;

            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                const uint8_t* batch_ptr =
                    packed_codes +
                    batch_idx * cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());
                rabitqlib::fastscan::accumulate_hacc(
                    batch_ptr,
                    lut.lut(),
                    accum.data(),
                    index.padded_dim_);

                const size_t valid =
                    std::min<size_t>(rabitqlib::fastscan::kBatchSize, cluster_size - batch_idx * rabitqlib::fastscan::kBatchSize);
                const size_t lane_base = batch_idx * rabitqlib::fastscan::kBatchSize;

                for (size_t lane = 0; lane < valid; ++lane) {
                    const int doc_id = packed_doc_ids[lane_base + lane];
                    const float ip =
                        lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
                    const float dist =
                        (ip - queries[j].cb1_sumq) * packed_factors[lane_base + lane];
                    auto [it, inserted] = doc_max.try_emplace(doc_id, dist);
                    if (!inserted && dist > it->second) {
                        it->second = dist;
                    }
                }
            }
        }
    }
    const auto t_scan_done = steady_clock::now();

    size_t total_touched_docs = 0;
    for (const auto& doc_max : query_doc_max) {
        total_touched_docs += doc_max.size();
    }

    std::vector<std::pair<int, float>> all_doc_scores;
    all_doc_scores.reserve(total_touched_docs);
    for (const auto& doc_max : query_doc_max) {
        for (const auto& [doc_id, score] : doc_max) {
            all_doc_scores.emplace_back(doc_id, score);
        }
    }

    std::sort(
        all_doc_scores.begin(),
        all_doc_scores.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<float, int>> reduced_scores;
    reduced_scores.reserve(all_doc_scores.size());
    for (size_t i = 0; i < all_doc_scores.size();) {
        const int doc_id = all_doc_scores[i].first;
        float score_sum = 0.0f;
        do {
            score_sum += all_doc_scores[i].second;
            ++i;
        } while (i < all_doc_scores.size() && all_doc_scores[i].first == doc_id);
        reduced_scores.emplace_back(score_sum, doc_id);
    }

    const size_t topk_count = std::min(k, reduced_scores.size());
    auto topk_order = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    if (topk_count < reduced_scores.size()) {
        std::partial_sort(
            reduced_scores.begin(),
            reduced_scores.begin() + topk_count,
            reduced_scores.end(),
            topk_order);
        reduced_scores.resize(topk_count);
    } else {
        std::sort(reduced_scores.begin(), reduced_scores.end(), topk_order);
    }

    output_ids.reserve(topk_count);
    for (const auto& [score, doc_id] : reduced_scores) {
        (void)score;
        output_ids.push_back(static_cast<size_t>(doc_id));
    }
    const auto t_reduce_done = steady_clock::now();

    if (profile != nullptr) {
        profile->stage1_probe_ms = ms_between(t_stage1_start, t_probe_done);
        profile->stage1_prepare_ms = ms_between(t_probe_done, t_prepare_done);
        profile->stage1_scan_ms = ms_between(t_prepare_done, t_scan_done);
        profile->stage1_reduce_ms = ms_between(t_scan_done, t_reduce_done);
    }
}

std::vector<size_t> search(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    const packed_token_stream_cache& stage2_cache,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime)
{
    return search_impl(index, stage1_cache, stage2_cache, queries, q_doclen, k, runtime, nullptr);
}

std::vector<size_t> search_profiled(
    const cpu_mvr_index& index,
    const clustered_stage1_cache& stage1_cache,
    const packed_token_stream_cache& stage2_cache,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime,
    search_profile* profile)
{
    search_profile local_profile;
    std::vector<size_t> result =
        search_impl(index, stage1_cache, stage2_cache, queries, q_doclen, k, runtime, &local_profile);
    if (profile != nullptr) {
        *profile = local_profile;
    }
    cpu_kernel_v2::print_search_profile(local_profile);
    return result;
}

void print_search_profile(const search_profile& profile)
{
    std::cout << "[search] stage1_cluster: " << profile.stage1_cluster_ms << " ms, "
              << "stage1_probe: " << profile.stage1_probe_ms << " ms, "
              << "stage1_prepare: " << profile.stage1_prepare_ms << " ms, "
              << "stage1_scan: " << profile.stage1_scan_ms << " ms, "
              << "stage1_reduce: " << profile.stage1_reduce_ms << " ms, "
              << "stage2_1bit: " << profile.stage2_1bit_ms << " ms, "
              << "stage3_exbits: " << profile.stage3_exbits_ms << " ms, "
              << "total: " << profile.total_ms << " ms\n";
}

}  // namespace cpu_kernel_v2
