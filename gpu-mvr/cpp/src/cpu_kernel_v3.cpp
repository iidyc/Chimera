#include "cpu_kernel_v3.hpp"

#include <cerrno>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "clustered_stage1_file_format.hpp"
#include "gpu_index_layout.hpp"
#include "query.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/index/lut.hpp"
#include "rabitqlib/index/query.hpp"

namespace cpu_kernel_v3 {

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

size_t next_power_of_two(size_t value)
{
    size_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
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

class token_doc_max_table {
   public:
    void reset(size_t expected_entries)
    {
        const size_t reserve_entries = std::max<size_t>(expected_entries, 1024);
        const size_t capacity = next_power_of_two(reserve_entries * 2);
        keys_.assign(capacity, kEmptyKey);
        values_.resize(capacity);
        touched_doc_ids_.clear();
        touched_doc_ids_.reserve(reserve_entries);
        size_ = 0;
    }

    void max_update(int doc_id, float score)
    {
        if ((size_ + 1) * 10 >= keys_.size() * 7) {
            rehash(keys_.size() * 2);
        }

        const size_t slot = find_slot(doc_id);
        if (keys_[slot] == kEmptyKey) {
            keys_[slot] = doc_id;
            values_[slot] = score;
            touched_doc_ids_.push_back(doc_id);
            ++size_;
            return;
        }

        if (score > values_[slot]) {
            values_[slot] = score;
        }
    }

    [[nodiscard]] float score_for(int doc_id) const
    {
        return values_[find_slot(doc_id)];
    }

    [[nodiscard]] size_t size() const { return size_; }

    [[nodiscard]] const std::vector<int>& touched_doc_ids() const
    {
        return touched_doc_ids_;
    }

   private:
    static constexpr int kEmptyKey = -1;

    [[nodiscard]] size_t find_slot(int doc_id) const
    {
        const size_t mask = keys_.size() - 1;
        size_t slot = static_cast<size_t>(static_cast<uint32_t>(doc_id)) & mask;
        while (true) {
            const int key = keys_[slot];
            if (key == kEmptyKey || key == doc_id) {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }

    void insert_for_rehash(int doc_id, float score)
    {
        const size_t slot = find_slot(doc_id);
        keys_[slot] = doc_id;
        values_[slot] = score;
        touched_doc_ids_.push_back(doc_id);
        ++size_;
    }

    void rehash(size_t new_capacity)
    {
        new_capacity = next_power_of_two(std::max<size_t>(new_capacity, 1024));
        auto old_keys = std::move(keys_);
        auto old_values = std::move(values_);
        auto old_touched = std::move(touched_doc_ids_);

        keys_.assign(new_capacity, kEmptyKey);
        values_.resize(new_capacity);
        touched_doc_ids_.clear();
        touched_doc_ids_.reserve(std::max<size_t>(old_touched.size(), 1024));
        size_ = 0;

        const size_t old_mask = old_keys.size() - 1;
        for (int doc_id : old_touched) {
            size_t old_slot = static_cast<size_t>(static_cast<uint32_t>(doc_id)) & old_mask;
            while (old_keys[old_slot] != doc_id) {
                old_slot = (old_slot + 1) & old_mask;
            }
            insert_for_rehash(doc_id, old_values[old_slot]);
        }
    }

    std::vector<int> keys_;
    std::vector<float> values_;
    std::vector<int> touched_doc_ids_;
    size_t size_ = 0;
};

struct doc_score_pair {
    int doc_id = -1;
    float score = 0.0f;
};

struct doc_score_span {
    size_t begin = 0;
    size_t end = 0;
};

inline constexpr size_t kFixedQueryDoclen = 32;
inline constexpr size_t kMaxStage1ClusterRuns = 1024;

[[nodiscard]] bool empty(doc_score_span span)
{
    return span.begin >= span.end;
}

bool better_doc_score_pair(const doc_score_pair& a, const doc_score_pair& b)
{
    if (a.score != b.score) return a.score > b.score;
    return a.doc_id < b.doc_id;
}

void merge_doc_score_pairs_max(
    const std::vector<doc_score_pair>& lhs,
    const std::vector<doc_score_pair>& rhs,
    std::vector<doc_score_pair>& out)
{
    out.clear();
    out.reserve(lhs.size() + rhs.size());

    size_t i = 0;
    size_t j = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i].doc_id < rhs[j].doc_id) {
            out.push_back(lhs[i++]);
        } else if (rhs[j].doc_id < lhs[i].doc_id) {
            out.push_back(rhs[j++]);
        } else {
            out.push_back({lhs[i].doc_id, std::max(lhs[i].score, rhs[j].score)});
            ++i;
            ++j;
        }
    }

    while (i < lhs.size()) {
        out.push_back(lhs[i++]);
    }
    while (j < rhs.size()) {
        out.push_back(rhs[j++]);
    }
}

class fixed_query_run_merge32 {
   public:
    void reset()
    {
        active_ = 0;
        total_size_ = 0;
    }

    void add_run(const std::vector<doc_score_pair>& arena, doc_score_span span)
    {
        add_run(arena.data() + span.begin, arena.data() + span.end);
    }

    void add_run(const doc_score_pair* begin, const doc_score_pair* end)
    {
        total_size_ += static_cast<size_t>(end - begin);
        if (begin == end) {
            return;
        }
        if (active_ >= kFixedQueryDoclen) {
            throw std::runtime_error("fixed_query_run_merge32 exceeded fixed run capacity");
        }

        cur_[active_] = begin;
        end_[active_] = end;
        head_doc_ids_[active_] = cur_[active_]->doc_id;
        head_scores_[active_] = cur_[active_]->score;
        ++active_;
    }

    void merge(std::vector<doc_score_pair>& out)
    {
        out.resize(total_size_);
        size_t out_size = 0;

        while (active_ > 0) {
            int min_doc_id = head_doc_ids_[0];
            for (size_t i = 1; i < active_; ++i) {
                min_doc_id = std::min(min_doc_id, head_doc_ids_[i]);
            }

            float score_sum = 0.0f;
            size_t i = 0;
            while (i < active_) {
                if (head_doc_ids_[i] != min_doc_id) {
                    ++i;
                    continue;
                }

                score_sum += head_scores_[i];
                advance_slot(i);
            }

            out[out_size++] = {min_doc_id, score_sum};
        }

        out.resize(out_size);
    }

    void merge_topk(std::vector<doc_score_pair>& out, size_t k)
    {
        auto min_heap_cmp = [](const doc_score_pair& a, const doc_score_pair& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.doc_id < b.doc_id;
        };

        std::priority_queue<
            doc_score_pair,
            std::vector<doc_score_pair>,
            decltype(min_heap_cmp)>
            min_heap(min_heap_cmp);

        while (active_ > 0) {
            int min_doc_id = head_doc_ids_[0];
            for (size_t i = 1; i < active_; ++i) {
                min_doc_id = std::min(min_doc_id, head_doc_ids_[i]);
            }

            float score_sum = 0.0f;
            size_t i = 0;
            while (i < active_) {
                if (head_doc_ids_[i] != min_doc_id) {
                    ++i;
                    continue;
                }

                score_sum += head_scores_[i];
                advance_slot(i);
            }

            const doc_score_pair candidate {min_doc_id, score_sum};
            if (k == 0) {
                continue;
            }
            if (min_heap.size() < k) {
                min_heap.push(candidate);
            } else if (better_doc_score_pair(candidate, min_heap.top())) {
                min_heap.pop();
                min_heap.push(candidate);
            }
        }

        out.clear();
        out.reserve(min_heap.size());
        while (!min_heap.empty()) {
            out.push_back(min_heap.top());
            min_heap.pop();
        }
        std::sort(out.begin(), out.end(), better_doc_score_pair);
    }

   private:
    void advance_slot(size_t idx)
    {
        ++cur_[idx];
        if (cur_[idx] == end_[idx]) {
            const size_t last = active_ - 1;
            if (idx != last) {
                cur_[idx] = cur_[last];
                end_[idx] = end_[last];
                head_doc_ids_[idx] = head_doc_ids_[last];
                head_scores_[idx] = head_scores_[last];
            }
            --active_;
            return;
        }

        head_doc_ids_[idx] = cur_[idx]->doc_id;
        head_scores_[idx] = cur_[idx]->score;
    }

    std::array<const doc_score_pair*, kFixedQueryDoclen> cur_ {};
    std::array<const doc_score_pair*, kFixedQueryDoclen> end_ {};
    std::array<int, kFixedQueryDoclen> head_doc_ids_ {};
    std::array<float, kFixedQueryDoclen> head_scores_ {};
    size_t active_ = 0;
    size_t total_size_ = 0;
};

class fixed_score_run_merge32 {
   public:
    void reset()
    {
        active_ = 0;
    }

    void add_run(const std::vector<doc_score_pair>& run)
    {
        add_run(run.data(), run.data() + run.size());
    }

    void add_run(const doc_score_pair* begin, const doc_score_pair* end)
    {
        if (begin == end) {
            return;
        }
        if (active_ >= kFixedQueryDoclen) {
            throw std::runtime_error("fixed_score_run_merge32 exceeded fixed run capacity");
        }

        cur_[active_] = begin;
        end_[active_] = end;
        head_doc_ids_[active_] = begin->doc_id;
        head_scores_[active_] = begin->score;
        ++active_;
    }

    void merge_topk(std::vector<doc_score_pair>& out, size_t k)
    {
        out.clear();
        out.reserve(k);

        while (active_ > 0 && out.size() < k) {
            size_t best_idx = 0;
            doc_score_pair best {head_doc_ids_[0], head_scores_[0]};
            for (size_t i = 1; i < active_; ++i) {
                const doc_score_pair candidate {head_doc_ids_[i], head_scores_[i]};
                if (better_doc_score_pair(candidate, best)) {
                    best = candidate;
                    best_idx = i;
                }
            }

            out.push_back(best);
            advance_slot(best_idx);
        }
    }

   private:
    void advance_slot(size_t idx)
    {
        ++cur_[idx];
        if (cur_[idx] == end_[idx]) {
            const size_t last = active_ - 1;
            if (idx != last) {
                cur_[idx] = cur_[last];
                end_[idx] = end_[last];
                head_doc_ids_[idx] = head_doc_ids_[last];
                head_scores_[idx] = head_scores_[last];
            }
            --active_;
            return;
        }

        head_doc_ids_[idx] = cur_[idx]->doc_id;
        head_scores_[idx] = cur_[idx]->score;
    }

    std::array<const doc_score_pair*, kFixedQueryDoclen> cur_ {};
    std::array<const doc_score_pair*, kFixedQueryDoclen> end_ {};
    std::array<int, kFixedQueryDoclen> head_doc_ids_ {};
    std::array<float, kFixedQueryDoclen> head_scores_ {};
    size_t active_ = 0;
};

struct token_balanced_doc_buckets {
    std::vector<std::vector<size_t>> buckets;
    std::vector<size_t> thread_loads;
};

token_balanced_doc_buckets build_token_balanced_doc_buckets(
    const cpu_mvr_index& index,
    const std::vector<size_t>& input_ids)
{
    const int thread_count = std::min<int>(
        omp_get_max_threads(),
        std::max<size_t>(1, input_ids.size()));

    std::vector<size_t> order(input_ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&index, &input_ids](size_t a, size_t b) {
        return index.doc_len(input_ids[a]) > index.doc_len(input_ids[b]);
    });

    std::vector<std::vector<size_t>> buckets(static_cast<size_t>(thread_count));
    std::vector<size_t> thread_loads(static_cast<size_t>(thread_count), 0);
    for (size_t idx : order) {
        int target_thread = 0;
        for (int tid = 1; tid < thread_count; ++tid) {
            if (thread_loads[static_cast<size_t>(tid)] <
                thread_loads[static_cast<size_t>(target_thread)]) {
                target_thread = tid;
            }
        }
        buckets[static_cast<size_t>(target_thread)].push_back(idx);
        thread_loads[static_cast<size_t>(target_thread)] += index.doc_len(input_ids[idx]);
    }

    return {std::move(buckets), std::move(thread_loads)};
}

#if defined(__AVX512F__)
float score_doc4ex_avx512(
    const cpu_mvr_index& index,
    query_object* queries,
    const float* query_bias,
    const std::vector<float>& one_bit_dists,
    const std::vector<size_t>& candidate_doc_ptrs,
    size_t q_doclen,
    size_t input_idx,
    size_t doc_id)
{
    if (q_doclen != kFixedQueryDoclen) {
        throw std::runtime_error("score_doc4ex_avx512 requires q_doclen == 32");
    }

    const float scale = static_cast<float>(1 << index.ex_bits);
    const size_t doc_start = static_cast<size_t>(index.doc_ptrs_[doc_id]);
    alignas(64) float max_ts[kFixedQueryDoclen];
    alignas(64) float ip_ex_buf[kFixedQueryDoclen];
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t j = 0; j < kFixedQueryDoclen; j += 16) {
        _mm512_store_ps(&max_ts[j], neg_inf);
    }

    for (size_t t = 0; t < index.doc_len(doc_id); ++t) {
        const size_t tid = doc_start + t;
        const char* ex_code =
            &index.ex_code_[tid * index.padded_dim_ * index.ex_bits / 8];
        for (size_t j = 0; j < kFixedQueryDoclen; ++j) {
            ip_ex_buf[j] = ip_ex_bits(
                queries + j,
                ex_code,
                index.ip_func_,
                index.padded_dim_);
        }

        const __m512 v_tok_scale =
            _mm512_set1_ps(scale / index.one_bit_factor_[tid]);
        const __m512 v_tok_exf = _mm512_set1_ps(index.ex_factor_[tid]);
        const float* ob_base =
            &one_bit_dists[(candidate_doc_ptrs[input_idx] + t) * q_doclen];

        for (size_t j = 0; j < kFixedQueryDoclen; j += 16) {
            const __m512 ob = _mm512_loadu_ps(&ob_base[j]);
            const __m512 exd = _mm512_load_ps(&ip_ex_buf[j]);
            const __m512 qb = _mm512_load_ps(&query_bias[j]);
            const __m512 combined = _mm512_mul_ps(
                _mm512_add_ps(_mm512_fmadd_ps(v_tok_scale, ob, qb), exd),
                v_tok_exf);
            _mm512_store_ps(
                &max_ts[j],
                _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
        }
    }

    __m512 sum = _mm512_load_ps(&max_ts[0]);
    for (size_t j = 16; j < kFixedQueryDoclen; j += 16) {
        sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
    }
    return _mm512_reduce_add_ps(sum);
}
#endif

doc_score_span append_merged_doc_score_pairs_max(
    const std::vector<doc_score_pair>& arena,
    doc_score_span lhs,
    doc_score_span rhs,
    std::vector<doc_score_pair>& out)
{
    const size_t begin = out.size();
    size_t i = lhs.begin;
    size_t j = rhs.begin;
    while (i < lhs.end && j < rhs.end) {
        if (arena[i].doc_id < arena[j].doc_id) {
            out.push_back(arena[i++]);
        } else if (arena[j].doc_id < arena[i].doc_id) {
            out.push_back(arena[j++]);
        } else {
            out.push_back({arena[i].doc_id, std::max(arena[i].score, arena[j].score)});
            ++i;
            ++j;
        }
    }
    while (i < lhs.end) {
        out.push_back(arena[i++]);
    }
    while (j < rhs.end) {
        out.push_back(arena[j++]);
    }
    return {begin, out.size()};
}

doc_score_span append_merged_doc_score_pairs_sum(
    const std::vector<doc_score_pair>& arena,
    doc_score_span lhs,
    doc_score_span rhs,
    std::vector<doc_score_pair>& out)
{
    const size_t begin = out.size();
    size_t i = lhs.begin;
    size_t j = rhs.begin;
    while (i < lhs.end && j < rhs.end) {
        if (arena[i].doc_id < arena[j].doc_id) {
            out.push_back(arena[i++]);
        } else if (arena[j].doc_id < arena[i].doc_id) {
            out.push_back(arena[j++]);
        } else {
            out.push_back({arena[i].doc_id, arena[i].score + arena[j].score});
            ++i;
            ++j;
        }
    }
    while (i < lhs.end) {
        out.push_back(arena[i++]);
    }
    while (j < rhs.end) {
        out.push_back(arena[j++]);
    }
    return {begin, out.size()};
}

doc_score_span append_copy_doc_score_pairs(
    const std::vector<doc_score_pair>& arena,
    doc_score_span span,
    std::vector<doc_score_pair>& out)
{
    const size_t begin = out.size();
    out.insert(out.end(), arena.begin() + static_cast<std::ptrdiff_t>(span.begin),
               arena.begin() + static_cast<std::ptrdiff_t>(span.end));
    return {begin, out.size()};
}

doc_score_span merge_doc_score_pairs_max_into(
    const std::vector<doc_score_pair>& arena,
    doc_score_span lhs,
    doc_score_span rhs,
    std::vector<doc_score_pair>& out,
    size_t out_begin)
{
    size_t write = out_begin;
    size_t i = lhs.begin;
    size_t j = rhs.begin;
    while (i < lhs.end && j < rhs.end) {
        if (arena[i].doc_id < arena[j].doc_id) {
            out[write++] = arena[i++];
        } else if (arena[j].doc_id < arena[i].doc_id) {
            out[write++] = arena[j++];
        } else {
            out[write++] = {arena[i].doc_id, std::max(arena[i].score, arena[j].score)};
            ++i;
            ++j;
        }
    }
    while (i < lhs.end) {
        out[write++] = arena[i++];
    }
    while (j < rhs.end) {
        out[write++] = arena[j++];
    }
    return {out_begin, write};
}

doc_score_span merge_doc_score_pairs_sum_into(
    const std::vector<doc_score_pair>& arena,
    doc_score_span lhs,
    doc_score_span rhs,
    std::vector<doc_score_pair>& out,
    size_t out_begin)
{
    size_t write = out_begin;
    size_t i = lhs.begin;
    size_t j = rhs.begin;
    while (i < lhs.end && j < rhs.end) {
        if (arena[i].doc_id < arena[j].doc_id) {
            out[write++] = arena[i++];
        } else if (arena[j].doc_id < arena[i].doc_id) {
            out[write++] = arena[j++];
        } else {
            out[write++] = {arena[i].doc_id, arena[i].score + arena[j].score};
            ++i;
            ++j;
        }
    }
    while (i < lhs.end) {
        out[write++] = arena[i++];
    }
    while (j < rhs.end) {
        out[write++] = arena[j++];
    }
    return {out_begin, write};
}

doc_score_span copy_doc_score_pairs_into(
    const std::vector<doc_score_pair>& arena,
    doc_score_span span,
    std::vector<doc_score_pair>& out,
    size_t out_begin)
{
    const size_t len = span.end - span.begin;
    std::copy(
        arena.begin() + static_cast<std::ptrdiff_t>(span.begin),
        arena.begin() + static_cast<std::ptrdiff_t>(span.end),
        out.begin() + static_cast<std::ptrdiff_t>(out_begin));
    return {out_begin, out_begin + len};
}

void plan_pairwise_merge_round(
    const std::vector<doc_score_span>& src_spans,
    std::vector<doc_score_span>& dst_spans,
    std::vector<doc_score_pair>& dst_arena)
{
    dst_spans.assign((src_spans.size() + 1) / 2, {});
    size_t offset = 0;
    for (size_t pair_idx = 0; pair_idx < dst_spans.size(); ++pair_idx) {
        const size_t lhs_idx = pair_idx * 2;
        const size_t rhs_idx = lhs_idx + 1;
        const auto lhs = src_spans[lhs_idx];
        size_t cap = lhs.end - lhs.begin;
        if (rhs_idx < src_spans.size()) {
            const auto rhs = src_spans[rhs_idx];
            cap += rhs.end - rhs.begin;
        }
        dst_spans[pair_idx] = {offset, offset + cap};
        offset += cap;
    }
    dst_arena.resize(offset);
}

class token_doc_max_table_fast {
   public:
    void reset(size_t expected_entries)
    {
        const size_t reserve_entries = std::max<size_t>(expected_entries, 1024);
        arena_a_.clear();
        spans_a_.clear();
        active_cluster_begin_ = 0;
        arena_a_.reserve(reserve_entries);
        spans_a_.reserve(64);
    }

    void begin_cluster(size_t cluster_size)
    {
        active_cluster_begin_ = arena_a_.size();
        arena_a_.reserve(arena_a_.size() + std::min<size_t>(cluster_size, 4096));
    }

    void push_positive(int doc_id, float score)
    {
        if (score <= 0.0f) {
            return;
        }

        if (arena_a_.size() == active_cluster_begin_ || arena_a_.back().doc_id != doc_id) {
            arena_a_.push_back({doc_id, score});
        } else if (score > arena_a_.back().score) {
            arena_a_.back().score = score;
        }
    }

    void end_cluster()
    {
        if (arena_a_.size() == active_cluster_begin_) {
            return;
        }
        spans_a_.push_back({active_cluster_begin_, arena_a_.size()});
    }

    [[nodiscard]] size_t reserve_output_capacity() const
    {
        return arena_a_.size();
    }

    doc_score_span finalize_into(std::vector<doc_score_pair>& out) const
    {
        if (spans_a_.empty()) {
            return {out.size(), out.size()};
        }

        if (spans_a_.size() > kMaxStage1ClusterRuns) {
            throw std::runtime_error("stage1 cluster merge exceeded fixed run capacity");
        }

        struct run_cursor {
            size_t pos = 0;
            size_t end = 0;
        };

        auto heap_less = [this](const run_cursor& a, const run_cursor& b) {
            return arena_a_[a.pos].doc_id > arena_a_[b.pos].doc_id;
        };

        std::vector<run_cursor> heap;
        heap.reserve(spans_a_.size());
        for (const auto& span : spans_a_) {
            if (!empty(span)) {
                heap.push_back({span.begin, span.end});
                std::push_heap(heap.begin(), heap.end(), heap_less);
            }
        }

        const size_t begin = out.size();
        out.resize(begin + arena_a_.size());
        size_t out_size = 0;

        auto pop_min_cursor = [&]() {
            std::pop_heap(heap.begin(), heap.end(), heap_less);
            run_cursor cur = heap.back();
            heap.pop_back();
            return cur;
        };

        auto push_cursor = [&](run_cursor cur) {
            if (cur.pos < cur.end) {
                heap.push_back(cur);
                std::push_heap(heap.begin(), heap.end(), heap_less);
            }
        };

        while (!heap.empty()) {
            run_cursor cur = pop_min_cursor();
            const int doc_id = arena_a_[cur.pos].doc_id;
            float best_score = arena_a_[cur.pos].score;
            ++cur.pos;
            push_cursor(cur);

            while (!heap.empty() && arena_a_[heap.front().pos].doc_id == doc_id) {
                run_cursor same = pop_min_cursor();
                best_score = std::max(best_score, arena_a_[same.pos].score);
                ++same.pos;
                push_cursor(same);
            }

            out[begin + out_size++] = {doc_id, best_score};
        }

        out.resize(begin + out_size);
        return {begin, begin + out_size};
    }

   private:
    std::vector<doc_score_pair> arena_a_;
    std::vector<doc_score_span> spans_a_;
    size_t active_cluster_begin_ = 0;
};

class token_doc_max_table_parallel {
   public:
    void reset(size_t expected_entries)
    {
        const size_t reserve_entries = std::max<size_t>(expected_entries, 1024);
        arena_a_.clear();
        spans_a_.clear();
        active_cluster_begin_ = 0;
        arena_a_.reserve(reserve_entries);
        spans_a_.reserve(64);
    }

    void begin_cluster(size_t cluster_size)
    {
        active_cluster_begin_ = arena_a_.size();
        arena_a_.reserve(arena_a_.size() + std::min<size_t>(cluster_size, 4096));
    }

    void push_positive(int doc_id, float score)
    {
        if (score <= 0.0f) {
            return;
        }

        if (arena_a_.size() == active_cluster_begin_ || arena_a_.back().doc_id != doc_id) {
            arena_a_.push_back({doc_id, score});
        } else if (score > arena_a_.back().score) {
            arena_a_.back().score = score;
        }
    }

    void end_cluster()
    {
        if (arena_a_.size() == active_cluster_begin_) {
            return;
        }
        spans_a_.push_back({active_cluster_begin_, arena_a_.size()});
    }

    [[nodiscard]] size_t reserve_output_capacity() const
    {
        return arena_a_.size();
    }

    doc_score_span finalize_into(std::vector<doc_score_pair>& out) const
    {
        if (spans_a_.empty()) {
            return {out.size(), out.size()};
        }

        std::vector<doc_score_pair> arena_b;
        arena_b.reserve(arena_a_.size());
        std::vector<doc_score_pair> arena_c;
        arena_c.reserve(arena_a_.size());
        std::vector<doc_score_span> spans_b;
        spans_b.reserve((spans_a_.size() + 1) / 2);
        std::vector<doc_score_span> spans_c;
        spans_c.reserve((spans_a_.size() + 1) / 2);

        const std::vector<doc_score_pair>* src_arena = &arena_a_;
        std::vector<doc_score_pair>* dst_arena = &arena_b;

        const std::vector<doc_score_span>* src_spans = &spans_a_;
        std::vector<doc_score_span>* dst_spans = &spans_b;

        while (src_spans->size() > 1) {
            plan_pairwise_merge_round(*src_spans, *dst_spans, *dst_arena);

            for (size_t pair_idx = 0; pair_idx < dst_spans->size(); ++pair_idx) {
                const size_t lhs_idx = pair_idx * 2;
                const size_t rhs_idx = lhs_idx + 1;
                const auto lhs = (*src_spans)[lhs_idx];
                const size_t out_begin = (*dst_spans)[pair_idx].begin;
                if (rhs_idx < src_spans->size()) {
                    const auto rhs = (*src_spans)[rhs_idx];
                    (*dst_spans)[pair_idx] = merge_doc_score_pairs_max_into(
                        *src_arena,
                        lhs,
                        rhs,
                        *dst_arena,
                        out_begin);
                } else {
                    (*dst_spans)[pair_idx] = copy_doc_score_pairs_into(
                        *src_arena,
                        lhs,
                        *dst_arena,
                        out_begin);
                }
            }

            src_arena = dst_arena;
            src_spans = dst_spans;
            if (dst_arena == &arena_b) {
                dst_arena = &arena_c;
                dst_spans = &spans_c;
            } else {
                dst_arena = &arena_b;
                dst_spans = &spans_b;
            }
        }

        const auto final_span = src_spans->front();
        const size_t begin = out.size();
        out.insert(
            out.end(),
            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.begin),
            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.end));
        return {begin, out.size()};
    }

   private:
    std::vector<doc_score_pair> arena_a_;
    std::vector<doc_score_span> spans_a_;
    size_t active_cluster_begin_ = 0;
};

void scan_cluster_doc_run_hash(
    const clustered_stage1_cache& stage1_cache,
    size_t cluster_id,
    const query_object& query,
    const rabitqlib::Lut<float>& lut,
    std::array<int32_t, rabitqlib::fastscan::kBatchSize>& accum,
    token_doc_max_table& doc_max,
    size_t padded_dim)
{
    const uint8_t* packed_codes = stage1_cache.packed_cluster_codes(cluster_id);
    const float* packed_factors = stage1_cache.packed_cluster_factors(cluster_id);
    const int* packed_doc_ids = stage1_cache.packed_cluster_doc_ids(cluster_id);
    const size_t cluster_size = stage1_cache.packed_cluster_token_count(cluster_id);
    const size_t num_batches =
        (cluster_size + rabitqlib::fastscan::kBatchSize - 1) /
        rabitqlib::fastscan::kBatchSize;

    int current_doc_id = -1;
    float current_doc_max = 0.0f;

    // auto flush_current_doc = [&doc_max, &current_doc_id, &current_doc_max]() {
    //     if (current_doc_id >= 0) {
    //         doc_max.max_update(current_doc_id, current_doc_max);
    //     }
    // };

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const uint8_t* batch_ptr =
            packed_codes +
            batch_idx * cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());
        rabitqlib::fastscan::accumulate_hacc(
            batch_ptr,
            lut.lut(),
            accum.data(),
            padded_dim);

        const size_t valid =
            std::min<size_t>(rabitqlib::fastscan::kBatchSize, cluster_size - batch_idx * rabitqlib::fastscan::kBatchSize);
        const size_t lane_base = batch_idx * rabitqlib::fastscan::kBatchSize;

        for (size_t lane = 0; lane < valid; ++lane) {
            const int doc_id = packed_doc_ids[lane_base + lane];
            const float ip =
                lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
            const float dist =
                (ip - query.cb1_sumq) * packed_factors[lane_base + lane];
            if (dist <= 0.0f) {
                continue;
            } else if (doc_id == current_doc_id && dist > current_doc_max) {
                current_doc_max = dist;
                doc_max.max_update(current_doc_id, current_doc_max);
            } else if (doc_id != current_doc_id) {
                current_doc_id = doc_id;
                current_doc_max = dist;
                doc_max.max_update(current_doc_id, current_doc_max);
            }

            // if (doc_id != current_doc_id) {
            //     flush_current_doc();
            //     current_doc_id = doc_id;
            //     current_doc_max = dist;
            // } else if (dist > current_doc_max) {
            //     current_doc_max = dist;
            // }
        }
    }

    // flush_current_doc();
}

void scan_cluster_sorted_merge_fast(
    const clustered_stage1_cache& stage1_cache,
    size_t cluster_id,
    const query_object& query,
    const rabitqlib::Lut<float>& lut,
    std::array<int32_t, rabitqlib::fastscan::kBatchSize>& accum,
    token_doc_max_table_fast& doc_max,
    size_t padded_dim)
{
    const uint8_t* packed_codes = stage1_cache.packed_cluster_codes(cluster_id);
    const float* packed_factors = stage1_cache.packed_cluster_factors(cluster_id);
    const int* packed_doc_ids = stage1_cache.packed_cluster_doc_ids(cluster_id);
    const size_t cluster_size = stage1_cache.packed_cluster_token_count(cluster_id);
    const size_t num_batches =
        (cluster_size + rabitqlib::fastscan::kBatchSize - 1) /
        rabitqlib::fastscan::kBatchSize;

    doc_max.begin_cluster(cluster_size);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const uint8_t* batch_ptr =
            packed_codes +
            batch_idx * cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());
        rabitqlib::fastscan::accumulate_hacc(
            batch_ptr,
            lut.lut(),
            accum.data(),
            padded_dim);

        const size_t valid =
            std::min<size_t>(rabitqlib::fastscan::kBatchSize, cluster_size - batch_idx * rabitqlib::fastscan::kBatchSize);
        const size_t lane_base = batch_idx * rabitqlib::fastscan::kBatchSize;

        for (size_t lane = 0; lane < valid; ++lane) {
            const int doc_id = packed_doc_ids[lane_base + lane];
            const float ip =
                lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
            const float dist =
                (ip - query.cb1_sumq) * packed_factors[lane_base + lane];
            doc_max.push_positive(doc_id, dist);
        }
    }

    doc_max.end_cluster();
}

void scan_cluster_sorted_merge_parallel(
    const clustered_stage1_cache& stage1_cache,
    size_t cluster_id,
    const query_object& query,
    const rabitqlib::Lut<float>& lut,
    std::array<int32_t, rabitqlib::fastscan::kBatchSize>& accum,
    token_doc_max_table_parallel& doc_max,
    size_t padded_dim)
{
    const uint8_t* packed_codes = stage1_cache.packed_cluster_codes(cluster_id);
    const float* packed_factors = stage1_cache.packed_cluster_factors(cluster_id);
    const int* packed_doc_ids = stage1_cache.packed_cluster_doc_ids(cluster_id);
    const size_t cluster_size = stage1_cache.packed_cluster_token_count(cluster_id);
    const size_t num_batches =
        (cluster_size + rabitqlib::fastscan::kBatchSize - 1) /
        rabitqlib::fastscan::kBatchSize;

    doc_max.begin_cluster(cluster_size);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const uint8_t* batch_ptr =
            packed_codes +
            batch_idx * cluster_batch_code_bytes(stage1_cache.code_bytes_per_vector());
        rabitqlib::fastscan::accumulate_hacc(
            batch_ptr,
            lut.lut(),
            accum.data(),
            padded_dim);

        const size_t valid =
            std::min<size_t>(rabitqlib::fastscan::kBatchSize, cluster_size - batch_idx * rabitqlib::fastscan::kBatchSize);
        const size_t lane_base = batch_idx * rabitqlib::fastscan::kBatchSize;

        for (size_t lane = 0; lane < valid; ++lane) {
            const int doc_id = packed_doc_ids[lane_base + lane];
            const float ip =
                lut.delta() * static_cast<float>(accum[lane]) + lut.sum_vl();
            const float dist =
                (ip - query.cb1_sumq) * packed_factors[lane_base + lane];
            doc_max.push_positive(doc_id, dist);
        }
    }

    doc_max.end_cluster();
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
    std::vector<float>& one_bit_dists,
    search_profile* profile)
{
    const auto t_stage2_start = std::chrono::high_resolution_clock::now();
    output_ids.clear();
    one_bit_dists.clear();

    std::vector<std::unique_ptr<rabitqlib::Lut<float>>> luts;
    luts.reserve(q_doclen);
    for (size_t j = 0; j < q_doclen; ++j) {
        luts.emplace_back(std::make_unique<rabitqlib::Lut<float>>(queries[j].rotated_query, index.padded_dim_, true));
    }
    const auto t_lut_done = std::chrono::high_resolution_clock::now();

    std::vector<float> doc_scores(input_ids.size(), 0.0f);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t idx = 0; idx < input_ids.size(); ++idx) {
        doc_scores[idx] = stage2_score_doc_fastscan(
            index,
            stage2_cache,
            luts,
            queries,
            q_doclen,
            input_ids[idx]);
    }
    const auto t_score_done = std::chrono::high_resolution_clock::now();

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
    const auto t_select_done = std::chrono::high_resolution_clock::now();

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
    const auto t_materialize_done = std::chrono::high_resolution_clock::now();

    if (profile != nullptr) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        profile->stage2_lut_ms = ms(t_stage2_start, t_lut_done);
        profile->stage2_score_docs_ms = ms(t_lut_done, t_score_done);
        profile->stage2_select_topk_ms = ms(t_score_done, t_select_done);
        profile->stage2_materialize_ms = ms(t_select_done, t_materialize_done);
    }
}

void rank_all_tokens_doc4ex(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    const std::vector<size_t>& input_ids,
    const std::vector<float>& one_bit_dists,
    size_t k,
    std::vector<size_t>& output_ids,
    search_profile* profile)
{
    output_ids.clear();
    const auto t_stage3_start = std::chrono::high_resolution_clock::now();
    std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
    size_t total_tokens = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        total_tokens += index.doc_len(input_ids[i]);
        candidate_doc_ptrs[i + 1] = total_tokens;
    }
    const auto t_prepare_done = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<float, size_t>> refined_scores(
        input_ids.size(),
        {-std::numeric_limits<float>::infinity(), 0});

#if defined(__AVX512F__)
    alignas(64) float query_bias[kFixedQueryDoclen];
    if (q_doclen == kFixedQueryDoclen) {
        const float scale = static_cast<float>(1 << index.ex_bits);
        for (size_t j = 0; j < kFixedQueryDoclen; ++j) {
            query_bias[j] = scale * queries[j].cb1_sumq - queries[j].cbex_sumq;
        }
    }
#endif

    if constexpr (kStage3DynamicSchedule) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int idx_i = 0; idx_i < static_cast<int>(input_ids.size()); ++idx_i) {
            const size_t idx = static_cast<size_t>(idx_i);
            const size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
#if defined(__AVX512F__)
            if (q_doclen == kFixedQueryDoclen) {
                doc_score = score_doc4ex_avx512(
                    index,
                    queries,
                    query_bias,
                    one_bit_dists,
                    candidate_doc_ptrs,
                    q_doclen,
                    idx,
                    doc_id);
                refined_scores[idx] = {doc_score, doc_id};
                continue;
            }
#endif
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < index.doc_len(doc_id); ++i) {
                    const size_t tid = static_cast<size_t>(index.doc_ptrs_[doc_id]) + i;
                    const float one_bit_dist =
                        one_bit_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j];
                    const float ip_ex_dist = ip_ex_bits(
                        queries + j,
                        &index.ex_code_[tid * index.padded_dim_ * index.ex_bits / 8],
                        index.ip_func_,
                        index.padded_dim_);
                    const float dist = combine_dists(
                        queries + j,
                        one_bit_dist,
                        ip_ex_dist,
                        index.one_bit_factor_[tid],
                        index.ex_factor_[tid],
                        index.ex_bits);
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
            refined_scores[idx] = {doc_score, doc_id};
        }
    } else {
        const auto assignment = build_token_balanced_doc_buckets(index, input_ids);
        const int thread_count = static_cast<int>(assignment.buckets.size());

#pragma omp parallel num_threads(thread_count)
        {
            const int omp_tid = omp_get_thread_num();
            const auto& bucket = assignment.buckets[static_cast<size_t>(omp_tid)];
            for (size_t bucket_pos = 0; bucket_pos < bucket.size(); ++bucket_pos) {
                const size_t idx = bucket[bucket_pos];
                const size_t doc_id = input_ids[idx];
                float doc_score = 0.0F;
#if defined(__AVX512F__)
                if (q_doclen == kFixedQueryDoclen) {
                    doc_score = score_doc4ex_avx512(
                        index,
                        queries,
                        query_bias,
                        one_bit_dists,
                        candidate_doc_ptrs,
                        q_doclen,
                        idx,
                        doc_id);
                    refined_scores[idx] = {doc_score, doc_id};
                    continue;
                }
#endif
                for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                    float max_token_score = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < index.doc_len(doc_id); ++i) {
                        const size_t tid = static_cast<size_t>(index.doc_ptrs_[doc_id]) + i;
                        const float one_bit_dist =
                            one_bit_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j];
                        const float ip_ex_dist = ip_ex_bits(
                            queries + j,
                            &index.ex_code_[tid * index.padded_dim_ * index.ex_bits / 8],
                            index.ip_func_,
                            index.padded_dim_);
                        const float dist = combine_dists(
                            queries + j,
                            one_bit_dist,
                            ip_ex_dist,
                            index.one_bit_factor_[tid],
                            index.ex_factor_[tid],
                            index.ex_bits);
                        max_token_score = std::max(max_token_score, dist);
                    }
                    doc_score += max_token_score;
                }
                refined_scores[idx] = {doc_score, doc_id};
            }
        }
    }
    const auto t_score_done = std::chrono::high_resolution_clock::now();

    const size_t topk_count = std::min(k, refined_scores.size());
    auto refined_order = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    if (topk_count < refined_scores.size()) {
        std::partial_sort(
            refined_scores.begin(),
            refined_scores.begin() + static_cast<std::ptrdiff_t>(topk_count),
            refined_scores.end(),
            refined_order);
    } else {
        std::sort(refined_scores.begin(), refined_scores.end(), refined_order);
    }

    output_ids.reserve(topk_count);
    for (size_t i = 0; i < topk_count; ++i) {
        output_ids.push_back(refined_scores[i].second);
    }
    const auto t_select_done = std::chrono::high_resolution_clock::now();

    if (profile != nullptr) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        profile->stage3_prepare_ms = ms(t_stage3_start, t_prepare_done);
        profile->stage3_score_docs_ms = ms(t_prepare_done, t_score_done);
        profile->stage3_select_topk_ms = ms(t_score_done, t_select_done);
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
    const auto t_query_setup_start = std::chrono::high_resolution_clock::now();
    std::vector<float> rotated_queries;
    std::vector<query_object> query_objs;
    build_query_objects(index, queries, q_doclen, rotated_queries, query_objs);
    const auto t_query_setup_done = std::chrono::high_resolution_clock::now();

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
        one_bit_dists,
        profile);
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<size_t> result;
    rank_all_tokens_doc4ex(
        index,
        query_objs.data(),
        q_doclen,
        rank_all_tokens_ids,
        one_bit_dists,
        k,
        result,
        profile);
    const auto t3 = std::chrono::high_resolution_clock::now();

    if (profile != nullptr) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        profile->query_setup_ms = ms(t_query_setup_start, t_query_setup_done);
        profile->stage1_cluster_ms = ms(t0, t1);
        profile->stage2_1bit_ms = ms(t1, t2);
        profile->stage3_exbits_ms = ms(t2, t3);
        profile->total_ms = ms(t0, t3);
        profile->end_to_end_ms = ms(t_query_setup_start, t3);
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
    steady_clock::time_point t_probe_done;
    steady_clock::time_point t_prepare_done;
    steady_clock::time_point t_scan_done;
    steady_clock::time_point t_reduce_done;
    {
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
        t_probe_done = steady_clock::now();

        if constexpr (kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_fast) {
            std::vector<token_doc_max_table_fast> query_doc_max(q_doclen);
            std::vector<std::vector<doc_score_pair>> query_final_runs(q_doclen);
            for (size_t j = 0; j < q_doclen; ++j) {
                const size_t expected_entries = std::min<size_t>(
                    index.num_docs,
                    std::max<size_t>(clusters_per_query[j].size() * 64, 1024));
                query_doc_max[j].reset(expected_entries);
            }
            t_prepare_done = steady_clock::now();

#pragma omp parallel for
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                const rabitqlib::Lut<float> lut(
                    queries[j].rotated_query,
                    index.padded_dim_,
                    true);

                auto& doc_max = query_doc_max[static_cast<size_t>(j)];
                std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

                for (size_t cid : clusters_per_query[j]) {
                    scan_cluster_sorted_merge_fast(
                        stage1_cache,
                        cid,
                        queries[static_cast<size_t>(j)],
                        lut,
                        accum,
                        doc_max,
                        index.padded_dim_);
                }

                auto& final_run = query_final_runs[static_cast<size_t>(j)];
                final_run.clear();
                final_run.reserve(doc_max.reserve_output_capacity());
                doc_max.finalize_into(final_run);
            }
            t_scan_done = steady_clock::now();

            const auto t_reduce_merge_start = steady_clock::now();
            std::vector<doc_score_pair> final_scores;
            bool final_scores_already_sorted_topk = false;
            if (!query_final_runs.empty()) {
                size_t total_finalized_pairs = 0;
                for (const auto& run : query_final_runs) {
                    total_finalized_pairs += run.size();
                }

                if (q_doclen == kFixedQueryDoclen) {
                    int min_doc_id = std::numeric_limits<int>::max();
                    int max_doc_id = std::numeric_limits<int>::min();
                    for (const auto& run : query_final_runs) {
                        if (run.empty()) {
                            continue;
                        }
                        min_doc_id = std::min(min_doc_id, run.front().doc_id);
                        max_doc_id = std::max(max_doc_id, run.back().doc_id);
                    }

                    if (min_doc_id <= max_doc_id) {
                        const int max_threads = std::max(1, omp_get_max_threads());
                        const int64_t doc_id_range =
                            static_cast<int64_t>(max_doc_id) - static_cast<int64_t>(min_doc_id) + 1;
                        const int partition_count = static_cast<int>(std::max<int64_t>(
                            1,
                            std::min<int64_t>(
                                static_cast<int64_t>(kFixedQueryDoclen),
                                std::min<int64_t>(static_cast<int64_t>(max_threads), doc_id_range))));

                        std::vector<float> shared_doc_scores(
                            static_cast<size_t>(doc_id_range), 0.0f);
                        std::vector<uint8_t> shared_doc_seen(
                            static_cast<size_t>(doc_id_range), 0);
                        std::vector<std::vector<doc_score_pair>> partition_topk(
                            static_cast<size_t>(partition_count));

#pragma omp parallel for schedule(static)
                        for (int partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
                            const int partition_begin = static_cast<int>(
                                static_cast<int64_t>(min_doc_id) +
                                (doc_id_range * partition_idx) / partition_count);
                            const int partition_end = static_cast<int>(
                                static_cast<int64_t>(min_doc_id) +
                                (doc_id_range * (partition_idx + 1)) / partition_count);

                            std::vector<int> touched_doc_ids;
                            touched_doc_ids.reserve(
                                std::max<size_t>(1024, total_finalized_pairs / static_cast<size_t>(partition_count)));

                            for (const auto& run : query_final_runs) {
                                if (run.empty()) {
                                    continue;
                                }

                                const auto lower = std::lower_bound(
                                    run.begin(),
                                    run.end(),
                                    partition_begin,
                                    [](const doc_score_pair& entry, int target_doc_id) {
                                        return entry.doc_id < target_doc_id;
                                    });
                                if (lower == run.end()) {
                                    continue;
                                }

                                const auto upper = std::lower_bound(
                                    lower,
                                    run.end(),
                                    partition_end,
                                    [](const doc_score_pair& entry, int target_doc_id) {
                                        return entry.doc_id < target_doc_id;
                                    });
                                if (lower == upper) {
                                    continue;
                                }

                                for (auto it = lower; it != upper; ++it) {
                                    const size_t score_idx =
                                        static_cast<size_t>(static_cast<int64_t>(it->doc_id) - min_doc_id);
                                    if (!shared_doc_seen[score_idx]) {
                                        shared_doc_seen[score_idx] = 1;
                                        touched_doc_ids.push_back(it->doc_id);
                                    }
                                    shared_doc_scores[score_idx] += it->score;
                                }
                            }

                            const size_t local_topk_count = std::min(k, touched_doc_ids.size());
                            auto topk_order = [&shared_doc_scores, min_doc_id](int a, int b) {
                                const float score_a =
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(a) - min_doc_id)];
                                const float score_b =
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(b) - min_doc_id)];
                                if (score_a != score_b) return score_a > score_b;
                                return a < b;
                            };
                            if (local_topk_count < touched_doc_ids.size()) {
                                std::partial_sort(
                                    touched_doc_ids.begin(),
                                    touched_doc_ids.begin() + static_cast<std::ptrdiff_t>(local_topk_count),
                                    touched_doc_ids.end(),
                                    topk_order);
                            } else {
                                std::sort(touched_doc_ids.begin(), touched_doc_ids.end(), topk_order);
                            }

                            auto& local_topk = partition_topk[static_cast<size_t>(partition_idx)];
                            local_topk.reserve(local_topk_count);
                            for (size_t local_idx = 0; local_idx < local_topk_count; ++local_idx) {
                                const int doc_id = touched_doc_ids[local_idx];
                                local_topk.push_back({
                                    doc_id,
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(doc_id) - min_doc_id)]
                                });
                            }
                        }

                        fixed_score_run_merge32 score_merge32;
                        score_merge32.reset();
                        for (const auto& local_topk : partition_topk) {
                            score_merge32.add_run(local_topk);
                        }
                        score_merge32.merge_topk(final_scores, k);
                        final_scores_already_sorted_topk = true;
                    }
                } else {
                    std::vector<doc_score_pair> work_arena_a;
                    work_arena_a.reserve(total_finalized_pairs);
                    std::vector<doc_score_span> spans_a;
                    spans_a.reserve(q_doclen);
                    for (const auto& run : query_final_runs) {
                        const size_t begin = work_arena_a.size();
                        work_arena_a.insert(work_arena_a.end(), run.begin(), run.end());
                        spans_a.push_back({begin, work_arena_a.size()});
                    }

                    std::vector<doc_score_pair> work_arena_b;
                    work_arena_b.reserve(total_finalized_pairs);
                    std::vector<doc_score_span> spans_b;
                    spans_b.reserve((q_doclen + 1) / 2);

                    auto* src_arena = &work_arena_a;
                    auto* dst_arena = &work_arena_b;
                    auto* src_spans = &spans_a;
                    auto* dst_spans = &spans_b;

                    while (src_spans->size() > 1) {
                        dst_arena->clear();
                        dst_spans->clear();
                        dst_spans->reserve((src_spans->size() + 1) / 2);

                        const size_t pair_count = src_spans->size() / 2;
                        for (size_t pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
                            dst_spans->push_back(append_merged_doc_score_pairs_sum(
                                *src_arena,
                                (*src_spans)[pair_idx * 2],
                                (*src_spans)[pair_idx * 2 + 1],
                                *dst_arena));
                        }

                        if (src_spans->size() % 2 != 0) {
                            dst_spans->push_back(
                                append_copy_doc_score_pairs(*src_arena, src_spans->back(), *dst_arena));
                        }

                        std::swap(src_arena, dst_arena);
                        std::swap(src_spans, dst_spans);
                    }

                    if (!src_spans->empty()) {
                        const doc_score_span final_span = src_spans->front();
                        final_scores.assign(
                            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.begin),
                            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.end));
                    }
                }
            }
            const auto t_reduce_merge_done = steady_clock::now();

            const size_t topk_count = std::min(k, final_scores.size());
            if (!final_scores_already_sorted_topk) {
                auto topk_order = [](const doc_score_pair& a, const doc_score_pair& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return a.doc_id < b.doc_id;
                };
                if (topk_count < final_scores.size()) {
                    std::partial_sort(
                        final_scores.begin(),
                        final_scores.begin() + topk_count,
                        final_scores.end(),
                        topk_order);
                    final_scores.resize(topk_count);
                } else {
                    std::sort(final_scores.begin(), final_scores.end(), topk_order);
                }
            }

            output_ids.reserve(topk_count);
            for (size_t i = 0; i < topk_count; ++i) {
                const auto& entry = final_scores[i];
                output_ids.push_back(static_cast<size_t>(entry.doc_id));
            }
            const auto t_reduce_sort_done = steady_clock::now();
            if (profile != nullptr) {
                profile->stage1_reduce_merge_ms = ms_between(t_reduce_merge_start, t_reduce_merge_done);
                profile->stage1_reduce_within_query_merge_ms = 0.0;
                profile->stage1_reduce_cross_query_merge_ms =
                    profile->stage1_reduce_merge_ms;
                profile->stage1_reduce_sort_ms = ms_between(t_reduce_merge_done, t_reduce_sort_done);
            }
        } else if constexpr (kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_hybrid) {
            std::vector<token_doc_max_table_parallel> query_doc_max(q_doclen);
            std::vector<std::vector<doc_score_pair>> query_final_runs(q_doclen);
            for (size_t j = 0; j < q_doclen; ++j) {
                const size_t expected_entries = std::min<size_t>(
                    index.num_docs,
                    std::max<size_t>(clusters_per_query[j].size() * 64, 1024));
                query_doc_max[j].reset(expected_entries);
            }
            t_prepare_done = steady_clock::now();

#pragma omp parallel for
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                const rabitqlib::Lut<float> lut(
                    queries[j].rotated_query,
                    index.padded_dim_,
                    true);

                auto& doc_max = query_doc_max[static_cast<size_t>(j)];
                std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

                for (size_t cid : clusters_per_query[j]) {
                    scan_cluster_sorted_merge_parallel(
                        stage1_cache,
                        cid,
                        queries[static_cast<size_t>(j)],
                        lut,
                        accum,
                        doc_max,
                        index.padded_dim_);
                }

                auto& final_run = query_final_runs[static_cast<size_t>(j)];
                final_run.clear();
                final_run.reserve(doc_max.reserve_output_capacity());
                doc_max.finalize_into(final_run);
            }
            t_scan_done = steady_clock::now();

            const auto t_reduce_merge_start = steady_clock::now();
            std::vector<doc_score_pair> final_scores;
            bool final_scores_already_sorted_topk = false;
            if (!query_final_runs.empty()) {
                size_t total_finalized_pairs = 0;
                for (const auto& run : query_final_runs) {
                    total_finalized_pairs += run.size();
                }

                if (q_doclen == kFixedQueryDoclen) {
                    int min_doc_id = std::numeric_limits<int>::max();
                    int max_doc_id = std::numeric_limits<int>::min();
                    for (const auto& run : query_final_runs) {
                        if (run.empty()) {
                            continue;
                        }
                        min_doc_id = std::min(min_doc_id, run.front().doc_id);
                        max_doc_id = std::max(max_doc_id, run.back().doc_id);
                    }

                    if (min_doc_id <= max_doc_id) {
                        const int max_threads = std::max(1, omp_get_max_threads());
                        const int64_t doc_id_range =
                            static_cast<int64_t>(max_doc_id) - static_cast<int64_t>(min_doc_id) + 1;
                        const int partition_count = static_cast<int>(std::max<int64_t>(
                            1,
                            std::min<int64_t>(
                                static_cast<int64_t>(kFixedQueryDoclen),
                                std::min<int64_t>(static_cast<int64_t>(max_threads), doc_id_range))));

                        std::vector<float> shared_doc_scores(
                            static_cast<size_t>(doc_id_range), 0.0f);
                        std::vector<uint8_t> shared_doc_seen(
                            static_cast<size_t>(doc_id_range), 0);
                        std::vector<std::vector<doc_score_pair>> partition_topk(
                            static_cast<size_t>(partition_count));

#pragma omp parallel for schedule(static)
                        for (int partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
                            const int partition_begin = static_cast<int>(
                                static_cast<int64_t>(min_doc_id) +
                                (doc_id_range * partition_idx) / partition_count);
                            const int partition_end = static_cast<int>(
                                static_cast<int64_t>(min_doc_id) +
                                (doc_id_range * (partition_idx + 1)) / partition_count);

                            std::vector<int> touched_doc_ids;
                            touched_doc_ids.reserve(
                                std::max<size_t>(1024, total_finalized_pairs / static_cast<size_t>(partition_count)));

                            for (const auto& run : query_final_runs) {
                                if (run.empty()) {
                                    continue;
                                }

                                const auto lower = std::lower_bound(
                                    run.begin(),
                                    run.end(),
                                    partition_begin,
                                    [](const doc_score_pair& entry, int target_doc_id) {
                                        return entry.doc_id < target_doc_id;
                                    });
                                if (lower == run.end()) {
                                    continue;
                                }

                                const auto upper = std::lower_bound(
                                    lower,
                                    run.end(),
                                    partition_end,
                                    [](const doc_score_pair& entry, int target_doc_id) {
                                        return entry.doc_id < target_doc_id;
                                    });
                                if (lower == upper) {
                                    continue;
                                }

                                for (auto it = lower; it != upper; ++it) {
                                    const size_t score_idx =
                                        static_cast<size_t>(static_cast<int64_t>(it->doc_id) - min_doc_id);
                                    if (!shared_doc_seen[score_idx]) {
                                        shared_doc_seen[score_idx] = 1;
                                        touched_doc_ids.push_back(it->doc_id);
                                    }
                                    shared_doc_scores[score_idx] += it->score;
                                }
                            }

                            const size_t local_topk_count = std::min(k, touched_doc_ids.size());
                            auto topk_order = [&shared_doc_scores, min_doc_id](int a, int b) {
                                const float score_a =
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(a) - min_doc_id)];
                                const float score_b =
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(b) - min_doc_id)];
                                if (score_a != score_b) return score_a > score_b;
                                return a < b;
                            };
                            if (local_topk_count < touched_doc_ids.size()) {
                                std::partial_sort(
                                    touched_doc_ids.begin(),
                                    touched_doc_ids.begin() + static_cast<std::ptrdiff_t>(local_topk_count),
                                    touched_doc_ids.end(),
                                    topk_order);
                            } else {
                                std::sort(touched_doc_ids.begin(), touched_doc_ids.end(), topk_order);
                            }

                            auto& local_topk = partition_topk[static_cast<size_t>(partition_idx)];
                            local_topk.reserve(local_topk_count);
                            for (size_t local_idx = 0; local_idx < local_topk_count; ++local_idx) {
                                const int doc_id = touched_doc_ids[local_idx];
                                local_topk.push_back({
                                    doc_id,
                                    shared_doc_scores[static_cast<size_t>(static_cast<int64_t>(doc_id) - min_doc_id)]
                                });
                            }
                        }

                        fixed_score_run_merge32 score_merge32;
                        score_merge32.reset();
                        for (const auto& local_topk : partition_topk) {
                            score_merge32.add_run(local_topk);
                        }
                        score_merge32.merge_topk(final_scores, k);
                        final_scores_already_sorted_topk = true;
                    }
                } else {
                    std::vector<doc_score_pair> work_arena_a;
                    work_arena_a.reserve(total_finalized_pairs);
                    std::vector<doc_score_span> spans_a;
                    spans_a.reserve(q_doclen);
                    for (const auto& run : query_final_runs) {
                        const size_t begin = work_arena_a.size();
                        work_arena_a.insert(work_arena_a.end(), run.begin(), run.end());
                        spans_a.push_back({begin, work_arena_a.size()});
                    }

                    std::vector<doc_score_pair> work_arena_b;
                    work_arena_b.reserve(total_finalized_pairs);
                    std::vector<doc_score_span> spans_b;
                    spans_b.reserve((q_doclen + 1) / 2);

                    auto* src_arena = &work_arena_a;
                    auto* dst_arena = &work_arena_b;
                    auto* src_spans = &spans_a;
                    auto* dst_spans = &spans_b;

                    while (src_spans->size() > 1) {
                        dst_arena->clear();
                        dst_spans->clear();
                        dst_spans->reserve((src_spans->size() + 1) / 2);

                        const size_t pair_count = src_spans->size() / 2;
                        for (size_t pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
                            dst_spans->push_back(append_merged_doc_score_pairs_sum(
                                *src_arena,
                                (*src_spans)[pair_idx * 2],
                                (*src_spans)[pair_idx * 2 + 1],
                                *dst_arena));
                        }

                        if (src_spans->size() % 2 != 0) {
                            dst_spans->push_back(
                                append_copy_doc_score_pairs(*src_arena, src_spans->back(), *dst_arena));
                        }

                        std::swap(src_arena, dst_arena);
                        std::swap(src_spans, dst_spans);
                    }

                    if (!src_spans->empty()) {
                        const doc_score_span final_span = src_spans->front();
                        final_scores.assign(
                            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.begin),
                            src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.end));
                    }
                }
            }
            const auto t_reduce_merge_done = steady_clock::now();

            const size_t topk_count = std::min(k, final_scores.size());
            if (!final_scores_already_sorted_topk) {
                auto topk_order = [](const doc_score_pair& a, const doc_score_pair& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return a.doc_id < b.doc_id;
                };
                if (topk_count < final_scores.size()) {
                    std::partial_sort(
                        final_scores.begin(),
                        final_scores.begin() + topk_count,
                        final_scores.end(),
                        topk_order);
                    final_scores.resize(topk_count);
                } else {
                    std::sort(final_scores.begin(), final_scores.end(), topk_order);
                }
            }

            output_ids.reserve(topk_count);
            for (size_t i = 0; i < topk_count; ++i) {
                const auto& entry = final_scores[i];
                output_ids.push_back(static_cast<size_t>(entry.doc_id));
            }
            const auto t_reduce_sort_done = steady_clock::now();
            if (profile != nullptr) {
                profile->stage1_reduce_merge_ms = ms_between(t_reduce_merge_start, t_reduce_merge_done);
                profile->stage1_reduce_within_query_merge_ms = 0.0;
                profile->stage1_reduce_cross_query_merge_ms =
                    profile->stage1_reduce_merge_ms;
                profile->stage1_reduce_sort_ms = ms_between(t_reduce_merge_done, t_reduce_sort_done);
            }
        } else if constexpr (kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_parallel) {
            std::vector<token_doc_max_table_parallel> query_doc_max(q_doclen);
            std::vector<std::vector<doc_score_pair>> query_final_runs(q_doclen);
            for (size_t j = 0; j < q_doclen; ++j) {
                const size_t expected_entries = std::min<size_t>(
                    index.num_docs,
                    std::max<size_t>(clusters_per_query[j].size() * 64, 1024));
                query_doc_max[j].reset(expected_entries);
            }
            t_prepare_done = steady_clock::now();

#pragma omp parallel for
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                const rabitqlib::Lut<float> lut(
                    queries[j].rotated_query,
                    index.padded_dim_,
                    true);

                auto& doc_max = query_doc_max[static_cast<size_t>(j)];
                std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

                for (size_t cid : clusters_per_query[j]) {
                    scan_cluster_sorted_merge_parallel(
                        stage1_cache,
                        cid,
                        queries[static_cast<size_t>(j)],
                        lut,
                        accum,
                        doc_max,
                        index.padded_dim_);
                }

                auto& final_run = query_final_runs[static_cast<size_t>(j)];
                final_run.clear();
                final_run.reserve(doc_max.reserve_output_capacity());
                doc_max.finalize_into(final_run);
            }
            t_scan_done = steady_clock::now();

            const auto t_reduce_merge_start = steady_clock::now();
            std::vector<doc_score_pair> final_scores;
            if (!query_final_runs.empty()) {
                size_t total_finalized_pairs = 0;
                for (const auto& run : query_final_runs) {
                    total_finalized_pairs += run.size();
                }

                std::vector<doc_score_pair> work_arena_a;
                work_arena_a.reserve(total_finalized_pairs);
                std::vector<doc_score_span> spans_a;
                spans_a.reserve(q_doclen);
                for (const auto& run : query_final_runs) {
                    const size_t begin = work_arena_a.size();
                    work_arena_a.insert(work_arena_a.end(), run.begin(), run.end());
                    spans_a.push_back({begin, work_arena_a.size()});
                }

                std::vector<doc_score_pair> work_arena_b;
                work_arena_b.reserve(total_finalized_pairs);
                std::vector<doc_score_span> spans_b;
                spans_b.reserve((q_doclen + 1) / 2);

                auto* src_arena = &work_arena_a;
                auto* dst_arena = &work_arena_b;
                auto* src_spans = &spans_a;
                auto* dst_spans = &spans_b;

                while (src_spans->size() > 1) {
                    plan_pairwise_merge_round(*src_spans, *dst_spans, *dst_arena);

                    const size_t pair_count = dst_spans->size();
#pragma omp parallel for schedule(static)
                    for (int pair_idx = 0; pair_idx < static_cast<int>(pair_count); ++pair_idx) {
                        const size_t lhs_idx = static_cast<size_t>(pair_idx) * 2;
                        const size_t rhs_idx = lhs_idx + 1;
                        const auto lhs = (*src_spans)[lhs_idx];
                        const size_t out_begin = (*dst_spans)[static_cast<size_t>(pair_idx)].begin;
                        if (rhs_idx < src_spans->size()) {
                            const auto rhs = (*src_spans)[rhs_idx];
                            (*dst_spans)[static_cast<size_t>(pair_idx)] =
                                merge_doc_score_pairs_sum_into(
                                    *src_arena,
                                    lhs,
                                    rhs,
                                    *dst_arena,
                                    out_begin);
                        } else {
                            (*dst_spans)[static_cast<size_t>(pair_idx)] =
                                copy_doc_score_pairs_into(
                                    *src_arena,
                                    lhs,
                                    *dst_arena,
                                    out_begin);
                        }
                    }

                    std::swap(src_arena, dst_arena);
                    std::swap(src_spans, dst_spans);
                }

                if (!src_spans->empty()) {
                    const doc_score_span final_span = src_spans->front();
                    final_scores.assign(
                        src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.begin),
                        src_arena->begin() + static_cast<std::ptrdiff_t>(final_span.end));
                }
            }
            const auto t_reduce_merge_done = steady_clock::now();

            const size_t topk_count = std::min(k, final_scores.size());
            auto topk_order = [](const doc_score_pair& a, const doc_score_pair& b) {
                if (a.score != b.score) return a.score > b.score;
                return a.doc_id < b.doc_id;
            };
            if (topk_count < final_scores.size()) {
                std::partial_sort(
                    final_scores.begin(),
                    final_scores.begin() + topk_count,
                    final_scores.end(),
                    topk_order);
                final_scores.resize(topk_count);
            } else {
                std::sort(final_scores.begin(), final_scores.end(), topk_order);
            }

            output_ids.reserve(topk_count);
            for (size_t i = 0; i < topk_count; ++i) {
                output_ids.push_back(static_cast<size_t>(final_scores[i].doc_id));
            }
            const auto t_reduce_sort_done = steady_clock::now();
            if (profile != nullptr) {
                profile->stage1_reduce_merge_ms = ms_between(t_reduce_merge_start, t_reduce_merge_done);
                profile->stage1_reduce_within_query_merge_ms = 0.0;
                profile->stage1_reduce_cross_query_merge_ms =
                    profile->stage1_reduce_merge_ms;
                profile->stage1_reduce_sort_ms = ms_between(t_reduce_merge_done, t_reduce_sort_done);
            }
        } else if constexpr (kStage1DocAccumMode == stage1_doc_accum_mode::doc_run_hash) {
            std::vector<float> doc_score_sum(index.num_docs, 0.0f);
            std::vector<int> touched_doc_ids;

            std::vector<token_doc_max_table> query_doc_max(q_doclen);
            for (size_t j = 0; j < q_doclen; ++j) {
                const size_t expected_entries = std::min<size_t>(
                    index.num_docs,
                    std::max<size_t>(clusters_per_query[j].size() * 64, 1024));
                query_doc_max[j].reset(expected_entries);
            }
            t_prepare_done = steady_clock::now();

#pragma omp parallel for
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                const rabitqlib::Lut<float> lut(
                    queries[j].rotated_query,
                    index.padded_dim_,
                    true);

                auto& doc_max = query_doc_max[static_cast<size_t>(j)];
                std::array<int32_t, rabitqlib::fastscan::kBatchSize> accum {};

                for (size_t cid : clusters_per_query[j]) {
                    scan_cluster_doc_run_hash(
                        stage1_cache,
                        cid,
                        queries[static_cast<size_t>(j)],
                        lut,
                        accum,
                        doc_max,
                        index.padded_dim_);
                }
            }
            t_scan_done = steady_clock::now();

            size_t total_touched_docs = 0;
            for (const auto& doc_max : query_doc_max) {
                total_touched_docs += doc_max.size();
            }

            const auto t_reduce_merge_start = steady_clock::now();
            std::vector<uint8_t> doc_seen(index.num_docs, 0);
            touched_doc_ids.reserve(std::min(total_touched_docs, index.num_docs));
            for (const auto& doc_max : query_doc_max) {
                for (int doc_id : doc_max.touched_doc_ids()) {
                    if (!doc_seen[doc_id]) {
                        doc_seen[doc_id] = 1;
                        touched_doc_ids.push_back(doc_id);
                    }
                    doc_score_sum[doc_id] += doc_max.score_for(doc_id);
                }
            }
            const auto t_reduce_merge_done = steady_clock::now();

            const size_t topk_count = std::min(k, touched_doc_ids.size());
            auto topk_order = [&doc_score_sum](int a, int b) {
                if (doc_score_sum[a] != doc_score_sum[b]) return doc_score_sum[a] > doc_score_sum[b];
                return a < b;
            };
            if (topk_count < touched_doc_ids.size()) {
                std::partial_sort(
                    touched_doc_ids.begin(),
                    touched_doc_ids.begin() + topk_count,
                    touched_doc_ids.end(),
                    topk_order);
                touched_doc_ids.resize(topk_count);
            } else {
                std::sort(touched_doc_ids.begin(), touched_doc_ids.end(), topk_order);
            }

            output_ids.reserve(topk_count);
            for (int doc_id : touched_doc_ids) {
                output_ids.push_back(static_cast<size_t>(doc_id));
            }
            const auto t_reduce_sort_done = steady_clock::now();
            if (profile != nullptr) {
                profile->stage1_reduce_merge_ms = ms_between(t_reduce_merge_start, t_reduce_merge_done);
                profile->stage1_reduce_sort_ms = ms_between(t_reduce_merge_done, t_reduce_sort_done);
            }
        } else {
            static_assert(
                kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_fast ||
                    kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_hybrid ||
                    kStage1DocAccumMode == stage1_doc_accum_mode::sorted_doc_merge_parallel ||
                    kStage1DocAccumMode == stage1_doc_accum_mode::doc_run_hash,
                "unsupported Stage 1 doc accumulation mode");
        }
        t_reduce_done = steady_clock::now();
    }
    const auto t_cleanup_done = steady_clock::now();

    if (profile != nullptr) {
        profile->stage1_probe_ms = ms_between(t_stage1_start, t_probe_done);
        profile->stage1_prepare_ms = ms_between(t_probe_done, t_prepare_done);
        profile->stage1_scan_ms = ms_between(t_prepare_done, t_scan_done);
        profile->stage1_reduce_ms = ms_between(t_scan_done, t_reduce_done);
        profile->stage1_cleanup_ms = ms_between(t_reduce_done, t_cleanup_done);
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
    search_profile* profile,
    bool print_profile)
{
    search_profile local_profile;
    std::vector<size_t> result =
        search_impl(index, stage1_cache, stage2_cache, queries, q_doclen, k, runtime, &local_profile);
    if (profile != nullptr) {
        *profile = local_profile;
    }
    if (print_profile) {
        cpu_kernel_v3::print_search_profile(local_profile);
    }
    return result;
}

void print_search_profile(const search_profile& profile)
{
    std::cout << "[search] stage1_cluster: " << profile.stage1_cluster_ms << " ms, "
              << "stage1_probe: " << profile.stage1_probe_ms << " ms, "
              << "stage1_prepare: " << profile.stage1_prepare_ms << " ms, "
              << "stage1_scan: " << profile.stage1_scan_ms << " ms, "
              << "stage1_reduce: " << profile.stage1_reduce_ms << " ms, "
              << "stage1_cleanup: " << profile.stage1_cleanup_ms << " ms, "
              << "stage2_1bit: " << profile.stage2_1bit_ms << " ms, "
              << "stage3_exbits: " << profile.stage3_exbits_ms << " ms, "
              << "total: " << profile.total_ms << " ms\n";
}

}  // namespace cpu_kernel_v3
