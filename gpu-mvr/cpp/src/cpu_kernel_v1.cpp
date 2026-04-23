#include "cpu_kernel_v1.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>

#include "estimator.hpp"

namespace cpu_kernel_v1 {

namespace {

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

std::vector<size_t> search_impl(
    const cpu_mvr_index& index,
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
        query_objs.data(),
        q_doclen,
        static_cast<size_t>(runtime.nprobe),
        static_cast<size_t>(runtime.k_rank_cluster),
        rank_cluster_doc_ids);
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<size_t> rank_all_tokens_ids;
    std::vector<float> one_bit_dists;
    rank_all_tokens_1bit(
        index,
        query_objs.data(),
        q_doclen,
        rank_cluster_doc_ids,
        static_cast<size_t>(runtime.k_rank_all_tokens),
        rank_all_tokens_ids,
        one_bit_dists);
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<size_t> result;
    rank_all_tokens_exbits(
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

void rank_cluster_dists(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    size_t nprobe,
    size_t k,
    std::vector<size_t>& output_ids)
{
    output_ids.clear();
    std::vector<bool> doc_found(index.num_docs, false);
    std::vector<float> doc_dists(q_doclen * index.num_docs);

#pragma omp parallel for
    for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
        std::vector<size_t> ids;
        index.ivf->search(queries[j].rotated_query, nprobe, ids);
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            const size_t emb_id = ids[idx];
            const float dist = distance_one_bit(
                queries + j,
                &index.one_bit_code_[emb_id * index.padded_dim_ / 8],
                index.one_bit_factor_[emb_id],
                index.padded_dim_);
            const int doc_id = index.doc_ids_[emb_id];
            doc_found[doc_id] = true;
            doc_dists[j * index.num_docs + doc_id] =
                std::max(doc_dists[j * index.num_docs + doc_id], dist);
        }
    }

    std::priority_queue<std::pair<float, int>> max_heap;
    for (int doc_id = 0; doc_id < static_cast<int>(index.num_docs); ++doc_id) {
        if (!doc_found[doc_id]) {
            continue;
        }
        float score = 0.0F;
        for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
            score += doc_dists[j * index.num_docs + doc_id];
        }
        max_heap.emplace(score, doc_id);
    }

    for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
        output_ids.push_back(static_cast<size_t>(max_heap.top().second));
        max_heap.pop();
    }
}

void rank_all_tokens_1bit(
    const cpu_mvr_index& index,
    query_object* queries,
    size_t q_doclen,
    const std::vector<size_t>& input_ids,
    size_t k,
    std::vector<size_t>& output_ids,
    std::vector<float>& one_bit_dists)
{
    output_ids.clear();
    one_bit_dists.clear();

    std::unordered_map<size_t, size_t> doc_id_to_index;
    std::priority_queue<std::pair<float, size_t>> max_heap;
    size_t total_tokens = 0;
    std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);

    for (size_t i = 0; i < input_ids.size(); ++i) {
        doc_id_to_index[input_ids[i]] = i;
        total_tokens += index.doc_len(input_ids[i]);
        candidate_doc_ptrs[i + 1] = total_tokens;
    }

    std::vector<float> token_dists(total_tokens * q_doclen);

#pragma omp parallel for
    for (size_t idx = 0; idx < input_ids.size(); ++idx) {
        const size_t doc_id = input_ids[idx];
        float doc_score = 0.0F;
        for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
            float max_token_score = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < index.doc_len(doc_id); ++i) {
                const size_t tid = static_cast<size_t>(index.doc_ptrs_[doc_id]) + i;
                const float dist = distance_one_bit(
                    queries + j,
                    &index.one_bit_code_[tid * index.padded_dim_ / 8],
                    index.one_bit_factor_[tid],
                    index.padded_dim_);
                token_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j] = dist;
                max_token_score = std::max(max_token_score, dist);
            }
            doc_score += max_token_score;
        }
#pragma omp critical
        max_heap.emplace(doc_score, doc_id);
    }

    for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
        const size_t doc_id = max_heap.top().second;
        output_ids.push_back(doc_id);
        const size_t doc_idx = doc_id_to_index[doc_id];
        for (size_t t = 0; t < index.doc_len(doc_id); ++t) {
            for (int j = 0; j < static_cast<int>(q_doclen); ++j) {
                one_bit_dists.push_back(token_dists[(candidate_doc_ptrs[doc_idx] + t) * q_doclen + j]);
            }
        }
        max_heap.pop();
    }
}

void rank_all_tokens_exbits(
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

    std::priority_queue<std::pair<float, size_t>> max_heap;

#pragma omp parallel for
    for (size_t idx = 0; idx < input_ids.size(); ++idx) {
        const size_t doc_id = input_ids[idx];
        float doc_score = 0.0F;
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
#pragma omp critical
        max_heap.emplace(doc_score, doc_id);
    }
    const auto t_score_done = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
        output_ids.push_back(max_heap.top().second);
        max_heap.pop();
    }
    const auto t_select_done = std::chrono::high_resolution_clock::now();

    if (profile != nullptr) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        profile->stage3_prepare_ms = ms(t_stage3_start, t_prepare_done);
        profile->stage3_score_docs_ms = ms(t_prepare_done, t_score_done);
        profile->stage3_select_topk_ms = ms(t_score_done, t_select_done);
    }
}

std::vector<size_t> search(
    const cpu_mvr_index& index,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime)
{
    return search_impl(index, queries, q_doclen, k, runtime, nullptr);
}

std::vector<size_t> search_profiled(
    const cpu_mvr_index& index,
    const float* queries,
    size_t q_doclen,
    size_t k,
    const gpu_search_runtime_options& runtime,
    search_profile* profile,
    bool print_profile)
{
    search_profile local_profile;
    std::vector<size_t> result = search_impl(index, queries, q_doclen, k, runtime, &local_profile);
    if (profile != nullptr) {
        *profile = local_profile;
    }
    if (print_profile) {
        print_search_profile(local_profile);
    }
    return result;
}

void accumulate_search_profile(search_profile& accum, const search_profile& sample)
{
    accum.query_setup_ms += sample.query_setup_ms;
    accum.stage1_cluster_ms += sample.stage1_cluster_ms;
    accum.stage1_probe_ms += sample.stage1_probe_ms;
    accum.stage1_prepare_ms += sample.stage1_prepare_ms;
    accum.stage1_scan_ms += sample.stage1_scan_ms;
    accum.stage1_reduce_ms += sample.stage1_reduce_ms;
    accum.stage1_cleanup_ms += sample.stage1_cleanup_ms;
    accum.stage2_1bit_ms += sample.stage2_1bit_ms;
    accum.stage2_lut_ms += sample.stage2_lut_ms;
    accum.stage2_score_docs_ms += sample.stage2_score_docs_ms;
    accum.stage2_select_topk_ms += sample.stage2_select_topk_ms;
    accum.stage2_materialize_ms += sample.stage2_materialize_ms;
    accum.stage3_exbits_ms += sample.stage3_exbits_ms;
    accum.stage3_prepare_ms += sample.stage3_prepare_ms;
    accum.stage3_score_docs_ms += sample.stage3_score_docs_ms;
    accum.stage3_select_topk_ms += sample.stage3_select_topk_ms;
    accum.total_ms += sample.total_ms;
    accum.end_to_end_ms += sample.end_to_end_ms;
}

search_profile average_search_profile(const search_profile& total, size_t count)
{
    search_profile average = total;
    if (count == 0) {
        return average;
    }

    const double denom = static_cast<double>(count);
    average.query_setup_ms /= denom;
    average.stage1_cluster_ms /= denom;
    average.stage1_probe_ms /= denom;
    average.stage1_prepare_ms /= denom;
    average.stage1_scan_ms /= denom;
    average.stage1_reduce_ms /= denom;
    average.stage1_cleanup_ms /= denom;
    average.stage2_1bit_ms /= denom;
    average.stage2_lut_ms /= denom;
    average.stage2_score_docs_ms /= denom;
    average.stage2_select_topk_ms /= denom;
    average.stage2_materialize_ms /= denom;
    average.stage3_exbits_ms /= denom;
    average.stage3_prepare_ms /= denom;
    average.stage3_score_docs_ms /= denom;
    average.stage3_select_topk_ms /= denom;
    average.total_ms /= denom;
    average.end_to_end_ms /= denom;
    return average;
}

void print_search_profile(const search_profile& profile)
{
    std::cout << "[search] query_setup: " << profile.query_setup_ms << " ms, "
              << "stage1_cluster: " << profile.stage1_cluster_ms << " ms, "
              << "stage2_1bit: " << profile.stage2_1bit_ms << " ms, "
              << "stage3_exbits: " << profile.stage3_exbits_ms << " ms, "
              << "total: " << profile.total_ms << " ms, "
              << "end_to_end: " << profile.end_to_end_ms << " ms\n";
}

void print_search_profile_summary(const search_profile& total, size_t count)
{
    const auto avg = average_search_profile(total, count);
    const double end_to_end_ms = avg.end_to_end_ms;
    const double profiled_total_ms = avg.total_ms;
    const double stage12_ms = avg.stage1_cluster_ms + avg.stage2_1bit_ms;
    const double accounted_ms =
        avg.query_setup_ms + avg.stage1_cluster_ms + avg.stage2_1bit_ms + avg.stage3_exbits_ms;
    const double unaccounted_ms = std::max(0.0, end_to_end_ms - accounted_ms);

    auto pct = [](double part, double total_value) {
        return total_value > 0.0 ? (100.0 * part / total_value) : 0.0;
    };

    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << "[PROFILE_AVG] queries=" << count
        << " end_to_end_ms=" << end_to_end_ms
        << " profiled_total_ms=" << profiled_total_ms
        << " accounted_ms=" << accounted_ms
        << " unaccounted_ms=" << unaccounted_ms
        << std::endl;
    std::cout
        << "[PROFILE_AVG] stage=query_setup"
        << " total_ms=" << avg.query_setup_ms
        << " pct_end_to_end=" << pct(avg.query_setup_ms, end_to_end_ms)
        << std::endl;
    std::cout
        << "[PROFILE_AVG] stage=stage1"
        << " total_ms=" << avg.stage1_cluster_ms
        << " pct_end_to_end=" << pct(avg.stage1_cluster_ms, end_to_end_ms)
        << " pct_profiled_total=" << pct(avg.stage1_cluster_ms, profiled_total_ms)
        << " probe_ms=" << avg.stage1_probe_ms
        << " prepare_ms=" << avg.stage1_prepare_ms
        << " scan_ms=" << avg.stage1_scan_ms
        << " reduce_ms=" << avg.stage1_reduce_ms
        << " cleanup_ms=" << avg.stage1_cleanup_ms
        << std::endl;
    std::cout
        << "[PROFILE_AVG] stage=stage2"
        << " total_ms=" << avg.stage2_1bit_ms
        << " pct_end_to_end=" << pct(avg.stage2_1bit_ms, end_to_end_ms)
        << " pct_profiled_total=" << pct(avg.stage2_1bit_ms, profiled_total_ms)
        << " lut_ms=" << avg.stage2_lut_ms
        << " score_docs_ms=" << avg.stage2_score_docs_ms
        << " select_topk_ms=" << avg.stage2_select_topk_ms
        << " materialize_ms=" << avg.stage2_materialize_ms
        << std::endl;
    std::cout
        << "[PROFILE_AVG] stage=stage3"
        << " total_ms=" << avg.stage3_exbits_ms
        << " pct_end_to_end=" << pct(avg.stage3_exbits_ms, end_to_end_ms)
        << " pct_profiled_total=" << pct(avg.stage3_exbits_ms, profiled_total_ms)
        << " prepare_ms=" << avg.stage3_prepare_ms
        << " score_docs_ms=" << avg.stage3_score_docs_ms
        << " select_topk_ms=" << avg.stage3_select_topk_ms
        << std::endl;
    std::cout
        << "[PROFILE_AVG] stage=stage12"
        << " total_ms=" << stage12_ms
        << " pct_end_to_end=" << pct(stage12_ms, end_to_end_ms)
        << " pct_profiled_total=" << pct(stage12_ms, profiled_total_ms)
        << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

}  // namespace cpu_kernel_v1
