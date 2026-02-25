#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <omp.h>

#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "ivf_dockmeans.hpp"

using namespace rabitqlib;

enum class IVFType { PG, DocKMeans };

struct cpu_mvr_index {
    size_t n;           // number of vectors
    size_t d;           // dimension
    size_t n_clusters;  // number of clusters
    size_t ex_bits;     // n_bits = 1 + ex_bits
    size_t padded_dim_; // multiple of 64 dimension after padding
    Rotator<float>* rotator_;

    IVFType ivf_type_ = IVFType::DocKMeans;
    IVF_PG* ivf = nullptr;
    IVF_DocKMeans* ivf_dk_ = nullptr;

    std::vector<char> one_bit_code_;
    std::vector<char> ex_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;

    size_t num_docs;
    std::vector<int> doc_ids_;  // document ids for each vector, size n
    std::vector<int> doc_ptrs_; // pointers to the start of each document, size num_docs + 1

    float (*ip_func_)(const float*, const uint8_t*, size_t);

    cpu_mvr_index(const std::string& filename) {
        load(filename);
        ip_func_ = select_excode_ipfunc(ex_bits);
    }

    cpu_mvr_index(
        size_t n,
        size_t d,
        size_t n_clusters,
        size_t ex_bits,
        IVFType ivf_type = IVFType::PG
    ): n(n), d(d), n_clusters(n_clusters), ex_bits(ex_bits), ivf_type_(ivf_type) {
        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, round_up_to_multiple(d, 64));
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();
        padded_dim_ = rotator_->size();

        one_bit_code_.resize(n * padded_dim_ / 8);
        ex_code_.resize(n * padded_dim_ * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);

        ip_func_ = select_excode_ipfunc(ex_bits);

        if (ivf_type_ == IVFType::DocKMeans) {
            ivf_dk_ = new IVF_DocKMeans(n_clusters, d);
        } else {
            ivf = new IVF_PG(n_clusters, d);
        }
    }

    void build_index(const float* data, size_t max_kmeans_iter = 20) {
        quantize(data);
        if (ivf_type_ == IVFType::DocKMeans) {
            ivf_dk_->build(data, doc_ptrs_.data(), num_docs, n, max_kmeans_iter);
        }
        // else: IVF_PG, call ivf->build_from_existing() separately
    }

    /***
     * @brief Quantize data points into RabitQ codes
     * @param data Data points to be quantized (n*d)
     */
    void quantize(const float* data) {
        size_t batch_size = 10240;
        for (size_t start = 0; start < n; start += batch_size) {
            // std::cout << "Quantizing data points " << start << " to "
            //           << std::min(start + batch_size, n) - 1 << " / " << n << "\n"
            //           << std::flush;
            size_t end = std::min(start + batch_size, n);
            size_t current_batch_size = end - start;

            std::vector<float> rotated_data(current_batch_size * padded_dim_);
#pragma omp parallel for
            for (size_t i = 0; i < current_batch_size; ++i) {
                // Rotate data
                rotator_->rotate(&data[(start + i) * d], &rotated_data[i * padded_dim_]);

                // Encode one-bit quantization and compute factor
                encode_one_bit(
                    &rotated_data[i * padded_dim_],
                    padded_dim_,
                    reinterpret_cast<uint64_t*>(&one_bit_code_[(start + i) * padded_dim_ / 8]),
                    &one_bit_factor_[start + i]
                );

                encode_ex_bits(
                    &rotated_data[i * padded_dim_],
                    padded_dim_,
                    ex_bits,
                    reinterpret_cast<uint8_t*>(&ex_code_[(start + i) * padded_dim_ * ex_bits / 8]),
                    &ex_factor_[start + i]
                );
            }
        }
    }

    void set_doc_mapping(const std::vector<int>& doc_lens) {
        num_docs = doc_lens.size();
        doc_ptrs_.resize(num_docs + 1);
        for (size_t i = 0; i < num_docs; ++i) {
            doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
        }
        doc_ids_.resize(n);
        for (size_t i = 0; i < num_docs; ++i) {
            for (size_t j = 0; j < doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
        if (doc_ptrs_[num_docs] != n) {
            throw std::runtime_error("Error in set_doc_mapping: total number of vectors does not match!");
        }
    }

    size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    std::vector<size_t> search(const float* queries, size_t q_doclen, size_t k, size_t nprobe) {
        int k_rank_cluster = 1800;
        int k_rank_all_tokens = 1800;

        std::vector<float> rotated_queries(q_doclen * padded_dim_);
        for (size_t i = 0; i < q_doclen; ++i) {
            rotator_->rotate(&queries[i * d], &rotated_queries[i * padded_dim_]);
        }
        std::vector<query_object> query_objs(q_doclen);
        for (size_t i = 0; i < q_doclen; ++i) {
            query_objs[i] = query_object(&rotated_queries[i * padded_dim_], padded_dim_, ex_bits);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<size_t> rank_cluster_doc_ids;
        rank_cluster_dists(query_objs.data(), q_doclen, nprobe, k_rank_cluster, rank_cluster_doc_ids);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<size_t> rank_all_tokens_ids;
        std::vector<float> one_bit_dists;
        rank_all_tokens_1bit(query_objs.data(), q_doclen, rank_cluster_doc_ids, k_rank_all_tokens, rank_all_tokens_ids, one_bit_dists);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<size_t> result;
        rank_all_tokens_exbits(
            query_objs.data(),
            q_doclen,
            rank_all_tokens_ids,
            one_bit_dists,
            k,
            result
        );
        auto t3 = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "[search] stage1_cluster: " << ms(t0, t1) << " ms, "
                  << "stage2_1bit: " << ms(t1, t2) << " ms, "
                  << "stage3_exbits: " << ms(t2, t3) << " ms, "
                  << "total: " << ms(t0, t3) << " ms\n";
        return result;
    }

    void rank_cluster_dists(query_object* queries, size_t q_doclen, size_t nprobe, size_t k, std::vector<size_t>& output_ids) {
        std::vector<bool> doc_found(num_docs, false);
        std::vector<float> doc_dists(q_doclen * num_docs);
        double gather_matrix_time = 0.0;
#pragma omp parallel for
        for (int j = 0; j < q_doclen; ++j) {
            std::vector<size_t> ids;
            ivf->search(queries[j].rotated_query, nprobe, ids);
            for (size_t idx = 0; idx < ids.size(); ++idx) {
                size_t emb_id = ids[idx];
                float dist = distance_one_bit(queries + j, &one_bit_code_[emb_id * padded_dim_ / 8], one_bit_factor_[emb_id], padded_dim_);
                int doc_id = doc_ids_[emb_id];
                doc_found[doc_id] = true;
                doc_dists[j * num_docs + doc_id] = std::max(doc_dists[j * num_docs + doc_id], dist);
            }
        }
        std::priority_queue<std::pair<float, int>> max_heap;
        for (int doc_id = 0; doc_id < num_docs; ++doc_id) {
            if (doc_found[doc_id]) {
                float score = 0.0F;
                for (int j = 0; j < q_doclen; ++j) {
                    score += doc_dists[j * num_docs + doc_id];
                }
                max_heap.emplace(score, doc_id);
            }
        }
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    void rank_cluster_dists_dockmeans(
        const float* queries,       // raw unrotated query vectors (q_doclen * d)
        size_t q_doclen,
        size_t nprobe,
        size_t k,
        std::vector<size_t>& output_ids
    ) {
        const float* centroids = ivf_dk_->centroids.data();

        // Score each centroid: SUM_i ||q_i - c_k||^2
        std::vector<float> centroid_scores(n_clusters);
#pragma omp parallel for
        for (size_t c = 0; c < n_clusters; ++c) {
            float score = 0.0f;
            for (size_t i = 0; i < q_doclen; ++i) {
                score += sqrt(l2_sqr(&queries[i * d], &centroids[c * d], d));
            }
            centroid_scores[c] = score;
        }

        // Find top nprobe centroids (smallest score)
        std::vector<size_t> cluster_order(n_clusters);
        std::iota(cluster_order.begin(), cluster_order.end(), 0);
        size_t actual_nprobe = std::min(nprobe, n_clusters);
        std::partial_sort(
            cluster_order.begin(),
            cluster_order.begin() + actual_nprobe,
            cluster_order.end(),
            [&](size_t a, size_t b) { return centroid_scores[a] < centroid_scores[b]; }
        );

        // Collect doc IDs from top nprobe clusters
        for (size_t p = 0; p < actual_nprobe; ++p) {
            size_t cid = cluster_order[p];
            size_t start = ivf_dk_->cluster_pos[cid];
            size_t end = ivf_dk_->cluster_pos[cid + 1];
            for (size_t i = start; i < end; ++i) {
                output_ids.push_back(static_cast<size_t>(ivf_dk_->inv_list[i]));
            }
        }
    }

    std::vector<size_t> search_dockmeans(const float* queries, size_t q_doclen, size_t k, size_t nprobe) {
        int k_rank_cluster = 1800;
        int k_rank_all_tokens = 1800;

        // Stage 1: brute-force centroid scan using raw (unrotated) queries
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<size_t> rank_cluster_doc_ids;
        rank_cluster_dists_dockmeans(queries, q_doclen, nprobe, k_rank_cluster, rank_cluster_doc_ids);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Rotate queries and build query objects for stages 2/3
        std::vector<float> rotated_queries(q_doclen * padded_dim_);
        for (size_t i = 0; i < q_doclen; ++i) {
            rotator_->rotate(&queries[i * d], &rotated_queries[i * padded_dim_]);
        }
        std::vector<query_object> query_objs(q_doclen);
        for (size_t i = 0; i < q_doclen; ++i) {
            query_objs[i] = query_object(&rotated_queries[i * padded_dim_], padded_dim_, ex_bits);
        }

        // Stage 2: rank all tokens with 1-bit distances (reused)
        std::vector<size_t> rank_all_tokens_ids;
        std::vector<float> one_bit_dists;
        rank_all_tokens_1bit(query_objs.data(), q_doclen, rank_cluster_doc_ids, k_rank_all_tokens, rank_all_tokens_ids, one_bit_dists);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Stage 3: refine with ex-bits (reused)
        std::vector<size_t> result;
        rank_all_tokens_exbits(
            query_objs.data(),
            q_doclen,
            rank_all_tokens_ids,
            one_bit_dists,
            k,
            result
        );
        auto t3 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "[search_dockmeans] stage1_cluster: " << ms(t0, t1) << " ms, "
                  << "stage1_doc_nums: " << rank_cluster_doc_ids.size() << ", "
                  << "stage2_1bit: " << ms(t1, t2) << " ms, "
                  << "stage3_exbits: " << ms(t2, t3) << " ms, "
                  << "total: " << ms(t0, t3) << " ms\n";
        return result;
    }

    void rank_all_tokens_1bit(
        query_object* queries, 
        size_t q_doclen, 
        std::vector<size_t>& input_ids, 
        size_t k, 
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists
    ) {
        std::unordered_map<size_t, size_t> doc_id_to_index;
        std::priority_queue<std::pair<float, size_t>> max_heap;
        size_t total_tokens = 0;
        std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            doc_id_to_index[input_ids[i]] = i;
            total_tokens += doc_len(input_ids[i]);
            candidate_doc_ptrs[i + 1] = total_tokens;
        }
        std::vector<float> token_dists(total_tokens * q_doclen);
#pragma omp parallel for
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
            for (int j = 0; j < q_doclen; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < doc_len(doc_id); ++i) {
                    size_t tid = doc_ptrs_[doc_id] + i;
                    float dist = distance_one_bit(queries + j, &one_bit_code_[tid * padded_dim_ / 8], one_bit_factor_[tid], padded_dim_);
                    token_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j] = dist;
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
#pragma omp critical
            max_heap.emplace(doc_score, doc_id);
        }
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            size_t doc_id = max_heap.top().second;
            output_ids.push_back(doc_id);
            for (size_t t = 0; t < doc_len(doc_id); ++t) {
                size_t doc_idx = doc_id_to_index[doc_id];
                for (int j = 0; j < q_doclen; ++j) {
                    one_bit_dists.push_back(token_dists[(candidate_doc_ptrs[doc_idx] + t) * q_doclen + j]);
                }
            }
            max_heap.pop();
        }
    }

    void rank_all_tokens_exbits(
        query_object* queries, 
        size_t q_doclen, 
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k, 
        std::vector<size_t>& output_ids
    ) {
        std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
        size_t total_tokens = 0;
        for (size_t i = 0; i < input_ids.size(); ++i) {
            total_tokens += doc_len(input_ids[i]);
            candidate_doc_ptrs[i + 1] = total_tokens;
        }
        std::priority_queue<std::pair<float, size_t>> max_heap;
#pragma omp parallel for
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
            for (int j = 0; j < q_doclen; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < doc_len(doc_id); ++i) {
                    size_t tid = doc_ptrs_[doc_id] + i;
                    float dist = distance_ex_bits(
                        queries + j, 
                        &ex_code_[tid * padded_dim_ * ex_bits / 8], 
                        ex_bits, 
                        ip_func_, 
                        one_bit_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j], 
                        one_bit_factor_[tid], 
                        ex_factor_[tid], 
                        padded_dim_
                    );
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
#pragma omp critical
            max_heap.emplace(doc_score, doc_id);
        }
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    void save(const std::string& filename) const {
        std::ofstream of(filename, std::ios::binary);
        size_t type_tag = static_cast<size_t>(ivf_type_);
        of.write((char*)&type_tag, sizeof(size_t));
        of.write((char*)&n, sizeof(size_t));
        of.write((char*)&d, sizeof(size_t));
        of.write((char*)&n_clusters, sizeof(size_t));
        of.write((char*)&ex_bits, sizeof(size_t));
        of.write((char*)&padded_dim_, sizeof(size_t));
        of.write(one_bit_code_.data(), one_bit_code_.size());
        of.write(ex_code_.data(), ex_code_.size());
        of.write((char*)one_bit_factor_.data(), one_bit_factor_.size() * sizeof(float));
        of.write((char*)ex_factor_.data(), ex_factor_.size() * sizeof(float));
        of.close();
        if (ivf_type_ == IVFType::DocKMeans && ivf_dk_) {
            ivf_dk_->save(filename);
        }
        // else: ivf->save(filename);
    }

    void load(const std::string& filename) {
        std::ifstream inf(filename, std::ios::binary);
        // size_t type_tag;
        // inf.read((char*)&type_tag, sizeof(size_t));
        // ivf_type_ = static_cast<IVFType>(type_tag);
        inf.read((char*)&n, sizeof(size_t));
        inf.read((char*)&d, sizeof(size_t));
        inf.read((char*)&n_clusters, sizeof(size_t));
        inf.read((char*)&ex_bits, sizeof(size_t));
        inf.read((char*)&padded_dim_, sizeof(size_t));
        one_bit_code_.resize(n * padded_dim_ / 8);
        ex_code_.resize(n * padded_dim_ * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);
        inf.read(one_bit_code_.data(), one_bit_code_.size());
        inf.read(ex_code_.data(), ex_code_.size());
        inf.read((char*)one_bit_factor_.data(), one_bit_factor_.size() * sizeof(float));
        inf.read((char*)ex_factor_.data(), ex_factor_.size() * sizeof(float));
        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, padded_dim_);
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();
        inf.close();
        if (ivf_type_ == IVFType::DocKMeans) {
            ivf_dk_ = new IVF_DocKMeans(n_clusters, d);
            ivf_dk_->load(filename);
            n_clusters = ivf_dk_->n_clusters; // ensure consistency
        } else {
            ivf = new IVF_PG(n_clusters, d);
            ivf->load(filename);
        }
    }

    ~cpu_mvr_index() {
        delete rotator_;
        delete ivf;
        delete ivf_dk_;
    }
};