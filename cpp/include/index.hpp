#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "rabitqlib/utils/rotator.hpp"
#include "estimator.hpp"
#include "ivf_pg.hpp"
#include "query.hpp"

enum class IVFType { PG, DocKMeans };

struct cpu_mvr_index {
    size_t n;            // number of vectors
    size_t d;            // dimension
    size_t n_clusters;   // number of clusters
    size_t ex_bits;      // n_bits = 1 + ex_bits
    size_t padded_dim_;  // multiple of 64 dimension after padding
    rabitqlib::Rotator<float>* rotator_;

    IVFType ivf_type_ = IVFType::PG;
    IVF_PG* ivf = nullptr;

    std::vector<char> one_bit_code_;
    std::vector<char> ex_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;

    size_t num_docs;
    std::vector<int> doc_ids_;   // document ids for each vector, size n
    std::vector<int> doc_ptrs_;  // pointers to the start of each document, size num_docs + 1

    float (*ip_func_)(const float*, const uint8_t*, size_t);

    explicit cpu_mvr_index(const std::string& filename);
    cpu_mvr_index(
        size_t n,
        size_t d,
        size_t n_clusters,
        size_t ex_bits,
        IVFType ivf_type = IVFType::PG
    );
    ~cpu_mvr_index();

    void build_index(const float* data, size_t max_kmeans_iter = 20);
    void quantize(const float* data);

    void set_doc_mapping(const std::vector<int>& doc_lens);
    size_t doc_len(size_t doc_id) const;

    void gather_ids(const float* queries, size_t q_doclen, size_t nprobe, std::vector<size_t>& output_ids);
    void gather_ids_rank_dists(const float* queries, size_t q_doclen, size_t nprobe, std::vector<size_t>& output_ids, int k);
    void gather_dists(const float* data, const float* queries, size_t q_doclen, size_t nprobe,
                      std::vector<size_t>& output_ids, std::vector<float>& output_dists);

    std::vector<size_t> search(const float* queries, size_t q_doclen, size_t k, size_t nprobe);

    void rank_cluster_dists(query_object* queries, size_t q_doclen, size_t nprobe, size_t k,
                            std::vector<size_t>& output_ids);
    void rank_all_tokens_1bit(
        query_object* queries,
        size_t q_doclen,
        std::vector<size_t>& input_ids,
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists
    );
    void rank_all_tokens_exbits(
        query_object* queries,
        size_t q_doclen,
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k,
        std::vector<size_t>& output_ids
    );

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};
