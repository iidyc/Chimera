//
// Created by Administrator on 2025/1/10.
//

#ifndef QUERYCORESET_HPP
#define QUERYCORESET_HPP

#include "include/alg/MatrixMulBLAS.hpp"

namespace VectorSetSearch
{
    class QueryCorset
    {
    public:
        uint32_t query_n_vec_, vec_dim_;
        float score_thres_;
        std::vector<float> query_vec_score_table_; // query_n_vec * query_n_vec
        std::vector<char> is_clustered_l_; // query_n_vec, default value is false

        QueryCorset() = default;

        QueryCorset(uint32_t query_n_vec, const uint32_t vec_dim, const float score_thres)
        {
            this->query_n_vec_ = query_n_vec;
            this->vec_dim_ = vec_dim;
            this->score_thres_ = score_thres;
            query_vec_score_table_.resize(query_n_vec_ * query_n_vec_);
            is_clustered_l_.resize(query_n_vec_, false);
        }

        void compute_query_corset(const float* query,
                                  float* query_corset, uint32_t& query_n_vec_actual)
        {
            is_clustered_l_.assign(query_n_vec_, false);
            MatrixMultiply(query, query, query_n_vec_, query_n_vec_, vec_dim_,
                           query_vec_score_table_.data());
            query_n_vec_actual = 0;
            for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            {
                if (is_clustered_l_[qvecID]) continue;
                is_clustered_l_[qvecID] = true;
                const float* query_vec = query + qvecID * vec_dim_;
                float* query_corset_vec = query_corset + query_n_vec_actual * vec_dim_;
                std::memcpy(query_corset_vec, query_vec, vec_dim_ * sizeof(float));
                for (uint32_t qvecID2 = qvecID + 1; qvecID2 < query_n_vec_; qvecID2++)
                {
                    if (is_clustered_l_[qvecID2]) continue;
                    if (query_vec_score_table_[qvecID * query_n_vec_ + qvecID2] >= score_thres_)
                    {
                        assert(!is_clustered_l_[qvecID2]);
                        is_clustered_l_[qvecID2] = true;
                        const float* query_vec2 = query + qvecID2 * vec_dim_;
                        std::transform(query_corset_vec,
                                       query_corset_vec + vec_dim_,
                                       query_vec2,
                                       query_corset_vec,
                                       std::plus<>());
                    }
                }
                query_n_vec_actual++;
            }

            assert(query_n_vec_actual <= query_n_vec_);
        }

        void compute_query_corset(const float* query,
                                  float* query_corset, uint32_t* corset_n_qvec_l, uint32_t& query_n_vec_actual)
        {
            // corset_n_qvec_l should be assigned with 0 at the initial stage
            is_clustered_l_.assign(query_n_vec_, false);
            MatrixMultiply(query, query, query_n_vec_, query_n_vec_, vec_dim_,
                           query_vec_score_table_.data());
            query_n_vec_actual = 0;
            for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            {
                if (is_clustered_l_[qvecID]) continue;
                is_clustered_l_[qvecID] = true;
                const float* query_vec = query + qvecID * vec_dim_;
                float* query_corset_vec = query_corset + query_n_vec_actual * vec_dim_;
                std::memcpy(query_corset_vec, query_vec, vec_dim_ * sizeof(float));
                if(!(corset_n_qvec_l[query_n_vec_actual] == 0))
                {
                    printf("query_n_vec_actual %d, corset_n_qvec_l[query_n_vec_actual] %d\n",
                        query_n_vec_actual, corset_n_qvec_l[query_n_vec_actual]);
                }
                assert(corset_n_qvec_l[query_n_vec_actual] == 0);
                corset_n_qvec_l[query_n_vec_actual]++;
                for (uint32_t qvecID2 = qvecID + 1; qvecID2 < query_n_vec_; qvecID2++)
                {
                    if (is_clustered_l_[qvecID2]) continue;
                    if (query_vec_score_table_[qvecID * query_n_vec_ + qvecID2] >= score_thres_)
                    {
                        assert(!is_clustered_l_[qvecID2]);
                        is_clustered_l_[qvecID2] = true;
                        const float* query_vec2 = query + qvecID2 * vec_dim_;
                        std::transform(query_corset_vec,
                                       query_corset_vec + vec_dim_,
                                       query_vec2,
                                       query_corset_vec,
                                       std::plus<>());
                        corset_n_qvec_l[query_n_vec_actual]++;
                    }
                }
                query_n_vec_actual++;
            }

            assert(query_n_vec_actual <= query_n_vec_);
        }
    };
}
#endif //QUERYCORESET_HPP
