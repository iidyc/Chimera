#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace raft {
class resources;
}

namespace rabitqlib {
template <typename T>
class Rotator;
}

namespace Chimera {

struct IVF_PG;

struct SearchOptions {
    int nprobe = 128;
    int k_refine = 3000;
    int k_full_bit = 300;
    int cagra_itopk_size = 150;
    int num_chunks = 5;
};

inline void check_cuda(
    cudaError_t status,
    const char* operation = "CUDA operation") {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

class chimera_index {
  public:
    chimera_index();
    ~chimera_index();

    void build(
        const std::vector<float>& embeddings,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

    std::vector<size_t> search(
        const std::vector<float>& query,
        size_t k);
    std::vector<size_t> search(
        const float* query,
        size_t query_size,
        size_t k);

    void save(const std::string& index_dir) const;
    void load(
        const std::string& index_dir,
        const SearchOptions& options = {});

    chimera_index(const chimera_index&) = delete;
    chimera_index& operator=(const chimera_index&) = delete;

  private:
    struct QueryState;

    struct IndexHeader {
        size_t num_embeddings;
        size_t dimension;
        size_t num_clusters;
        size_t ex_bits;
        size_t padded_dimension;
    };

    struct Workspace;

    void begin_initialization(const SearchOptions& options);
    void initialize_search_state();
    std::vector<size_t> search_one(const float* queries, size_t k);
    void release_resources() noexcept;
    void set_doc_mapping(const std::vector<int>& doc_lens);
    void compute_candidate_bounds();
    void allocate_workspace();
    void ensure_candidate_capacity(size_t required_rows);
    void bind_candidate_arena(
        char* base,
        size_t row_capacity,
        size_t score_buffer_capacity);

    void generate_and_refine_candidates(
        size_t nprobe,
        size_t k_refine,
        int& num_refined_candidates,
        cudaStream_t stream = nullptr);
    void collaborative_document_scoring(
        int num_refined_candidates,
        size_t k,
        size_t k_full_bit,
        QueryState* queries,
        std::vector<size_t>& result);
    void compute_full_bit_scores(
        const int* refined_doc_ids,
        const int* full_bit_candidate_indices,
        int num_full_bit_candidates,
        const float* queries_flat,
        const QueryState* queries,
        std::pair<float, int>* full_bit_scores);

    size_t n = 0;
    size_t d = 0;
    size_t n_clusters = 0;
    size_t ex_bits = 0;
    size_t num_docs = 0;
    int max_cluster_size = 0;
    size_t max_probed_clusters_ = 0;
    size_t max_retrieved_tokens_per_query_ = 0;
    size_t max_candidate_docs_per_query_ = 0;

    rabitqlib::Rotator<float>* rotator_ = nullptr;
    IVF_PG* ivf = nullptr;
    std::vector<char> full_bit_codes_;
    std::vector<float> full_bit_factors_;
    std::vector<std::uint64_t> one_bit_codes_;
    std::vector<float> one_bit_factors_;
    std::vector<char> clustered_data_;
    std::vector<int> doc_lens_;
    std::vector<int> doc_ptrs_;
    void (*unpack_func_)(const uint8_t*, float*, size_t) = nullptr;

    int* d_doc_ptrs_ = nullptr;
    size_t* d_cluster_pos_ = nullptr;
    char* d_clustered_one_bit_codes_ = nullptr;
    float* d_clustered_one_bit_factors_ = nullptr;
    int* d_clustered_doc_ids_ = nullptr;
    uint32_t* d_token_to_cluster_pos_ = nullptr;

    uint32_t* d_candidate_bitmap_ = nullptr;
    uint32_t* d_candidate_bitmap_offsets_ = nullptr;
    size_t candidate_bitmap_bucket_count_ = 0;
    size_t candidate_bitmap_offset_count_ = 0;
    size_t max_candidate_docs_ = 0;
    size_t candidate_capacity_ = 0;
    char* d_candidate_arena_raw_ = nullptr;
    char* d_candidate_arena_ = nullptr;

    SearchOptions options_;
    std::unique_ptr<raft::resources> cagra_res_;
    std::unique_ptr<Workspace> ws_;
    bool has_index_ = false;
    bool search_ready_ = false;
};

}  // namespace Chimera
