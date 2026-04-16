#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct MmapedEmbeddings {
    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t mapping_size_ = 0;
    const float* data_ = nullptr;
    size_t num_embeddings = 0;
    size_t d = 0;

    MmapedEmbeddings() = default;
    MmapedEmbeddings(const MmapedEmbeddings&) = delete;
    MmapedEmbeddings& operator=(const MmapedEmbeddings&) = delete;
    MmapedEmbeddings(MmapedEmbeddings&& other) noexcept;
    MmapedEmbeddings& operator=(MmapedEmbeddings&& other) noexcept;
    ~MmapedEmbeddings();

    const float* embedding_ptr(size_t index) const;
    void copy_embeddings(size_t start, size_t count, float* dst) const;
    void advise_sequential() const;
    void prefetch_embeddings(size_t start, size_t count) const;
};

std::vector<float> load_data(size_t& num_embeddings, size_t& d, std::string filename = "embeddings.bin");
MmapedEmbeddings load_data_mmap(std::string filename = "embeddings.bin");

std::vector<float> load_query(size_t& q_doclen, size_t& num_q, size_t& d, std::string filename = "query_embeddings.bin");

std::vector<int> load_doclens(std::string filename = "doclens.bin");
