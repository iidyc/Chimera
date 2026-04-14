#include "io.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

constexpr size_t kEmbeddingHeaderBytes = 2 * sizeof(int);

void reset_mmaped_embeddings(MmapedEmbeddings& embeddings) {
    embeddings.fd_ = -1;
    embeddings.mapping_ = nullptr;
    embeddings.mapping_size_ = 0;
    embeddings.data_ = nullptr;
    embeddings.num_embeddings = 0;
    embeddings.d = 0;
}

}  // namespace

MmapedEmbeddings::MmapedEmbeddings(MmapedEmbeddings&& other) noexcept {
    *this = std::move(other);
}

MmapedEmbeddings& MmapedEmbeddings::operator=(MmapedEmbeddings&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }

    fd_ = other.fd_;
    mapping_ = other.mapping_;
    mapping_size_ = other.mapping_size_;
    data_ = other.data_;
    num_embeddings = other.num_embeddings;
    d = other.d;

    reset_mmaped_embeddings(other);
    return *this;
}

MmapedEmbeddings::~MmapedEmbeddings() {
    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

const float* MmapedEmbeddings::embedding_ptr(size_t index) const {
    if (index >= num_embeddings) {
        throw std::out_of_range("Embedding index out of range in MmapedEmbeddings::embedding_ptr");
    }
    return data_ + index * d;
}

void MmapedEmbeddings::copy_embeddings(size_t start, size_t count, float* dst) const {
    if (start + count > num_embeddings) {
        throw std::out_of_range("Embedding range out of range in MmapedEmbeddings::copy_embeddings");
    }
    std::memcpy(dst, data_ + start * d, count * d * sizeof(float));
}

std::vector<float> load_data(size_t& num_embeddings, size_t& d, std::string filename) {
    int n_, d_;
    std::ifstream emb_file(filename, std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&n_), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d_), sizeof(int));
    num_embeddings = size_t(n_);
    d = size_t(d_);
    std::vector<float> embeddings(num_embeddings * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    std::cout << ">>> Loaded " << num_embeddings << " embeddings of dimension " << d << std::endl;
    return embeddings;
}

MmapedEmbeddings load_data_mmap(std::string filename) {
    MmapedEmbeddings embeddings;

    embeddings.fd_ = open(filename.c_str(), O_RDONLY);
    if (embeddings.fd_ == -1) {
        throw std::runtime_error("Failed to open embedding file for mmap: " + filename);
    }

    struct stat file_stat {};
    if (fstat(embeddings.fd_, &file_stat) == -1) {
        close(embeddings.fd_);
        throw std::runtime_error("Failed to stat embedding file for mmap: " + filename);
    }

    if (static_cast<size_t>(file_stat.st_size) < kEmbeddingHeaderBytes) {
        close(embeddings.fd_);
        throw std::runtime_error("Embedding file is too small to contain a header: " + filename);
    }

    embeddings.mapping_size_ = static_cast<size_t>(file_stat.st_size);
    embeddings.mapping_ =
        mmap(nullptr, embeddings.mapping_size_, PROT_READ, MAP_PRIVATE, embeddings.fd_, 0);
    if (embeddings.mapping_ == MAP_FAILED) {
        close(embeddings.fd_);
        throw std::runtime_error("Failed to mmap embedding file: " + filename);
    }

    int n_ = 0;
    int d_ = 0;
    const auto* mapped_bytes = static_cast<const char*>(embeddings.mapping_);
    std::memcpy(&n_, mapped_bytes, sizeof(int));
    std::memcpy(&d_, mapped_bytes + sizeof(int), sizeof(int));

    const size_t num_embeddings = static_cast<size_t>(n_);
    const size_t dim = static_cast<size_t>(d_);
    const size_t expected_size =
        kEmbeddingHeaderBytes + num_embeddings * dim * sizeof(float);

    if (expected_size != embeddings.mapping_size_) {
        munmap(embeddings.mapping_, embeddings.mapping_size_);
        close(embeddings.fd_);
        reset_mmaped_embeddings(embeddings);
        throw std::runtime_error(
            "Embedding file size does not match header metadata: " + filename
        );
    }

    embeddings.num_embeddings = num_embeddings;
    embeddings.d = dim;
    embeddings.data_ =
        reinterpret_cast<const float*>(mapped_bytes + kEmbeddingHeaderBytes);

    std::cout << ">>> Memory-mapped " << embeddings.num_embeddings
              << " embeddings of dimension " << embeddings.d << std::endl;
    return embeddings;
}

std::vector<float> load_query(size_t& q_doclen, size_t& num_q, size_t& d, std::string filename) {
    std::ifstream qemb_file(filename, std::ios::binary);
    int num_q_, q_doclen_, d_;
    qemb_file.read(reinterpret_cast<char*>(&num_q_), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&q_doclen_), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&d_), sizeof(int));
    num_q = size_t(num_q_);
    q_doclen = size_t(q_doclen_);
    d = size_t(d_);
    std::vector<float> Q(num_q * q_doclen * d);
    qemb_file.read(reinterpret_cast<char*>(Q.data()), Q.size() * sizeof(float));
    qemb_file.close();
    std::cout << ">>> Loaded " << num_q << " queries, each with " << q_doclen << " embeddings of dimension " << d << std::endl;
    return Q;
}

std::vector<int> load_doclens(std::string filename) {
    std::ifstream doc_lens_file(filename, std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    std::vector<int> doclens(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::cout << ">>> Loaded " << doclens_size << " document lengths" << std::endl;
    return doclens;
}
