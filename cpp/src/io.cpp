#include "chimera/io.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace Chimera {

namespace {

void require_read(
    std::ifstream& input,
    char* destination,
    size_t bytes,
    const std::string& filename) {
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Input payload is too large: " + filename);
    }
    input.read(destination, static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("Failed to read input file: " + filename);
    }
}

template <typename T>
T read_value(std::ifstream& input, const std::string& filename) {
    T value {};
    require_read(
        input,
        reinterpret_cast<char*>(&value),
        sizeof(value),
        filename);
    return value;
}

size_t checked_product(size_t lhs, size_t rhs, const std::string& filename) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error("Input dimensions overflow size_t: " + filename);
    }
    return lhs * rhs;
}

std::ifstream open_input(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filename);
    }
    return input;
}

}  // namespace

std::vector<float> load_data(size_t& num_embeddings, size_t& d, std::string filename) {
    auto input = open_input(filename);
    const int stored_count = read_value<int>(input, filename);
    const int stored_dimension = read_value<int>(input, filename);
    if (stored_count <= 0 || stored_dimension <= 0) {
        throw std::runtime_error("Embedding header contains non-positive dimensions: " + filename);
    }

    num_embeddings = static_cast<size_t>(stored_count);
    d = static_cast<size_t>(stored_dimension);
    std::vector<float> embeddings(checked_product(num_embeddings, d, filename));
    require_read(
        input,
        reinterpret_cast<char*>(embeddings.data()),
        checked_product(embeddings.size(), sizeof(float), filename),
        filename);
    return embeddings;
}

std::vector<float> load_query(size_t& q_doclen, size_t& num_q, size_t& d, std::string filename) {
    auto input = open_input(filename);
    const int stored_query_count = read_value<int>(input, filename);
    const int stored_query_length = read_value<int>(input, filename);
    const int stored_dimension = read_value<int>(input, filename);
    if (stored_query_count <= 0 || stored_query_length <= 0 || stored_dimension <= 0) {
        throw std::runtime_error("Query header contains non-positive dimensions: " + filename);
    }

    num_q = static_cast<size_t>(stored_query_count);
    q_doclen = static_cast<size_t>(stored_query_length);
    d = static_cast<size_t>(stored_dimension);
    const size_t query_vectors = checked_product(num_q, q_doclen, filename);
    std::vector<float> queries(checked_product(query_vectors, d, filename));
    require_read(
        input,
        reinterpret_cast<char*>(queries.data()),
        checked_product(queries.size(), sizeof(float), filename),
        filename);
    return queries;
}

std::vector<int> load_doclens(std::string filename) {
    auto input = open_input(filename);
    const int stored_count = read_value<int>(input, filename);
    if (stored_count <= 0) {
        throw std::runtime_error("Document-length header contains a non-positive count: " + filename);
    }

    std::vector<int> doclens(static_cast<size_t>(stored_count));
    require_read(
        input,
        reinterpret_cast<char*>(doclens.data()),
        checked_product(doclens.size(), sizeof(int), filename),
        filename);
    return doclens;
}


}  // namespace Chimera
