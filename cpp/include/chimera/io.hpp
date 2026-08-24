#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Chimera {

std::vector<float> load_data(size_t& num_embeddings, size_t& d, std::string filename = "embeddings.bin");

std::vector<float> load_query(size_t& q_doclen, size_t& num_q, size_t& d, std::string filename = "query_embeddings.bin");

std::vector<int> load_doclens(std::string filename = "doclens.bin");


}  // namespace Chimera
