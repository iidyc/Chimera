#pragma once

#include <cstddef>
#include <vector>

#include "build_index.hpp"
#include "io.hpp"

namespace Chimera {

std::vector<float> sample_centroids(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    const CentroidSampleOptions& options);

}  // namespace Chimera
