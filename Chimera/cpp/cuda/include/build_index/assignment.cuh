#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "io.hpp"

namespace Chimera {

struct AssignmentTiming
{
    double total_ms = 0.0;
    double cagra_build_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double search_ms = 0.0;
    size_t assigned_embeddings = 0;
    size_t launched_batches = 0;
};

struct AssignmentBatch
{
    size_t start = 0;
    size_t count = 0;
    size_t dim = 0;
    int device_id = -1;
    const float* embeddings = nullptr;
    const uint32_t* labels = nullptr;
};

using AssignmentBatchCallback = std::function<void(const AssignmentBatch&)>;

std::vector<uint32_t> assign_embeddings_multi_gpu(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    AssignmentTiming* timing = nullptr,
    const AssignmentBatchCallback& batch_callback = {});

}  // namespace Chimera
