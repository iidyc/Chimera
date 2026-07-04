#include "build_index/centroid_sampling.cuh"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace Chimera {

std::vector<float> sample_centroids(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    const CentroidSampleOptions& options)
{
    if (n_clusters == 0)
    {
        throw std::runtime_error("n_clusters must be greater than zero");
    }
    if (n_clusters > data.num_embeddings)
    {
        throw std::runtime_error(
            "n_clusters cannot exceed the number of token embeddings");
    }

    const size_t bucket_size = std::max<size_t>(1, options.bucket_size);
    std::vector<float> centroids(n_clusters * data.d);

    if (bucket_size == 1)
    {
        std::vector<size_t> indices(data.num_embeddings);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(n_clusters);

        for (size_t i = 0; i < n_clusters; ++i)
        {
            std::memcpy(
                &centroids[i * data.d],
                data.embedding_ptr(indices[i]),
                data.d * sizeof(float));
        }
        return centroids;
    }

    const size_t n_buckets = (data.num_embeddings + bucket_size - 1) / bucket_size;
    std::vector<size_t> bucket_ids(n_buckets);
    std::iota(bucket_ids.begin(), bucket_ids.end(), 0);

    std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));
    std::shuffle(bucket_ids.begin(), bucket_ids.end(), rng);

    size_t copied = 0;
    for (const size_t bucket_id : bucket_ids)
    {
        if (copied == n_clusters)
        {
            break;
        }

        const size_t start = bucket_id * bucket_size;
        const size_t bucket_count =
            std::min(bucket_size, data.num_embeddings - start);
        const size_t copy_count =
            std::min(bucket_count, n_clusters - copied);
        data.copy_embeddings(
            start,
            copy_count,
            centroids.data() + copied * data.d);
        copied += copy_count;
    }

    if (copied != n_clusters)
    {
        throw std::runtime_error("Failed to sample the requested number of centroids");
    }

    return centroids;
}

}  // namespace Chimera
