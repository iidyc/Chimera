#include "gpu_kernels_v8.cuh"

namespace {

__device__ __forceinline__ uint32_t cluster_virtual_to_actual_emb_pos_v8(
    uint32_t cluster_start,
    int cluster_size,
    int virtual_local_idx
) {
    const int misalign = static_cast<int>(cluster_start & (kStage1ClusterAlignTokensV8 - 1));
    const int head = (misalign == 0)
        ? 0
        : static_cast<int>(kStage1ClusterAlignTokensV8) - misalign;
    if (head == 0 || head >= cluster_size) {
        return cluster_start + static_cast<uint32_t>(virtual_local_idx);
    }

    const int aligned_run = cluster_size - head;
    if (virtual_local_idx < aligned_run) {
        return cluster_start + static_cast<uint32_t>(head + virtual_local_idx);
    }
    return cluster_start + static_cast<uint32_t>(virtual_local_idx - aligned_run);
}

}  // namespace

__global__ void stage1_binary_ip_lut_v8_align_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          d_doc_query_max,
    int*            d_doc_touched,
    int*            d_touched_doc_list,
    int*            d_num_touched_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ float    smem_lut[LUT_ENTRIES_PER_QUERY];
    __shared__ uint32_t smem_cstart[kStage1MaxNprobeV8];
    __shared__ int      smem_prefix[kStage1MaxNprobeV8 + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        uint32_t label = my_labels[p];
        if (label < (uint32_t)n_clusters) {
            size_t start = d_cluster_pos[label];
            smem_cstart[p] = (uint32_t)start;
            smem_prefix[p + 1] = (int)(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) smem_prefix[0] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; i++)
            smem_prefix[i] += smem_prefix[i - 1];
    }
    __syncthreads();

    const int   total_elements = smem_prefix[nprobe];
    const float cb1_sumq       = d_cb1_sumq[query_idx];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x)
    {
        int lo = 0, hi = nprobe;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx)
                lo = mid + 1;
            else
                hi = mid;
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_actual_emb_pos_v8(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const float factor = d_clustered_factor[emb_pos];
        const int   doc_id = d_clustered_doc_ids[emb_pos];

        float ip = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[n * LUT_SIZE + ((code128.x >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(8 + n) * LUT_SIZE + ((code128.y >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(16 + n) * LUT_SIZE + ((code128.z >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(24 + n) * LUT_SIZE + ((code128.w >> (n * 4)) & 0xF)];

        float dist = (ip - cb1_sumq) * factor;

        if (dist > 0.0f && (size_t)doc_id < num_docs) {
            float* slot = &d_doc_query_max[(size_t)doc_id * Q_DOCLEN + query_idx];
            if (dist > *slot) {
                atomicMax(
                    reinterpret_cast<int*>(slot),
                    __float_as_int(dist));
            }

            if (d_doc_touched[doc_id] == 0) {
                int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
                if (was_touched == 0) {
                    int pos = atomicAdd(d_num_touched_docs, 1);
                    d_touched_doc_list[pos] = doc_id;
                }
            }
        }
    }
}
