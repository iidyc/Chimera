#include "build_index/assignment.cuh"

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <raft/core/resource/device_memory_resource.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "build_index/build_utils.cuh"
#include "ivf_pg.hpp"
#include "numa_topology.hpp"

namespace Chimera {
namespace {

constexpr const char* kCagraSearchStartMarker = "build_index:carga_search_start";
constexpr const char* kCagraSearchEndMarker = "build_index:carga_search_end";
constexpr const char* kCagraWorkspaceMbEnv = "CHIMERA_CAGRA_WS_MB";
constexpr const char* kLegacyCagraWorkspaceBytesEnv =
    "CHIMERA_BUILD_ASSIGN_CAGRA_WORKSPACE_BYTES";
constexpr size_t kMiB = 1024ULL * 1024ULL;
constexpr size_t kDefaultAssignmentCagraWorkspaceMiB = 512ULL;
constexpr size_t kCagraWorkspacePoolFloorBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kCagraWorkspacePoolAlignmentBytes = 16ULL * 1024ULL * 1024ULL;

struct AssignmentProfileState
{
    size_t batch_limit = 0;
    std::atomic<size_t> next_batch_id {0};
    std::atomic<size_t> completed_searches {0};
    bool capture_started = false;
    bool capture_stopped = false;
    bool exit_after_capture = false;
    std::mutex capture_mutex;
    std::mutex output_mutex;

    bool enabled() const
    {
        return batch_limit > 0;
    }
};

struct AssignmentDeviceTiming
{
    int device_id = -1;
    double total_ms = 0.0;
    double cagra_build_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double search_ms = 0.0;
    size_t assigned_embeddings = 0;
    size_t launched_batches = 0;
};

size_t parse_assignment_profile_batch_limit()
{
    const char* value = std::getenv("CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES");
    if (value == nullptr || *value == '\0')
    {
        return 0;
    }

    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0')
    {
        throw std::runtime_error(
            "CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES must be a non-negative integer");
    }
    return static_cast<size_t>(parsed);
}

bool parse_assignment_profile_exit_after_capture()
{
    const char* value = std::getenv("CHIMERA_BUILD_ASSIGN_PROFILE_EXIT_AFTER_CAPTURE");
    if (value == nullptr || *value == '\0')
    {
        return false;
    }
    return std::string(value) != "0";
}

NumaMemoryPolicy parse_assignment_numa_policy()
{
    NumaMemoryPolicy policy = NumaMemoryPolicy::Bind;
    std::string error;
    const char* raw = std::getenv("CHIMERA_BUILD_ASSIGN_NUMA_POLICY");
    if (raw == nullptr || raw[0] == '\0')
    {
        raw = std::getenv("CHIMERA_NUMA_MEM_POLICY");
    }

    if (raw == nullptr || raw[0] == '\0')
    {
        return policy;
    }

    if (!numa_memory_policy_from_string(raw, &policy, &error))
    {
        throw std::runtime_error(error);
    }
    return policy;
}

size_t parse_size_env_or_default(const char* env_name, size_t default_value)
{
    const char* value = std::getenv(env_name);
    if (value == nullptr || *value == '\0')
    {
        return default_value;
    }

    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0')
    {
        throw std::runtime_error(std::string(env_name) + " must be a positive integer");
    }
    if (parsed == 0 || parsed > std::numeric_limits<size_t>::max())
    {
        throw std::runtime_error(std::string(env_name) + " must be a positive integer");
    }
    return static_cast<size_t>(parsed);
}

size_t parse_size_env_allow_zero_or_default(const char* env_name, size_t default_value)
{
    const char* value = std::getenv(env_name);
    if (value == nullptr || *value == '\0')
    {
        return default_value;
    }

    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max())
    {
        throw std::runtime_error(std::string(env_name) + " must be a non-negative integer");
    }
    return static_cast<size_t>(parsed);
}

size_t parse_cagra_workspace_bytes(const char** used_env_name, bool* used_default)
{
    if (used_env_name != nullptr)
    {
        *used_env_name = kCagraWorkspaceMbEnv;
    }
    if (used_default != nullptr)
    {
        *used_default = false;
    }

    const char* value = std::getenv(kCagraWorkspaceMbEnv);
    if (value != nullptr && *value != '\0')
    {
        const size_t requested_mib =
            parse_size_env_allow_zero_or_default(kCagraWorkspaceMbEnv, 0);
        if (requested_mib > std::numeric_limits<size_t>::max() / kMiB)
        {
            throw std::runtime_error(
                std::string(kCagraWorkspaceMbEnv) + " is too large");
        }
        return requested_mib * kMiB;
    }

    value = std::getenv(kLegacyCagraWorkspaceBytesEnv);
    if (value != nullptr && *value != '\0')
    {
        if (used_env_name != nullptr)
        {
            *used_env_name = kLegacyCagraWorkspaceBytesEnv;
        }
        return parse_size_env_allow_zero_or_default(kLegacyCagraWorkspaceBytesEnv, 0);
    }

    if (used_default != nullptr)
    {
        *used_default = true;
    }
    return kDefaultAssignmentCagraWorkspaceMiB * kMiB;
}

size_t ceil_div(size_t numerator, size_t denominator)
{
    return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

size_t align_up(size_t value, size_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }
    return ceil_div(value, alignment) * alignment;
}

void configure_assignment_cagra_workspace(raft::resources& res, int device_id)
{
    const char* used_env_name = kCagraWorkspaceMbEnv;
    bool used_default = false;
    const size_t requested_bytes =
        parse_cagra_workspace_bytes(&used_env_name, &used_default);
    if (requested_bytes == 0)
    {
        static std::mutex output_mutex;
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "[build_index][cagra] device=" << device_id
                  << " assignment_workspace_pool=disabled" << std::endl;
        return;
    }

    const size_t pool_bytes = std::max(
        kCagraWorkspacePoolFloorBytes,
        align_up(requested_bytes, kCagraWorkspacePoolAlignmentBytes));
    raft::resource::set_workspace_to_pool_resource(res, pool_bytes);

    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << "[build_index][cagra] device=" << device_id
              << " assignment_workspace_pool_bytes=" << pool_bytes
              << " env=" << (used_default ? "default" : used_env_name)
              << std::endl;
}

size_t assignment_pipeline_slots(size_t assign_batch, size_t dim)
{
    constexpr size_t kDefaultQueryBufferBytesPerSide = 1ULL << 30;
    constexpr size_t kMinSlots = 2;
    constexpr size_t kMaxSlots = 128;

    const size_t explicit_slots =
        parse_size_env_or_default("CHIMERA_BUILD_ASSIGN_BUFFER_SLOTS", 0);
    if (explicit_slots > 0)
    {
        return std::max(kMinSlots, std::min(kMaxSlots, explicit_slots));
    }

    const size_t target_bytes = parse_size_env_or_default(
        "CHIMERA_BUILD_ASSIGN_PIPELINE_BYTES",
        kDefaultQueryBufferBytesPerSide);
    const size_t query_bytes = assign_batch * dim * sizeof(float);
    if (query_bytes == 0)
    {
        return kMinSlots;
    }
    return std::max(kMinSlots, std::min(kMaxSlots, ceil_div(target_bytes, query_bytes)));
}

std::string cuda_device_pci_bus_id(int device_id)
{
    char bus_id[32] {};
    BUILD_CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_id));
    return std::string(bus_id);
}

struct DeviceNumaBinding
{
    std::string pci_bus_id;
    int numa_node = -1;
    int cpu = -1;
};

DeviceNumaBinding resolve_device_numa_binding(int device_id)
{
    DeviceNumaBinding binding;
    binding.pci_bus_id = cuda_device_pci_bus_id(device_id);
    binding.numa_node = numa_node_for_pci_bus_id(binding.pci_bus_id);
    binding.cpu = choose_cpu_for_numa_node(binding.numa_node, device_id);
    return binding;
}

void log_device_numa_binding(
    int device_id,
    const DeviceNumaBinding& binding,
    NumaMemoryPolicy policy,
    const ScopedCpuAffinity& cpu_affinity,
    const ScopedNumaMemoryPolicy& memory_policy)
{
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << "[build_index][numa] device=" << device_id
              << " pci_bus_id=" << binding.pci_bus_id
              << " numa_node=" << binding.numa_node
              << " cpu=" << binding.cpu
              << " cpu_affinity=" << (cpu_affinity.active() ? "applied" : "skipped")
              << " memory_policy=" << numa_memory_policy_name(policy)
              << " memory_policy_state="
              << (memory_policy.active() ? "applied" : "skipped")
              << std::endl;
}

void mark_cagra_search_start(
    AssignmentProfileState& profile_state,
    int device_id,
    size_t profile_batch_id)
{
    nvtxEventAttributes_t attr {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFF2E86DE;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = kCagraSearchStartMarker;

    if (profile_state.enabled())
    {
        std::lock_guard<std::mutex> lock(profile_state.capture_mutex);
        if (!profile_state.capture_started)
        {
            BUILD_CUDA_CHECK(cudaProfilerStart());
            profile_state.capture_started = true;
            std::lock_guard<std::mutex> output_lock(profile_state.output_mutex);
            std::cout << "[build_index][profile] Started profiler capture before "
                      << "assignment batch " << profile_batch_id
                      << " CAGRA search on device " << device_id << "." << std::endl;
        }
    }

    nvtxMarkEx(&attr);
}

void mark_cagra_search_end()
{
    nvtxEventAttributes_t attr {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFFE74C3C;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = kCagraSearchEndMarker;

    nvtxMarkEx(&attr);
}

void stop_profile_capture_after_search_if_needed(
    AssignmentProfileState& profile_state,
    int device_id,
    size_t profile_batch_id)
{
    const size_t completed =
        profile_state.completed_searches.fetch_add(1, std::memory_order_relaxed) + 1;
    if (completed < profile_state.batch_limit)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(profile_state.capture_mutex);
    if (profile_state.capture_stopped)
    {
        return;
    }

    BUILD_CUDA_CHECK(cudaProfilerStop());
    profile_state.capture_stopped = true;
    std::lock_guard<std::mutex> output_lock(profile_state.output_mutex);
    std::cout << "[build_index][profile] Stopped profiler capture after "
              << completed << " profiled CAGRA search batch(es); last batch "
              << profile_batch_id << " on device " << device_id << "." << std::endl;
}

void assign_shard_batches_on_device(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    int device_id,
    std::atomic<size_t>& next_batch_start,
    std::atomic<bool>& should_stop,
    std::vector<uint32_t>& list_nos,
    ProgressState& progress,
    AssignmentProfileState& profile_state,
    std::vector<AssignmentDeviceTiming>& device_timings,
    std::mutex& device_timings_mutex,
    const AssignmentBatchCallback& batch_callback)
{
    const auto device_total_start = std::chrono::steady_clock::now();
    AssignmentDeviceTiming device_timing;
    device_timing.device_id = device_id;
    BUILD_CUDA_CHECK(cudaSetDevice(device_id));
    const NumaMemoryPolicy numa_policy = parse_assignment_numa_policy();
    const DeviceNumaBinding numa_binding =
        numa_policy == NumaMemoryPolicy::Off ?
        DeviceNumaBinding {cuda_device_pci_bus_id(device_id), -1, -1} :
        resolve_device_numa_binding(device_id);
    ScopedCpuAffinity cpu_affinity(numa_binding.cpu);
    ScopedNumaMemoryPolicy memory_policy(numa_binding.numa_node, numa_policy);
    log_device_numa_binding(
        device_id,
        numa_binding,
        numa_policy,
        cpu_affinity,
        memory_policy);

    struct BufferSlot
    {
        cudaStream_t stream = nullptr;
        float* d_q_ptr = nullptr;
        uint32_t* d_lab_ptr = nullptr;
        float* d_dist_ptr = nullptr;
        float* h_q_ptr = nullptr;
        uint32_t* h_labels = nullptr;
        size_t start = 0;
        size_t cur = 0;
        bool in_flight = false;
        bool profile_enabled = false;
        size_t profile_batch_id = 0;
        double host_load_ms = 0.0;
        double host_enqueue_ms = 0.0;
        cudaEvent_t h2d_start = nullptr;
        cudaEvent_t h2d_end = nullptr;
        cudaEvent_t search_end = nullptr;
        cudaEvent_t d2h_start = nullptr;
        cudaEvent_t d2h_end = nullptr;
    };

    const size_t buffer_count = assignment_pipeline_slots(assign_batch, data.d);
    std::vector<BufferSlot> buffers(buffer_count);
    {
        static std::mutex output_mutex;
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "[build_index][prefetch] device=" << device_id
                  << " pipeline_slots=" << buffer_count
                  << " query_bytes_per_slot="
                  << (assign_batch * data.d * sizeof(float))
                  << " host_query_buffer_bytes="
                  << (buffer_count * assign_batch * data.d * sizeof(float))
                  << " device_query_buffer_bytes="
                  << (buffer_count * assign_batch * data.d * sizeof(float))
                  << std::endl;
    }

    auto cleanup = [&]()
    {
        for (auto& buffer : buffers)
        {
            if (buffer.h_labels != nullptr)
            {
                cudaFreeHost(buffer.h_labels);
            }
            if (buffer.h_q_ptr != nullptr)
            {
                cudaFreeHost(buffer.h_q_ptr);
            }
            if (buffer.d_dist_ptr != nullptr)
            {
                cudaFree(buffer.d_dist_ptr);
            }
            if (buffer.d_lab_ptr != nullptr)
            {
                cudaFree(buffer.d_lab_ptr);
            }
            if (buffer.d_q_ptr != nullptr)
            {
                cudaFree(buffer.d_q_ptr);
            }
            if (buffer.stream != nullptr)
            {
                cudaStreamDestroy(buffer.stream);
            }
            if (buffer.h2d_start != nullptr)
            {
                cudaEventDestroy(buffer.h2d_start);
            }
            if (buffer.h2d_end != nullptr)
            {
                cudaEventDestroy(buffer.h2d_end);
            }
            if (buffer.search_end != nullptr)
            {
                cudaEventDestroy(buffer.search_end);
            }
            if (buffer.d2h_end != nullptr)
            {
                cudaEventDestroy(buffer.d2h_end);
            }
            if (buffer.d2h_start != nullptr)
            {
                cudaEventDestroy(buffer.d2h_start);
            }
        }
    };

    try
    {
        const auto cagra_build_start = std::chrono::steady_clock::now();
        PG_CAGRA assignment_cagra(n_clusters, data.d);
        configure_assignment_cagra_workspace(assignment_cagra.res_, device_id);
        assignment_cagra.build_index(centroids.data());
        const auto cagra_build_end = std::chrono::steady_clock::now();
        device_timing.cagra_build_ms =
            elapsed_ms(cagra_build_start, cagra_build_end);

        const auto buffer_alloc_start = std::chrono::steady_clock::now();
        for (auto& buffer : buffers)
        {
            BUILD_CUDA_CHECK(cudaStreamCreate(&buffer.stream));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_q_ptr, assign_batch * data.d * sizeof(float)));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_lab_ptr, assign_batch * sizeof(uint32_t)));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_dist_ptr, assign_batch * sizeof(float)));
            BUILD_CUDA_CHECK(
                cudaMallocHost(&buffer.h_q_ptr, assign_batch * data.d * sizeof(float)));
            BUILD_CUDA_CHECK(cudaMallocHost(&buffer.h_labels, assign_batch * sizeof(uint32_t)));
            if (profile_state.enabled())
            {
                BUILD_CUDA_CHECK(cudaEventCreate(&buffer.h2d_start));
                BUILD_CUDA_CHECK(cudaEventCreate(&buffer.h2d_end));
                BUILD_CUDA_CHECK(cudaEventCreate(&buffer.search_end));
                BUILD_CUDA_CHECK(cudaEventCreate(&buffer.d2h_start));
                BUILD_CUDA_CHECK(cudaEventCreate(&buffer.d2h_end));
            }
        }
        const auto buffer_alloc_end = std::chrono::steady_clock::now();
        device_timing.buffer_alloc_ms =
            elapsed_ms(buffer_alloc_start, buffer_alloc_end);

        auto finalize_buffer = [&](BufferSlot& buffer)
        {
            if (!buffer.in_flight)
            {
                return;
            }

            if (buffer.profile_enabled)
            {
                BUILD_CUDA_CHECK(cudaEventSynchronize(buffer.search_end));
                stop_profile_capture_after_search_if_needed(
                    profile_state,
                    device_id,
                    buffer.profile_batch_id);

                BUILD_CUDA_CHECK(cudaEventRecord(buffer.d2h_start, buffer.stream));
                BUILD_CUDA_CHECK(cudaMemcpyAsync(
                    buffer.h_labels,
                    buffer.d_lab_ptr,
                    buffer.cur * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    buffer.stream));
                BUILD_CUDA_CHECK(cudaEventRecord(buffer.d2h_end, buffer.stream));
                BUILD_CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
            }
            else
            {
                BUILD_CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
            }

            if (batch_callback)
            {
                AssignmentBatch batch;
                batch.start = buffer.start;
                batch.count = buffer.cur;
                batch.dim = data.d;
                batch.device_id = device_id;
                batch.embeddings = buffer.h_q_ptr;
                batch.labels = buffer.h_labels;
                batch_callback(batch);
            }

            std::memcpy(
                list_nos.data() + buffer.start,
                buffer.h_labels,
                buffer.cur * sizeof(uint32_t));
            report_progress(progress, buffer.cur);
            if (buffer.profile_enabled)
            {
                float h2d_ms = 0.0f;
                float search_ms = 0.0f;
                float d2h_ms = 0.0f;
                float total_gpu_ms = 0.0f;
                BUILD_CUDA_CHECK(
                    cudaEventElapsedTime(&h2d_ms, buffer.h2d_start, buffer.h2d_end));
                BUILD_CUDA_CHECK(
                    cudaEventElapsedTime(&search_ms, buffer.h2d_end, buffer.search_end));
                BUILD_CUDA_CHECK(
                    cudaEventElapsedTime(&d2h_ms, buffer.d2h_start, buffer.d2h_end));
                BUILD_CUDA_CHECK(
                    cudaEventElapsedTime(&total_gpu_ms, buffer.h2d_start, buffer.d2h_end));

                std::lock_guard<std::mutex> lock(profile_state.output_mutex);
                std::cout << std::fixed << std::setprecision(3)
                          << "[build_index][profile] assignment batch "
                          << buffer.profile_batch_id
                          << " device=" << device_id
                          << " start=" << buffer.start
                          << " count=" << buffer.cur
                          << " host_load_ms=" << buffer.host_load_ms
                          << " host_enqueue_ms=" << buffer.host_enqueue_ms
                          << " gpu_h2d_ms=" << h2d_ms
                          << " gpu_cagra_ms=" << search_ms
                          << " gpu_d2h_ms=" << d2h_ms
                          << " gpu_total_ms=" << total_gpu_ms
                          << std::defaultfloat << std::endl;
                if (profile_state.exit_after_capture && profile_state.capture_stopped)
                {
                    std::cout << "[build_index][profile] Exiting after focused "
                              << "assignment capture by request." << std::endl;
                    std::exit(0);
                }
            }
            buffer.in_flight = false;
            buffer.profile_enabled = false;
        };

        auto launch_buffer = [&](BufferSlot& buffer, size_t start, size_t cur)
        {
            buffer.profile_enabled = false;
            if (profile_state.enabled())
            {
                const size_t profile_batch_id =
                    profile_state.next_batch_id.fetch_add(1, std::memory_order_relaxed);
                if (profile_batch_id < profile_state.batch_limit)
                {
                    buffer.profile_enabled = true;
                    buffer.profile_batch_id = profile_batch_id;
                }
            }

            const auto load_start = std::chrono::steady_clock::now();
            data.copy_embeddings(start, cur, buffer.h_q_ptr);
            const auto load_end = std::chrono::steady_clock::now();
            if (buffer.profile_enabled)
            {
                buffer.host_load_ms = elapsed_ms(load_start, load_end);
                BUILD_CUDA_CHECK(cudaEventRecord(buffer.h2d_start, buffer.stream));
            }

            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                buffer.d_q_ptr,
                buffer.h_q_ptr,
                cur * data.d * sizeof(float),
                cudaMemcpyHostToDevice,
                buffer.stream));
            if (buffer.profile_enabled)
            {
                BUILD_CUDA_CHECK(cudaEventRecord(buffer.h2d_end, buffer.stream));
                mark_cagra_search_start(
                    profile_state,
                    device_id,
                    buffer.profile_batch_id);
            }

            assignment_cagra.search_batch_gpu(
                buffer.d_q_ptr,
                cur,
                1,
                buffer.d_dist_ptr,
                buffer.d_lab_ptr,
                buffer.stream);
            if (buffer.profile_enabled)
            {
                mark_cagra_search_end();
            }
            if (buffer.profile_enabled)
            {
                BUILD_CUDA_CHECK(cudaEventRecord(buffer.search_end, buffer.stream));
            }

            if (!buffer.profile_enabled)
            {
                BUILD_CUDA_CHECK(cudaMemcpyAsync(
                    buffer.h_labels,
                    buffer.d_lab_ptr,
                    cur * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    buffer.stream));
            }
            const auto enqueue_end = std::chrono::steady_clock::now();
            if (buffer.profile_enabled)
            {
                buffer.host_enqueue_ms = elapsed_ms(load_end, enqueue_end);
            }

            buffer.start = start;
            buffer.cur = cur;
            buffer.in_flight = true;
        };

        const auto search_start = std::chrono::steady_clock::now();
        size_t next_buffer = 0;
        while (!should_stop.load(std::memory_order_relaxed))
        {
            BufferSlot& buffer = buffers[next_buffer];
            finalize_buffer(buffer);

            if (profile_state.enabled() &&
                profile_state.exit_after_capture &&
                profile_state.next_batch_id.load(std::memory_order_relaxed) >=
                    profile_state.batch_limit)
            {
                break;
            }

            const size_t start =
                next_batch_start.fetch_add(assign_batch, std::memory_order_relaxed);
            if (start >= data.num_embeddings)
            {
                break;
            }

            const size_t cur = std::min(assign_batch, data.num_embeddings - start);
            launch_buffer(buffer, start, cur);
            device_timing.assigned_embeddings += cur;
            device_timing.launched_batches += 1;
            next_buffer = (next_buffer + 1) % buffer_count;
        }

        for (auto& buffer : buffers)
        {
            finalize_buffer(buffer);
        }
        const auto search_end = std::chrono::steady_clock::now();
        device_timing.search_ms = elapsed_ms(search_start, search_end);
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    cleanup();
    const auto device_total_end = std::chrono::steady_clock::now();
    device_timing.total_ms = elapsed_ms(device_total_start, device_total_end);
    {
        std::lock_guard<std::mutex> lock(device_timings_mutex);
        device_timings.push_back(device_timing);
    }
}

}  // namespace

std::vector<uint32_t> assign_embeddings_multi_gpu(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    AssignmentTiming* timing,
    const AssignmentBatchCallback& batch_callback)
{
    auto device_ids = visible_gpu_ids();
    const size_t total_batches = (data.num_embeddings + assign_batch - 1) / assign_batch;
    const size_t worker_count = std::min(device_ids.size(), total_batches);
    device_ids.resize(worker_count);

    std::cout << "[build_index] Step 2: Assigning embeddings with " << worker_count
              << " visible GPU(s)." << std::endl;

    std::vector<uint32_t> list_nos(data.num_embeddings);
    if (worker_count == 0)
    {
        return list_nos;
    }

    std::atomic<size_t> next_batch_start {0};
    std::atomic<bool> should_stop {false};
    ProgressState progress;
    progress.step_label = "Step 2";
    progress.status_label = "documents handled";
    progress.total_items = data.num_embeddings;
    progress.start_time = std::chrono::steady_clock::now();
    AssignmentProfileState profile_state;
    profile_state.batch_limit = parse_assignment_profile_batch_limit();
    profile_state.exit_after_capture = parse_assignment_profile_exit_after_capture();
    if (profile_state.enabled())
    {
        std::cout << "[build_index][profile] Profiling first "
                  << profile_state.batch_limit
                  << " assignment batch(es) with NVTX markers "
                  << kCagraSearchStartMarker << " and "
                  << kCagraSearchEndMarker
                  << ". Set CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES=0 to disable.";
        if (profile_state.exit_after_capture)
        {
            std::cout << " Will exit after the focused capture.";
        }
        std::cout
                  << std::endl;
    }
    std::mutex error_mutex;
    std::mutex device_timings_mutex;
    std::exception_ptr first_error;
    std::vector<AssignmentDeviceTiming> device_timings;
    device_timings.reserve(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (const int device_id : device_ids)
    {
        workers.emplace_back([&, device_id]()
        {
            try
            {
                assign_shard_batches_on_device(
                    data,
                    centroids,
                    n_clusters,
                    assign_batch,
                    device_id,
                    next_batch_start,
                    should_stop,
                    list_nos,
                    progress,
                    profile_state,
                    device_timings,
                    device_timings_mutex,
                    batch_callback);
            }
            catch (...)
            {
                should_stop.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error)
                {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    if (first_error)
    {
        std::rethrow_exception(first_error);
    }

    AssignmentTiming aggregate_timing;
    for (const auto& device_timing : device_timings)
    {
        aggregate_timing.total_ms =
            std::max(aggregate_timing.total_ms, device_timing.total_ms);
        aggregate_timing.cagra_build_ms =
            std::max(aggregate_timing.cagra_build_ms, device_timing.cagra_build_ms);
        aggregate_timing.buffer_alloc_ms =
            std::max(aggregate_timing.buffer_alloc_ms, device_timing.buffer_alloc_ms);
        aggregate_timing.search_ms =
            std::max(aggregate_timing.search_ms, device_timing.search_ms);
        aggregate_timing.assigned_embeddings += device_timing.assigned_embeddings;
        aggregate_timing.launched_batches += device_timing.launched_batches;
    }
    if (timing != nullptr)
    {
        *timing = aggregate_timing;
    }

    {
        std::lock_guard<std::mutex> lock(progress.output_mutex);
        std::cout << "[build_index] Step 2 timing summary: "
                  << "max_worker_total="
                  << format_elapsed(aggregate_timing.total_ms)
                  << ", step2a_temp_cagra_build="
                  << format_elapsed(aggregate_timing.cagra_build_ms)
                  << ", step2a_buffer_alloc="
                  << format_elapsed(aggregate_timing.buffer_alloc_ms)
                  << ", step2b_assignment_search="
                  << format_elapsed(aggregate_timing.search_ms)
                  << ", assigned_embeddings="
                  << aggregate_timing.assigned_embeddings
                  << ", launched_batches="
                  << aggregate_timing.launched_batches
                  << "." << std::endl;
        for (const auto& device_timing : device_timings)
        {
            std::cout << "[build_index] Step 2 device timing: device="
                      << device_timing.device_id
                      << ", total=" << format_elapsed(device_timing.total_ms)
                      << ", temp_cagra_build="
                      << format_elapsed(device_timing.cagra_build_ms)
                      << ", buffer_alloc="
                      << format_elapsed(device_timing.buffer_alloc_ms)
                      << ", assignment_search="
                      << format_elapsed(device_timing.search_ms)
                      << ", assigned_embeddings="
                      << device_timing.assigned_embeddings
                      << ", launched_batches="
                      << device_timing.launched_batches
                      << "." << std::endl;
        }
    }

    BUILD_CUDA_CHECK(cudaSetDevice(device_ids.front()));
    return list_nos;
}

}  // namespace Chimera
