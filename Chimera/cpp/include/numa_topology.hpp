#pragma once

#include <string>
#include <vector>

namespace Chimera {

enum class NumaMemoryPolicy
{
    Off,
    Preferred,
    Bind
};

std::vector<int> current_allowed_cpus();

int numa_node_for_cpu(int cpu);

int current_numa_node();

std::vector<int> allowed_numa_nodes();

std::vector<int> cpus_for_numa_node(int numa_node);

int choose_cpu_for_numa_node(int numa_node, int hint = 0);

int numa_node_for_pci_bus_id(const std::string& pci_bus_id);

std::vector<int> parse_numa_node_list(const std::string& raw, std::string* error);

bool pin_current_thread_to_cpu(int cpu);

bool set_preferred_numa_memory_policy_for_current_thread(int numa_node);

bool set_bind_numa_memory_policy_for_current_thread(int numa_node);

bool reset_numa_memory_policy_for_current_thread();

bool numa_memory_policy_from_string(
    const std::string& raw,
    NumaMemoryPolicy* policy,
    std::string* error);

bool numa_memory_policy_from_env(
    const char* env_name,
    NumaMemoryPolicy default_policy,
    NumaMemoryPolicy* policy,
    std::string* error);

const char* numa_memory_policy_name(NumaMemoryPolicy policy);

class ScopedCpuAffinity
{
  public:
    ScopedCpuAffinity() = default;
    explicit ScopedCpuAffinity(int cpu);
    ScopedCpuAffinity(const ScopedCpuAffinity&) = delete;
    ScopedCpuAffinity& operator=(const ScopedCpuAffinity&) = delete;
    ~ScopedCpuAffinity();

    bool active() const
    {
        return active_;
    }

  private:
    std::vector<unsigned char> old_mask_storage_;
    bool active_ = false;
};

class ScopedNumaMemoryPolicy
{
  public:
    ScopedNumaMemoryPolicy() = default;
    ScopedNumaMemoryPolicy(int numa_node, NumaMemoryPolicy policy);
    ScopedNumaMemoryPolicy(const ScopedNumaMemoryPolicy&) = delete;
    ScopedNumaMemoryPolicy& operator=(const ScopedNumaMemoryPolicy&) = delete;
    ~ScopedNumaMemoryPolicy();

    bool active() const
    {
        return active_;
    }

  private:
    bool active_ = false;
};

}  // namespace Chimera
