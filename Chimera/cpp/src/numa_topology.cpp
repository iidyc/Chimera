#include "numa_topology.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Chimera {
namespace {

std::string trim(std::string text)
{
    const auto not_space = [](unsigned char ch)
    {
        return !std::isspace(ch);
    };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

std::string lower_copy(std::string text)
{
    for (char& ch : text)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return text;
}

bool read_small_text_file(const std::filesystem::path& path, std::string* out)
{
    if (out == nullptr)
    {
        return false;
    }
    std::ifstream in(path);
    if (!in)
    {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
}

std::vector<int> parse_cpu_list(const std::string& text)
{
    std::vector<int> cpus;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item = trim(item);
        if (item.empty())
        {
            continue;
        }

        const size_t dash = item.find('-');
        try
        {
            if (dash == std::string::npos)
            {
                cpus.push_back(std::stoi(item));
            }
            else
            {
                const int first = std::stoi(item.substr(0, dash));
                const int last = std::stoi(item.substr(dash + 1));
                for (int cpu = first; cpu <= last; ++cpu)
                {
                    cpus.push_back(cpu);
                }
            }
        }
        catch (const std::exception&)
        {
        }
    }
    return cpus;
}

std::string left_pad(std::string text, size_t width)
{
    if (text.size() >= width)
    {
        return text;
    }
    return std::string(width - text.size(), '0') + text;
}

std::string normalize_pci_bus_id(const std::string& raw)
{
    std::string text = lower_copy(trim(raw));
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ':'))
    {
        parts.push_back(part);
    }

    std::string domain = "0000";
    std::string bus;
    std::string device;
    if (parts.size() == 3)
    {
        domain = parts[0];
        bus = parts[1];
        device = parts[2];
    }
    else if (parts.size() == 2)
    {
        bus = parts[0];
        device = parts[1];
    }
    else
    {
        return text;
    }

    if (domain.size() > 4)
    {
        domain = domain.substr(domain.size() - 4);
    }
    domain = left_pad(domain, 4);
    bus = left_pad(bus, 2);
    return domain + ":" + bus + ":" + device;
}

#if defined(__linux__)
long set_memory_policy(int mode, const unsigned long* nodemask, unsigned long maxnode)
{
    return syscall(SYS_set_mempolicy, mode, nodemask, maxnode);
}

std::vector<unsigned long> make_node_mask(int numa_node, unsigned long* maxnode)
{
    constexpr int kBitsPerWord = static_cast<int>(8 * sizeof(unsigned long));
    const size_t word_count = static_cast<size_t>(numa_node / kBitsPerWord) + 1;
    std::vector<unsigned long> mask(word_count, 0);
    mask[static_cast<size_t>(numa_node / kBitsPerWord)] =
        1UL << static_cast<unsigned>(numa_node % kBitsPerWord);
    *maxnode = static_cast<unsigned long>(numa_node + 1);
    return mask;
}
#endif

}  // namespace

std::vector<int> current_allowed_cpus()
{
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (CPU_ISSET(cpu, &mask))
            {
                cpus.push_back(cpu);
            }
        }
    }
#endif
    return cpus;
}

int numa_node_for_cpu(int cpu)
{
#if defined(__linux__)
    static const std::vector<int> cpu_to_node = []()
    {
        std::vector<int> map(CPU_SETSIZE, -1);
        const std::filesystem::path node_root("/sys/devices/system/node");
        std::error_code ec;
        if (!std::filesystem::exists(node_root, ec) || ec)
        {
            return map;
        }

        for (const auto& entry : std::filesystem::directory_iterator(node_root, ec))
        {
            if (ec)
            {
                break;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("node", 0) != 0)
            {
                continue;
            }

            int node = -1;
            try
            {
                node = std::stoi(name.substr(4));
            }
            catch (const std::exception&)
            {
                continue;
            }

            std::string cpulist;
            if (!read_small_text_file(entry.path() / "cpulist", &cpulist))
            {
                continue;
            }
            for (int listed_cpu : parse_cpu_list(cpulist))
            {
                if (listed_cpu >= 0 && listed_cpu < CPU_SETSIZE)
                {
                    map[static_cast<size_t>(listed_cpu)] = node;
                }
            }
        }
        return map;
    }();

    if (cpu >= 0 && cpu < static_cast<int>(cpu_to_node.size()))
    {
        return cpu_to_node[static_cast<size_t>(cpu)];
    }
#else
    (void)cpu;
#endif
    return -1;
}

int current_numa_node()
{
#if defined(__linux__)
    return numa_node_for_cpu(sched_getcpu());
#else
    return -1;
#endif
}

std::vector<int> allowed_numa_nodes()
{
    std::set<int> nodes;
    for (int cpu : current_allowed_cpus())
    {
        const int node = numa_node_for_cpu(cpu);
        if (node >= 0)
        {
            nodes.insert(node);
        }
    }
    return std::vector<int>(nodes.begin(), nodes.end());
}

std::vector<int> cpus_for_numa_node(int numa_node)
{
    std::vector<int> cpus;
    const std::vector<int> allowed = current_allowed_cpus();
    if (numa_node < 0)
    {
        return allowed;
    }
    for (int cpu : allowed)
    {
        if (numa_node_for_cpu(cpu) == numa_node)
        {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

int choose_cpu_for_numa_node(int numa_node, int hint)
{
    const std::vector<int> cpus = cpus_for_numa_node(numa_node);
    if (cpus.empty())
    {
        return -1;
    }
    const size_t offset =
        static_cast<size_t>(std::max(0, hint)) % static_cast<size_t>(cpus.size());
    return cpus[offset];
}

int numa_node_for_pci_bus_id(const std::string& pci_bus_id)
{
#if defined(__linux__)
    const std::filesystem::path path =
        std::filesystem::path("/sys/bus/pci/devices") /
        normalize_pci_bus_id(pci_bus_id) /
        "numa_node";
    std::string text;
    if (!read_small_text_file(path, &text))
    {
        return -1;
    }
    try
    {
        const int node = std::stoi(trim(text));
        return node >= 0 ? node : -1;
    }
    catch (const std::exception&)
    {
        return -1;
    }
#else
    (void)pci_bus_id;
    return -1;
#endif
}

std::vector<int> parse_numa_node_list(const std::string& raw, std::string* error)
{
    std::string text = lower_copy(trim(raw));
    if (text.empty() || text == "auto" || text == "all")
    {
        if (error != nullptr)
        {
            error->clear();
        }
        return allowed_numa_nodes();
    }
    if (text == "none" || text == "off" || text == "false")
    {
        if (error != nullptr)
        {
            error->clear();
        }
        return {};
    }

    std::set<int> unique_nodes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item = trim(item);
        if (item.empty())
        {
            continue;
        }
        try
        {
            const int node = std::stoi(item);
            if (node < 0)
            {
                if (error != nullptr)
                {
                    *error = "NUMA node list cannot contain negative values";
                }
                return {};
            }
            unique_nodes.insert(node);
        }
        catch (const std::exception&)
        {
            if (error != nullptr)
            {
                *error = "NUMA node list must be auto, none, or comma-separated integers";
            }
            return {};
        }
    }
    if (error != nullptr)
    {
        error->clear();
    }
    return std::vector<int>(unique_nodes.begin(), unique_nodes.end());
}

bool pin_current_thread_to_cpu(int cpu)
{
#if defined(__linux__)
    if (cpu < 0)
    {
        return false;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0;
#else
    (void)cpu;
    return false;
#endif
}

bool set_preferred_numa_memory_policy_for_current_thread(int numa_node)
{
#if defined(__linux__)
    if (numa_node < 0)
    {
        return false;
    }
    unsigned long maxnode = 0;
    const std::vector<unsigned long> mask = make_node_mask(numa_node, &maxnode);
    return set_memory_policy(MPOL_PREFERRED, mask.data(), maxnode) == 0;
#else
    (void)numa_node;
    return false;
#endif
}

bool set_bind_numa_memory_policy_for_current_thread(int numa_node)
{
#if defined(__linux__)
    if (numa_node < 0)
    {
        return false;
    }
    unsigned long maxnode = 0;
    const std::vector<unsigned long> mask = make_node_mask(numa_node, &maxnode);
    return set_memory_policy(MPOL_BIND, mask.data(), maxnode) == 0;
#else
    (void)numa_node;
    return false;
#endif
}

bool reset_numa_memory_policy_for_current_thread()
{
#if defined(__linux__)
    return set_memory_policy(MPOL_DEFAULT, nullptr, 0) == 0;
#else
    return false;
#endif
}

bool numa_memory_policy_from_string(
    const std::string& raw,
    NumaMemoryPolicy* policy,
    std::string* error)
{
    if (policy == nullptr)
    {
        if (error != nullptr)
        {
            *error = "policy output pointer is null";
        }
        return false;
    }

    const std::string text = lower_copy(trim(raw));
    if (text.empty() || text == "off" || text == "none" || text == "false" ||
        text == "0")
    {
        *policy = NumaMemoryPolicy::Off;
    }
    else if (text == "preferred" || text == "prefer")
    {
        *policy = NumaMemoryPolicy::Preferred;
    }
    else if (text == "bind" || text == "membind" || text == "pin")
    {
        *policy = NumaMemoryPolicy::Bind;
    }
    else
    {
        if (error != nullptr)
        {
            *error = "NUMA memory policy must be off, preferred, or bind";
        }
        return false;
    }

    if (error != nullptr)
    {
        error->clear();
    }
    return true;
}

bool numa_memory_policy_from_env(
    const char* env_name,
    NumaMemoryPolicy default_policy,
    NumaMemoryPolicy* policy,
    std::string* error)
{
    if (policy == nullptr)
    {
        if (error != nullptr)
        {
            *error = "policy output pointer is null";
        }
        return false;
    }
    if (env_name == nullptr || env_name[0] == '\0')
    {
        *policy = default_policy;
        if (error != nullptr)
        {
            error->clear();
        }
        return true;
    }

    const char* raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0')
    {
        *policy = default_policy;
        if (error != nullptr)
        {
            error->clear();
        }
        return true;
    }
    return numa_memory_policy_from_string(raw, policy, error);
}

const char* numa_memory_policy_name(NumaMemoryPolicy policy)
{
    switch (policy)
    {
    case NumaMemoryPolicy::Off:
        return "off";
    case NumaMemoryPolicy::Preferred:
        return "preferred";
    case NumaMemoryPolicy::Bind:
        return "bind";
    }
    return "unknown";
}

ScopedCpuAffinity::ScopedCpuAffinity(int cpu)
{
#if defined(__linux__)
    if (cpu < 0)
    {
        return;
    }
    old_mask_storage_.resize(sizeof(cpu_set_t));
    auto* old_mask = reinterpret_cast<cpu_set_t*>(old_mask_storage_.data());
    CPU_ZERO(old_mask);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), old_mask) != 0)
    {
        old_mask_storage_.clear();
        return;
    }
    if (pin_current_thread_to_cpu(cpu))
    {
        active_ = true;
    }
    else
    {
        old_mask_storage_.clear();
    }
#else
    (void)cpu;
#endif
}

ScopedCpuAffinity::~ScopedCpuAffinity()
{
#if defined(__linux__)
    if (active_ && !old_mask_storage_.empty())
    {
        auto* old_mask = reinterpret_cast<cpu_set_t*>(old_mask_storage_.data());
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), old_mask);
    }
#endif
}

ScopedNumaMemoryPolicy::ScopedNumaMemoryPolicy(
    int numa_node,
    NumaMemoryPolicy policy)
{
    switch (policy)
    {
    case NumaMemoryPolicy::Off:
        return;
    case NumaMemoryPolicy::Preferred:
        active_ = set_preferred_numa_memory_policy_for_current_thread(numa_node);
        return;
    case NumaMemoryPolicy::Bind:
        active_ = set_bind_numa_memory_policy_for_current_thread(numa_node);
        return;
    }
}

ScopedNumaMemoryPolicy::~ScopedNumaMemoryPolicy()
{
    if (active_)
    {
        (void)reset_numa_memory_policy_for_current_thread();
    }
}

}  // namespace Chimera
