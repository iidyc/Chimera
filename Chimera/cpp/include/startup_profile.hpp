#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifndef CHIMERA_IMPL_NAME
#define CHIMERA_IMPL_NAME "unknown"
#endif

namespace Chimera {

inline bool startup_profile_enabled() {
    const char* env = std::getenv("CHIMERA_STARTUP_PROFILE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

inline const char* startup_profile_impl_name() {
    return CHIMERA_IMPL_NAME;
}

class StartupProfile {
public:
    explicit StartupProfile(const char* scope)
        : enabled_(startup_profile_enabled()),
          scope_(scope),
          last_(Clock::now()) {}

    void mark(const std::string& label) {
        if (!enabled_) {
            return;
        }
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
        std::cout << "[STARTUP][" << startup_profile_impl_name() << "][" << scope_
                  << "] " << label << ": " << ms << " ms" << std::endl;
        last_ = now;
    }

    void note(const std::string& label, const std::string& value) const {
        if (!enabled_) {
            return;
        }
        std::cout << "[STARTUP][" << startup_profile_impl_name() << "][" << scope_
                  << "] " << label << ": " << value << std::endl;
    }

private:
    using Clock = std::chrono::steady_clock;

    bool enabled_;
    const char* scope_;
    Clock::time_point last_;
};

}  // namespace Chimera
