#pragma once

#include <stdexcept>
#include <string>

inline std::string require_value(int argc, char* argv[], int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++i];
}
