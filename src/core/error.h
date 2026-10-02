#pragma once
#include <string>

namespace synaption {

// Prints "Error: <msg>" to stderr, then throws std::invalid_argument(msg).
// Use for bad arguments / invalid configuration (pybind11 maps this to Python ValueError).
[[noreturn]] void throw_invalid_argument(const std::string& msg);

// Prints "Error: <msg>" to stderr, then throws std::runtime_error(msg).
// Use for invalid call order / bad state (pybind11 maps this to Python RuntimeError).
[[noreturn]] void throw_runtime_error(const std::string& msg);

} // namespace synaption
