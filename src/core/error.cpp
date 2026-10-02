#include "error.h"
#include <iostream>
#include <stdexcept>

namespace synaption {

void throw_invalid_argument(const std::string& msg) {
    std::cerr << "Error: " << msg << std::endl;
    throw std::invalid_argument(msg);
}

void throw_runtime_error(const std::string& msg) {
    std::cerr << "Error: " << msg << std::endl;
    throw std::runtime_error(msg);
}

} // namespace synaption
