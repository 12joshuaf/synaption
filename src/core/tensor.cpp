#include "tensor.h"
#include <numeric>
#include <functional>

namespace synaption {

    Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
        size_t total = std::accumulate(shape_.begin(), shape_.end(),
            size_t(1), std::multiplies<size_t>());
        data_.assign(total, 0.0f);
    }

    float& Tensor::at(size_t i) { return data_[i]; }
    float  Tensor::at(size_t i) const { return data_[i]; }

} // namespace synaption