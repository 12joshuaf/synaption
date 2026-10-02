#include "tensor.h"
#include "error.h"
#include <numeric>
#include <functional>
#include <string>

namespace synaption {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    if (shape_.empty())
        throw_invalid_argument("Tensor: shape must have at least one dimension");
    for (size_t dim : shape_) {
        if (dim == 0)
            throw_invalid_argument("Tensor: shape dimensions must all be > 0");
    }
    size_t total = std::accumulate(shape_.begin(), shape_.end(),
                                    size_t(1), std::multiplies<size_t>());
    data_.assign(total, 0.0f);
    grad_.assign(total, 0.0f);
}

float& Tensor::at(size_t i) {
    if (i >= data_.size())
        throw_invalid_argument("Tensor::at: index " + std::to_string(i) +
                                " out of range (size " + std::to_string(data_.size()) + ")");
    return data_[i];
}

float Tensor::at(size_t i) const {
    if (i >= data_.size())
        throw_invalid_argument("Tensor::at: index " + std::to_string(i) +
                                " out of range (size " + std::to_string(data_.size()) + ")");
    return data_[i];
}

void Tensor::zero_grad() { std::fill(grad_.begin(), grad_.end(), 0.0f); }

} // namespace synaption
