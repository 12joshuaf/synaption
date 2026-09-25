#pragma once
#include <vector>
#include <cstddef>

namespace synaption {

    class Tensor {
    public:
        Tensor() = default;
        Tensor(std::vector<size_t> shape);

        float& at(size_t i);
        float  at(size_t i) const;
        const std::vector<size_t>& shape() const { return shape_; }
        size_t size() const { return data_.size(); }

        std::vector<float> data_;
        std::vector<size_t> shape_;
    };

} // namespace synaption