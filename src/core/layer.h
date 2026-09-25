#pragma once
#include "tensor.h"

namespace synaption {

    class Layer {
    public:
        Layer(size_t in_features, size_t out_features);

        Tensor forward(const Tensor& input);

    private:
        size_t in_features_, out_features_;
        Tensor weights_;
        Tensor bias_;
    };

} // namespace synaption