#pragma once
#include "layer.h"
#include <vector>

namespace synaption {

    class Network {
    public:
        void add_layer(size_t in_features, size_t out_features);
        Tensor forward(const Tensor& input);

    private:
        std::vector<Layer> layers_;
    };

} // namespace synaption