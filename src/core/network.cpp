#include "network.h"

namespace synaption {

    void Network::add_layer(size_t in_features, size_t out_features) {
        layers_.emplace_back(in_features, out_features);
    }

    Tensor Network::forward(const Tensor& input) {
        Tensor x = input;
        for (auto& layer : layers_)
            x = layer.forward(x);
        return x;
    }

} // namespace synaption