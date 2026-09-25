#include "layer.h"
#include <random>

namespace synaption {

    Layer::Layer(size_t in_features, size_t out_features)
        : in_features_(in_features), out_features_(out_features),
        weights_({ in_features, out_features }), bias_({ out_features }) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto& w : weights_.data_) w = dist(gen);
    }

    Tensor Layer::forward(const Tensor& input) {
        Tensor out({ out_features_ });
        for (size_t o = 0; o < out_features_; ++o) {
            float sum = bias_.at(o);
            for (size_t i = 0; i < in_features_; ++i)
                sum += input.at(i) * weights_.at(i * out_features_ + o);
            out.at(o) = sum;
        }
        return out;
    }

} // namespace synaption