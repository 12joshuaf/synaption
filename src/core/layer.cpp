#include "layer.h"
#include <random>
#include <cmath>

namespace synaption {

    Layer::Layer(size_t in_features, size_t out_features, Activation act)
        : in_features_(in_features), out_features_(out_features), activation_(act),
        weights_({ in_features, out_features }), bias_({ out_features }) {

        // Xavier/Glorot uniform init: U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out))
        float limit = std::sqrt(6.0f / float(in_features + out_features));
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& w : weights_.data_) w = dist(gen);
        // biases start at zero — standard practice
    }

    Tensor Layer::forward(const Tensor& input) {
        last_input_ = input;
        Tensor out({ out_features_ });
        pre_activation_.assign(out_features_, 0.0f);

        for (size_t o = 0; o < out_features_; ++o) {
            float sum = bias_.at(o);
            for (size_t i = 0; i < in_features_; ++i)
                sum += input.at(i) * weights_.at(i * out_features_ + o);
            pre_activation_[o] = sum;
            out.at(o) = sum;
        }
        apply_activation(activation_, out.data_);
        return out;
    }

    Tensor Layer::backward(const Tensor& grad_output) {
        // grad_output: dL/dy for this layer's output
        std::vector<float> grad_z = grad_output.data_;              // will become dL/dz
        activation_backward(activation_, pre_activation_, grad_z);   // dL/dz = dL/dy * dy/dz

        Tensor grad_input({ in_features_ });

        for (size_t o = 0; o < out_features_; ++o) {
            float g = grad_z[o];
            bias_.grad_[o] += g;
            for (size_t i = 0; i < in_features_; ++i) {
                weights_.grad_[i * out_features_ + o] += g * last_input_.at(i);
                grad_input.at(i) += g * weights_.at(i * out_features_ + o);
            }
        }
        return grad_input;
    }

} // namespace synaption