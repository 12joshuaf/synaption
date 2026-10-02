#include "layer.h"
#include "error.h"
#include <random>
#include <cmath>
#include <string>

namespace synaption {

Layer::Layer(size_t in_features, size_t out_features, Activation act)
    : in_features_(in_features), out_features_(out_features), activation_(act) {

    if (in_features == 0 || out_features == 0) {
        throw_invalid_argument(
            "Layer: in_features and out_features must both be > 0 (got in_features=" +
            std::to_string(in_features) + ", out_features=" + std::to_string(out_features) + ")");
    }

    weights_ = Tensor({in_features, out_features});
    bias_ = Tensor({out_features});

    float limit = std::sqrt(6.0f / float(in_features + out_features));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& w : weights_.data_) w = dist(gen);
}

Tensor Layer::forward(const Tensor& input) {
    if (input.size() != in_features_) {
        throw_invalid_argument(
            "Layer::forward: input size (" + std::to_string(input.size()) +
            ") does not match this layer's in_features (" + std::to_string(in_features_) + ")");
    }

    last_input_ = input;
    Tensor out({out_features_});
    pre_activation_.assign(out_features_, 0.0f);

    for (size_t o = 0; o < out_features_; ++o) {
        float sum = bias_.at(o);
        for (size_t i = 0; i < in_features_; ++i)
            sum += input.at(i) * weights_.at(i * out_features_ + o);
        pre_activation_[o] = sum;
        out.at(o) = sum;
    }
    apply_activation(activation_, out.data_);
    forward_called_ = true;
    return out;
}

Tensor Layer::backward(const Tensor& grad_output) {
    if (!forward_called_) {
        throw_runtime_error(
            "Layer::backward: called before forward() — there is no cached input to backpropagate through");
    }
    if (grad_output.size() != out_features_) {
        throw_invalid_argument(
            "Layer::backward: grad_output size (" + std::to_string(grad_output.size()) +
            ") does not match this layer's out_features (" + std::to_string(out_features_) + ")");
    }

    std::vector<float> grad_z = grad_output.data_;
    activation_backward(activation_, pre_activation_, grad_z);

    Tensor grad_input({in_features_});

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
