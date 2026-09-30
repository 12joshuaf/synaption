#pragma once
#include "tensor.h"
#include "activation.h"

namespace synaption {

    class Layer {
    public:
        Layer(size_t in_features, size_t out_features, Activation act = Activation::ReLU);

        Tensor forward(const Tensor& input);
        Tensor backward(const Tensor& grad_output); // returns grad w.r.t. input

        Tensor& weights() { return weights_; }
        Tensor& bias() { return bias_; }

    private:
        size_t in_features_, out_features_;
        Activation activation_;
        Tensor weights_;   // [in_features * out_features]
        Tensor bias_;      // [out_features]

        // cached for backward
        Tensor last_input_;
        std::vector<float> pre_activation_;
    };

} // namespace synaption