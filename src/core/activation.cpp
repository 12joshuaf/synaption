#include "activation.h"
#include <cmath>
#include <algorithm>

namespace synaption {

static float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float activation_scalar(Activation act, float z) {
    switch (act) {
        case Activation::Identity: return z;
        case Activation::ReLU:     return std::max(0.0f, z);
        case Activation::Sigmoid:  return sigmoid_f(z);
        case Activation::Tanh:     return std::tanh(z);
        case Activation::Swish:    return z * sigmoid_f(z);
        case Activation::GELU: {
            float t = std::tanh(0.7978845608f * (z + 0.044715f * z * z * z));
            return 0.5f * z * (1.0f + t);
        }
        case Activation::ELU:      return z >= 0.0f ? z : (std::exp(z) - 1.0f);
    }
    return z;
}

float activation_derivative_scalar(Activation act, float z) {
    switch (act) {
        case Activation::Identity: return 1.0f;
        case Activation::ReLU:     return z > 0.0f ? 1.0f : 0.0f;
        case Activation::Sigmoid:  { float s = sigmoid_f(z); return s * (1.0f - s); }
        case Activation::Tanh:     { float t = std::tanh(z); return 1.0f - t * t; }
        case Activation::Swish:    { float s = sigmoid_f(z); return s + z * s * (1.0f - s); }
        case Activation::GELU: {
            float t = std::tanh(0.7978845608f * (z + 0.044715f * z * z * z));
            float dt = (1.0f - t * t) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * z * z);
            return 0.5f * (1.0f + t) + 0.5f * z * dt;
        }
        case Activation::ELU:      return z >= 0.0f ? 1.0f : std::exp(z);
    }
    return 1.0f;
}

void apply_activation(Activation act, std::vector<float>& x) {
    for (auto& v : x) v = activation_scalar(act, v);
}

void activation_backward(Activation act, const std::vector<float>& pre_act,
                          std::vector<float>& grad) {
    for (size_t i = 0; i < grad.size(); ++i)
        grad[i] *= activation_derivative_scalar(act, pre_act[i]);
}

} // namespace synaption
