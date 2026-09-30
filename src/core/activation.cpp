#include "activation.h"
#include <cmath>
#include <algorithm>

namespace synaption {

    static float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    void apply_activation(Activation act, std::vector<float>& x) {
        for (auto& v : x) {
            switch (act) {
            case Activation::Identity: break;
            case Activation::ReLU:     v = std::max(0.0f, v); break;
            case Activation::Sigmoid:  v = sigmoid_f(v); break;
            case Activation::Tanh:     v = std::tanh(v); break;
            case Activation::Swish:    v = v * sigmoid_f(v); break;
            case Activation::GELU:     v = 0.5f * v * (1.0f + std::tanh(0.7978845608f *
                (v + 0.044715f * v * v * v))); break;
            case Activation::ELU:      v = v >= 0.0f ? v : (std::exp(v) - 1.0f); break;
            }
        }
    }

    void activation_backward(Activation act, const std::vector<float>& pre_act,
        std::vector<float>& grad) {
        for (size_t i = 0; i < grad.size(); ++i) {
            float z = pre_act[i];
            float d = 1.0f;
            switch (act) {
            case Activation::Identity: d = 1.0f; break;
            case Activation::ReLU:     d = z > 0.0f ? 1.0f : 0.0f; break;
            case Activation::Sigmoid: { float s = sigmoid_f(z); d = s * (1.0f - s); break; }
            case Activation::Tanh: { float t = std::tanh(z); d = 1.0f - t * t; break; }
            case Activation::Swish: { float s = sigmoid_f(z); d = s + z * s * (1.0f - s); break; }
            case Activation::GELU: {
                float t = std::tanh(0.7978845608f * (z + 0.044715f * z * z * z));
                float dt = (1.0f - t * t) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * z * z);
                d = 0.5f * (1.0f + t) + 0.5f * z * dt;
                break;
            }
            case Activation::ELU:      d = z >= 0.0f ? 1.0f : std::exp(z); break;
            }
            grad[i] *= d;
        }
    }


} // namespace synaption