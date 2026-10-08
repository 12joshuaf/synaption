#pragma once
#include <vector>

namespace synaption {

enum class Activation { Identity, ReLU, Sigmoid, Tanh, Swish, GELU, ELU };

// Scalar versions, used directly inside the per-neuron parallel compute loop.
float activation_scalar(Activation act, float z);
float activation_derivative_scalar(Activation act, float z);

// Vector convenience wrappers (elementwise; kept for any single-threaded/test use).
void apply_activation(Activation act, std::vector<float>& x);
void activation_backward(Activation act, const std::vector<float>& pre_act,
                          std::vector<float>& grad);

} // namespace synaption
