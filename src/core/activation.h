#pragma once
#include <vector>

namespace synaption {

    enum class Activation { Identity, ReLU, Sigmoid, Tanh, Swish, GELU, ELU };

    // Applies activation in place; writes pre-activation values into pre_act (needed for backward)
    void apply_activation(Activation act, std::vector<float>& x);

    // Given pre-activation values z and upstream grad dL/dy, returns dL/dz (in place into grad)
    void activation_backward(Activation act, const std::vector<float>& pre_act,
        std::vector<float>& grad);

} // namespace synaption