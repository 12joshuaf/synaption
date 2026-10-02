#include "network.h"
#include "error.h"
#include <string>

namespace synaption {

void Network::add_layer(size_t in_features, size_t out_features, Activation act) {
    if (!layers_.empty() && in_features != layers_.back().out_features()) {
        throw_invalid_argument(
            "Network::add_layer: in_features (" + std::to_string(in_features) +
            ") does not match the previous layer's out_features (" +
            std::to_string(layers_.back().out_features()) +
            "). Layers must chain: this layer's in_features must equal the prior layer's out_features.");
    }
    layers_.emplace_back(in_features, out_features, act);
}

Tensor Network::forward(const Tensor& input) {
    if (layers_.empty())
        throw_runtime_error("Network::forward: network has no layers (call add_layer first)");

    if (input.size() != layers_.front().in_features()) {
        throw_invalid_argument(
            "Network::forward: input size (" + std::to_string(input.size()) +
            ") does not match the first layer's in_features (" +
            std::to_string(layers_.front().in_features()) + ")");
    }

    Tensor x = input;
    for (auto& layer : layers_) x = layer.forward(x);
    forward_called_ = true;
    return x;
}

void Network::backward(const Tensor& grad_output) {
    if (layers_.empty())
        throw_runtime_error("Network::backward: network has no layers");
    if (!forward_called_)
        throw_runtime_error("Network::backward: called before forward()");

    if (grad_output.size() != layers_.back().out_features()) {
        throw_invalid_argument(
            "Network::backward: grad_output size (" + std::to_string(grad_output.size()) +
            ") does not match the final layer's out_features (" +
            std::to_string(layers_.back().out_features()) + ")");
    }

    Tensor grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = it->backward(grad);
}

std::vector<Tensor*> Network::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : layers_) {
        params.push_back(&layer.weights());
        params.push_back(&layer.bias());
    }
    return params;
}

float Network::train_epoch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets,
                            Loss loss_type, SGD& optimizer) {
    if (layers_.empty())
        throw_runtime_error("Network::train_epoch: network has no layers");
    if (inputs.empty())
        throw_invalid_argument("Network::train_epoch: inputs is empty");
    if (inputs.size() != targets.size()) {
        throw_invalid_argument(
            "Network::train_epoch: inputs.size() (" + std::to_string(inputs.size()) +
            ") does not match targets.size() (" + std::to_string(targets.size()) + ")");
    }

    auto params = parameters();
    float total_loss = 0.0f;

    for (size_t i = 0; i < inputs.size(); ++i) {
        optimizer.zero_grad(params);

        Tensor pred = forward(inputs[i]);
        Tensor grad_out;
        total_loss += compute_loss(loss_type, pred, targets[i], grad_out);

        backward(grad_out);
        optimizer.step(params);
    }
    return total_loss / float(inputs.size());
}

} // namespace synaption
