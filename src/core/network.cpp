// network.cpp
#include "network.h"

namespace synaption {

    void Network::add_layer(size_t in_features, size_t out_features, Activation act) {
        layers_.emplace_back(in_features, out_features, act);
    }

    Tensor Network::forward(const Tensor& input) {
        Tensor x = input;
        for (auto& layer : layers_) x = layer.forward(x);
        return x;
    }

    void Network::backward(const Tensor& grad_output) {
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