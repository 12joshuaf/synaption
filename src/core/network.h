#pragma once
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include <vector>

namespace synaption {

    class Network {
    public:
        void add_layer(size_t in_features, size_t out_features, Activation act = Activation::ReLU);

        Tensor forward(const Tensor& input);
        void   backward(const Tensor& grad_output);
        std::vector<Tensor*> parameters();

        // Trains on one epoch pass over inputs/targets (aligned vectors). Returns average loss.
        float train_epoch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets,
            Loss loss_type, SGD& optimizer);

    private:
        std::vector<Layer> layers_;
    };

} // namespace synaption