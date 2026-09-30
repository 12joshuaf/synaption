#pragma once
#include "tensor.h"
#include <vector>

namespace synaption {

    class SGD {
    public:
        explicit SGD(float learning_rate) : lr_(learning_rate) {}

        void step(std::vector<Tensor*>& params);
        void zero_grad(std::vector<Tensor*>& params);

    private:
        float lr_;
    };

} // namespace synaption