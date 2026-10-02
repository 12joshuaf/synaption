#include "optimizer.h"
#include "error.h"
#include <string>

namespace synaption {

SGD::SGD(float learning_rate) : lr_(learning_rate) {
    if (!(learning_rate > 0.0f)) {
        throw_invalid_argument(
            "SGD: learning_rate must be > 0 (got " + std::to_string(learning_rate) + ")");
    }
}

void SGD::step(std::vector<Tensor*>& params) {
    for (auto* p : params)
        for (size_t i = 0; i < p->size(); ++i)
            p->data_[i] -= lr_ * p->grad_[i];
}

void SGD::zero_grad(std::vector<Tensor*>& params) {
    for (auto* p : params) p->zero_grad();
}

} // namespace synaption
