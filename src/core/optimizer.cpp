#include "optimizer.h"

namespace synaption {

    void SGD::step(std::vector<Tensor*>& params) {
        for (auto* p : params)
            for (size_t i = 0; i < p->size(); ++i)
                p->data_[i] -= lr_ * p->grad_[i];
    }

    void SGD::zero_grad(std::vector<Tensor*>& params) {
        for (auto* p : params) p->zero_grad();
    }

} // namespace synaption