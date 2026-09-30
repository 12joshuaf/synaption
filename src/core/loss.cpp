#include "loss.h"
#include <cmath>
#include <algorithm>

namespace synaption {

    static constexpr float kEps = 1e-7f;

    float compute_loss(Loss loss_type, const Tensor& pred, const Tensor& target, Tensor& grad_out) {
        size_t n = pred.size();
        grad_out = Tensor(pred.shape());
        float loss = 0.0f;

        switch (loss_type) {
        case Loss::MSE: {
            for (size_t i = 0; i < n; ++i) {
                float diff = pred.at(i) - target.at(i);
                loss += diff * diff;
                grad_out.at(i) = 2.0f * diff / float(n);
            }
            loss /= float(n);
            break;
        }
        case Loss::BCE: { // binary cross-entropy, pred expected in (0,1) e.g. after sigmoid
            for (size_t i = 0; i < n; ++i) {
                float p = std::clamp(pred.at(i), kEps, 1.0f - kEps);
                float y = target.at(i);
                loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
                grad_out.at(i) = (p - y) / (p * (1.0f - p)) / float(n);
            }
            loss /= float(n);
            break;
        }
        case Loss::CCE: { // categorical cross-entropy, pred expected to be a softmax distribution
            for (size_t i = 0; i < n; ++i) {
                float p = std::clamp(pred.at(i), kEps, 1.0f);
                loss += -target.at(i) * std::log(p);
                grad_out.at(i) = -target.at(i) / p / float(n);
            }
            break;
        }
        case Loss::Hinge: { // one-vs-rest style hinge, target in {-1, +1}
            for (size_t i = 0; i < n; ++i) {
                float margin = 1.0f - target.at(i) * pred.at(i);
                loss += std::max(0.0f, margin);
                grad_out.at(i) = margin > 0.0f ? -target.at(i) / float(n) : 0.0f;
            }
            break;
        }
        }
        return loss;
    }

} // namespace synaption