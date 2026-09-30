#pragma once
#include "tensor.h"

namespace synaption {

	enum class Loss { MSE, BCE, CCE, Hinge };

	// Computes scalar loss and fills grad_out with dL/dpred (same shape as pred/target)
	float compute_loss(Loss loss_type, const Tensor& pred, const Tensor& target, Tensor& grad_out);

} // namespace synaption