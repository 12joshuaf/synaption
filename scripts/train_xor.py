import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from synaption import Tensor, Network, Activation, Loss, SGD

net = Network()
net.add_layer(2, 4, Activation.Tanh)
net.add_layer(4, 4, Activation.ReLU)
net.add_layer(4, 1, Activation.Sigmoid)

def make_tensor(vals):
    t = Tensor([len(vals)])
    for i, v in enumerate(vals):
        t[i] = v
    return t

xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_targets = [0, 1, 1, 0]

inputs = [make_tensor(x) for x in xor_inputs]
targets = [make_tensor([y]) for y in xor_targets]

optimizer = SGD(0.5)

for epoch in range(2000):
    loss = net.train_epoch(inputs, targets, Loss.BCE, optimizer)
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.4f}")

print("\nfinal predictions:")
for x, y in zip(xor_inputs, xor_targets):
    pred = net.forward(make_tensor(x))
    print(f"  {x} -> {pred[0]:.3f}  (target {y})")