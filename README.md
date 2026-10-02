# Synaption

## An Educational Machine Learning Python Module Built with C++

### Current Version: 0.2.2

---

Many programs teaching computer science to high school age children rely on complicated Python modules such as **TensorFlow, PyTorch, Scikit-learn, JAX**, and others.

These libraries can be quite difficult to understand for a newcomer to Python. Often times, students will be relegated to copying code without understanding its true meaning.

***Synaption*** is a less granular alternative that allows for simpler, hands-on learning of machine learning concepts.

---

## Example Python Synaption Code

```python
#Importing synaption module components
from synaption import Tensor, Network, Activation, Loss, SGD

#creating neural network
net = Network()
net.add_layer(2, 4, Activation.Tanh)
net.add_layer(4, 4, Activation.ReLU)
net.add_layer(4, 1, Activation.Sigmoid)

#function to create tensors for training
def make_tensor(vals):
    t = Tensor([len(vals)])
    for i, v in enumerate(vals):
        t[i] = v
    return t

#XOR values to be converted into tensors for training
xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_targets = [0, 1, 1, 0]

#converting python lists into tensors
inputs = [make_tensor(x) for x in xor_inputs]
targets = [make_tensor([y]) for y in xor_targets]

#creating optimizer, SGD with LR of .5
optimizer = SGD(0.5)

#training loop
for epoch in range(2000):
    loss = net.train_epoch(inputs, targets, Loss.BCE, optimizer)
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.4f}")

#final predictions
print("\nfinal predictions:")
for x, y in zip(xor_inputs, xor_targets):
    pred = net.forward(make_tensor(x))
    print(f"  {x} -> {pred[0]:.3f}  (target {y})")
```

---

### Run Instructions for Alpha Version:

1. Clone Respository
2. Run premake5 in project root
3. Build Solution (Visual Studio Recommended)
4. Run the example python script in the scripts directory (or create your own)











