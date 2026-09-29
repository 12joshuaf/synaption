# Synaption

## An Educational Machine Learning Python Module Built with C++

Many programs teaching computer science to high school age children rely on complicated Python modules such as **TensorFlow, PyTorch, Scikit-learn, JAX**, and others.

These libraries can be quite difficult to understand for a newcomer to Python. Often times, students will be relegated to copying code without understanding its true meaning.

***Synaption*** is a less granular alternative that allows for simpler, hands-on learning of machine learning concepts.

---

## Example Python Synaption Code

```python
from synaption import Tensor, Network

net = Network()

net.add_layer(2, 4)
net.add_layer(4, 1)

x = Tensor([2])

x[0] = 1.0
x[1] = 0.0

y = net.forward(x)

print("output:", [y[i] for i in range(y.size())])
```
