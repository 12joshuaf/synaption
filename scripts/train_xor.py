import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from synaption import Tensor, Network

net = Network()
net.add_layer(2, 4)
net.add_layer(4, 1)

x = Tensor([2])
x[0], x[1] = 1.0, 0.0

y = net.forward(x)
print("output:", [y[i] for i in range(y.size())])