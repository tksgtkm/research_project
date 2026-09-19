if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math

import jax.numpy as jnp

from liouville import Variable, Function
from liouville.utils import plot_dot_graph

class Sin(Function):

    def forward(self, x):
        y = jnp.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * jnp.cos(x)
        return gx

def sin(x):
    return Sin()(x)

x = Variable(jnp.array(jnp.pi / 4))
y = sin(x)
y.backward()
print('--- original sin ---')
print(y.data)
print(x.grad)

# テイラー展開の実装
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x = Variable(jnp.array(jnp.pi / 4))
y = my_sin(x)
# y = my_sin(x, threshold=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin.png')