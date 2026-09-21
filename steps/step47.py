if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
jax.random.key(0)
import liouville.functions as F
from liouville import Variable, as_variable
from liouville.models import MLP

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y

model = MLP((10, 3))

x = Variable(jnp.array([[0.2, -0.4]]))
y = model(x)
p = softmax1d(y)
print(y)
print(p)

x = jnp.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = jnp.array([2, 0, 1, 0])

y = model(x)
p = softmax_simple(y)
print(y)
print(p)

loss = F.softmax_cross_entropy(y, t)
loss.backward()
print(loss)