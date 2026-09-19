if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.numpy as jnp
from jax import random

import liouville.functions as F
from liouville import Variable

x = Variable(jnp.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(jnp.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(jnp.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

key = random.key(0)
randn = random.normal(key, (2, 3, 4, 5))
x = Variable(randn)
y = x.sum(keepdims=True)
print(y.shape)