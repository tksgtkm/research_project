if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.numpy as jnp
from jax import random

import liouville.functions as F
from liouville import Variable

key = random.key(0)
randn = random.normal(key, (2, 3))
x = Variable(randn)
randn = random.normal(key, (3, 4))
w = Variable(randn)
y = F.matmul(x, w)
y.backward()

print(x.grad.shape)
print(w.grad.shape)