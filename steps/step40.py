if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.numpy as jnp
from liouville import Variable

x0 = Variable(jnp.array([1, 2, 3]))
x1 = Variable(jnp.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)