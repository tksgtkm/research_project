if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
from liouville import Variable

x = Variable(jnp.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)