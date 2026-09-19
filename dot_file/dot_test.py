if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp

from liouville import Variable
from liouville.utils import get_dot_graph

x0 = Variable(jnp.array(1.0))
x1 = Variable(jnp.array(1.0))
y = x0 + x1

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

with open('sample.dot', 'w') as o:
    o.write(txt)

