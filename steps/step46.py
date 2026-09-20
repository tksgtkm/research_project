if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp

import liouville.functions as F
from liouville import optimizers
from liouville.models import MLP

key = jax.random.PRNGKey(0)
key, kx, kn, k1, k2 = jax.random.split(key, 5)

x = jax.random.uniform(kx, (100, 1))
y = jnp.sin(2 * jnp.pi * x) + jax.random.uniform(kn, (100, 1))

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
# optimizers = optimizers.SGD(lr).setup(model)
optimizers = optimizers.MomentumSGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizers.update()
    if i % 1000 == 0:
        print(loss)