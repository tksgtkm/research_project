if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from liouville import Variable
import liouville.functions as F
import liouville.layers as L

key = jax.random.PRNGKey(0)
key, kx, kn, k1, k2 = jax.random.split(key, 5)

x = jax.random.uniform(kx, (100, 1))
y = jnp.sin(2 * jnp.pi * x) + jax.random.uniform(kn, (100, 1))

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)