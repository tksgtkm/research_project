if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp

from liouville import Model
import liouville.layers as L
import liouville.functions as F

key = jax.random.PRNGKey(0)
key, kx, kn, k1, k2 = jax.random.split(key, 5)

x = jax.random.uniform(kx, (100, 1))
y = jnp.sin(2 * jnp.pi * x) + jax.random.uniform(kn, (100, 1))

lr = 0.2
max_iter = 10000
hidden_size = 10

class TwoLayerNet(Model):

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)
model.plot(x)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)