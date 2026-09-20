if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from liouville import Variable
import  liouville.functions as F

key = jax.random.PRNGKey(0)
key, kx, kn, k1, k2 = jax.random.split(key, 5)

x = jax.random.uniform(kx, (100, 1))
y = jnp.sin(2 * jnp.pi * x) + jax.random.uniform(kn, (100, 1))

I, H, O = 1, 10, 1
W1 = Variable(0.01 * jax.random.normal(k1, (I, H)))
b1 = Variable(jnp.zeros(H))
W2 = Variable(0.01 * jax.random.normal(k2, (H, O)))
b2 = Variable(jnp.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = jnp.arange(0, 1, .01)[:, jnp.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()