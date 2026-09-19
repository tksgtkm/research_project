if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import liouville.functions as F
from liouville import Variable

key = jax.random.key(0)
key_x, key_noise = jax.random.split(key)

x = jax.random.uniform(key_x, (100, 1))
y = 5 + 2 * x + jax.random.uniform(key_noise, (100, 1))
x, y = Variable(x), Variable(y)

W = Variable(jnp.zeros((1, 1)))
b = Variable(jnp.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()