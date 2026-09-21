if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import liouville
import liouville.functions as F
from liouville import optimizers
from liouville.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 10

x, t = liouville.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

key = jax.random.key(1984)
for epoch in range(max_epoch):
    key, subkey = jax.random.split(key)
    index = jax.random.permutation(subkey, data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = jnp.meshgrid(jnp.arange(x_min, x_max, h), jnp.arange(y_min, y_max, h))
X = jnp.c_[xx.ravel(), yy.ravel()]

with liouville.no_grad():
    score = model(X)
predict_cls = jnp.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
plt.show()