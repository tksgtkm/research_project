if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

import liouville
import liouville.functions as F
from liouville import optimizers
from liouville import DataLoader
from liouville.models import MLP

max_epoch = 30
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = liouville.datasets.MNIST(train=True)
test_set = liouville.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

import numpy as np
import matplotlib.pyplot as plt

# ---- 1. 中身の確認 ----
x_all, t_all = train_set.arrays()
print('x_all:', x_all.shape, x_all.dtype, float(x_all.min()), '-', float(x_all.max()))
print('t_all:', t_all.shape, t_all.dtype)
print('label dist:', np.bincount(np.asarray(t_all)))

x_b, t_b = train_loader.next()
print('batch x:', type(x_b), x_b.shape, x_b.dtype)
print('batch t:', type(t_b), t_b.shape, t_b.dtype)
train_loader.reset()

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for ax, img, lbl in zip(axes, x_b[:8], t_b[:8]):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap='gray')
    ax.set_title(int(lbl)); ax.axis('off')
plt.show()

# ---- 2. 一括前処理が従来のtransform経路と一致するか ----
for i in [0, 1, 12345, len(train_set) - 1]:
    x_i, t_i = train_set[i]
    assert jnp.allclose(x_all[i], x_i), f'x mismatch at {i}'
    assert int(t_all[i]) == int(t_i), f't mismatch at {i}'
print('arrays() == transform経路: OK')

# ---- 3. シャッフルの挙動 ----
def first_labels(loader, n_epochs=3):
    out = []
    for _ in range(n_epochs):
        _, t = loader.next()
        out.append(np.asarray(t[:10]))
        loader.reset()          # 次のエポックへ(キーが1つ進む)
    return out

a = first_labels(DataLoader(train_set, batch_size, seed=42))
b = first_labels(DataLoader(train_set, batch_size, seed=42))
c = first_labels(DataLoader(train_set, batch_size, seed=0))

print('エポックごとに順序が変わる:', not np.array_equal(a[0], a[1]))
print('同じseedなら完全に再現   :', all(np.array_equal(p, q) for p, q in zip(a, b)))
print('seedが違えば別の順序     :', not np.array_equal(a[0], c[0]))

_, t_test = DataLoader(test_set, batch_size, shuffle=False).next()
print('テストは先頭から順番    :', np.array_equal(np.asarray(t_test), np.asarray(test_set.label[:batch_size])))

train_set.show()