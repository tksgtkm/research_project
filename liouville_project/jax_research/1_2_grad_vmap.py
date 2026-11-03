"""
変換gradとvmapについて

jaxの中核をなすのは関数変換という機能で、これはユーザーが作成した関数を受け取って、
関連する計算を行う新しい関数を返す高階関数として存在する。

jax.grad(): 関数を自動微分として勾配を計算する関数に変換する
jax.vmap(): 単一の関数を自動ベクトル化によってバッチ処理で動作する関数に変換する
他にもjax.jit()があるが、これは別途扱う
"""

import jax
import jax.numpy as jnp

grad_tanh = jax.grad(jnp.tanh)
print(grad_tanh(2.0))

# jax.gradを繰り返し適用することで高階微分を求めることもできる
f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)

print(dfdx(1.0))
print(d2fdx(1.0))
print(d3fdx(1.0))

# パラメータは他の引数、または複数の引数を同時に選択する
# ロジスティック回帰の例

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])

targets = jnp.array([True, True, False, True])
W = jnp.array([0.1, 0.4, -0.3])
b = 0.5

def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))

W_grad = jax.grad(loss, argnums=0)(W, b)
print(f'{W_grad=}')

W_grad = jax.grad(loss)(W, b)
print(f'{W_grad=}')

b_grad = jax.grad(loss, 1)(W, b)
print(f'{b_grad=}')

W_grad, b_grad = jax.grad(loss, (0, 1))(W, b)
print(f'{W_grad=}')
print(f'{b_grad=}')

# 学習時の進捗状況をログに記録するときなど、損失値と勾配の両方が必要となる
# そういうときjaxだと
loss_value, Wb_grad = jax.value_and_grad(loss, (0, 1))(W, b)
print(loss_value)

# 関数によっては微分対象のスカラー値とともに返すべき中間結果を計算する場合がある。
def loss_and_preds(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs)), preds

W_grad, preds = jax.grad(loss_and_preds, has_aux=True)(W, b)
print(preds)

# 続いてvmapを使うと単一の入力関数を入力のバッチ処理に対応できるように変換できる

# 例えば1次元ベクトルの畳込みを計算する関数を考えてみる
x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

print(convolve(x, w))

# この関数をxsとwsのバッチ全体に適用したいとする。
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

# 簡単にはループ処理で対処する。
def manually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[1], ws[1]))
    return jnp.stack(output)

print(manually_batched_convolve(xs, ws))

# この方法では正しく動作するが配列レベルの並列レベルに対応したハードウェアでは
# パフォーマンスが低下する。
# 効率的なバッチ処理にするには、通常すべての操作がバッチ次元で実行されるように
# 関数を手動で書き直すが、vmapを使うともっと簡単にかける。
auto_batch_convolve = jax.vmap(convolve)
print(auto_batch_convolve(xs, ws))

# デフォルトではvmapすべての入力の先頭軸にマッピングされる。
# 引数in_axesとout_axesはこれを上書きする。
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

print(auto_batch_convolve_v2(xst, wst))

# in_axesのNoneのエントリの意味はこの引数をマッピングしないということで、
# 代わりにすべての呼び出しにブロードキャストされる。
# ここでは共有された1つのwに対してxのバッチを畳み込む。
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])
print(batch_convolve_v3(xs, w))

# vmap合成を行う。

def dist(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

def all_pairs(f):
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))

points = jnp.array([[0., 0.], [1., 0.], [0., 2.]])
print(all_pairs(dist)(points, points))

# `vmap` と `grad` を組み合わせると、ほかの方法では表現しにくい問いに答えられる。
# たとえば、ここで扱っているロジスティック回帰の損失の勾配を、バッチ全体で合計するのではなく、
# サンプルごとに個別に求めたい場合です。1サンプル分の損失を書き、それを微分してから、その導関数をベクトル化する。

def example_loss(W, b, x, y):
    pred = predict(W, b, x)
    label_prob = pred * y + (1 - pred) * (1 - y)
    return -jnp.log(label_prob)

per_example_grads = jax.vmap(jax.grad(example_loss, (0, 1)), in_axes=(None, None, 0, 0))
W_grads, b_grads = per_example_grads(W, b, inputs, targets)
print(W_grads)
print(b_grads)