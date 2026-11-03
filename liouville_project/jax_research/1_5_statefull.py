"""
実際のプログラムには状態がある。
トレーニングステップ間で変化するモデルパラメータ、オプティマイザのモーメンタム
実行統計、カウンターなど。

jaxが状態を持つ計算をどう表現するか？

1: 純粋関数を通して状態を渡す：状態は引数として入力され、戻り値として出力される。
   jaxのエコシステム全体で採用されている状態の処理方式
2: Refs：明示的に変更可能な配列
"""

import jax
import jax.numpy as jnp

# 追跡不可能な状態の問題
class Counter:

    def __init__(self):
        self.n = 0

    def count(self) -> int:
        self.n += 1
        return self.n

counter = Counter()

for _ in range(3):
    print(counter.count())

print(jax.jit(counter.count).trace().jaxpr)

# 純粋関数を通して状態をスレッド化する

CounterState = int

class CounterV2:

    def count(self, n: CounterState) -> tuple[int, CounterState]:
        return n + 1, n + 1

    def reset(self) -> CounterState:
        return 0

counter = CounterV2()
state = counter.reset()

for _ in range(3):
    value, state = counter.count(state)
    print(value)

print(jax.jit(counter.count).trace(0).jaxpr)

def update(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, batch)
    new_params, new_opt_state = optimizer_step(params, grads, opt_state)
    return new_params, new_opt_state

def loss_fn(params, batch):
    x, y = batch
    return jnp.mean((params['w'] * x + params['b'] - y) ** 2)

def optimizer_step(params, grads, opt_state, lr=0.1, decay=0.9):
    new_opt_state = jax.tree.map(lambda m, g: decay * m + g, opt_state, grads)
    new_params = jax.tree.map(lambda p, m: p - lr * m, params, new_opt_state)
    return new_params, new_opt_state

params = {'w': jnp.float32(1.0), 'b': jnp.float32(0.0)}
opt_state = jax.tree.map(jnp.zeros_like, params)
batch = (jnp.array([1.0, 2.0, 3.0]), jnp.array([3.0, 5.0, 7.0]))

for step in range(100):
    params, opt_state = update(params, opt_state, batch)

print(jax.tree.map(lambda x: round(float(x), 2), params))