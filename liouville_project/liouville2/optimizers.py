import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util as tu

__all__ = [
    "Optimizer", "SGD", "MomentumSGD", "Adam", "WeightDecay", "ClipGrad",
    "ReversibleSGD", "RevState",
]

class Optimizer:

    def __init__(self):
        self.hooks = []

    def add_hook(self, f):
        """f(grads, params) -> grads"""
        self.hooks.append(f)
        return self

    def init(self, params):
        return ()

    def update(self, params, grads, state):
        for f in self.hooks:
            grads = f(grads, params)
        return self.update_params(params, grads, state)

    def update_params(self, params, grads, state):
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_params(self, params, grads, state):
        params = tu.tree_map(lambda p, g: p - self.lr * g, params, grads)
        return params, state


class MomentumSGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum

    def init(self, params):
        return tu.tree_map(jnp.zeros_like, params)

    def update_params(self, params, grads, vs):
        vs = tu.tree_map(lambda v, g: self.momentum * v - self.lr * g, vs, grads)
        params = tu.tree_map(jnp.add, params, vs)
        return params, vs


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.alpha, self.beta1, self.beta2, self.eps = alpha, beta1, beta2, eps

    def init(self, params):
        zeros = tu.tree_map(jnp.zeros_like, params)
        return (jnp.zeros((), jnp.int32), zeros, zeros)

    def update_params(self, params, grads, state):
        t, ms, vs = state
        t = t + 1
        b1, b2 = self.beta1, self.beta2
        ms = tu.tree_map(lambda m, g: b1 * m + (1 - b1) * g, ms, grads)
        vs = tu.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, vs, grads)
        lr = self.alpha * jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        params = tu.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + self.eps), params, ms, vs)
        return params, (t, ms, vs)


class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, grads, params):
        return tu.tree_map(lambda g, p: g + self.rate * p, grads, params)


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, grads, params):
        norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in tu.tree_leaves(grads)))
        rate = jnp.minimum(1.0, self.max_norm / (norm + 1e-6))
        return tu.tree_map(lambda g: g * rate, grads)

class RevState(NamedTuple):
    w: object   # パラメータ (int64 固定小数点の pytree)
    v: object   # 速度      (同上)
    B: object   # 情報バッファの下位部分 (int64, 非負)


def _mul_gamma(v, B, n, d):
    B = B * d + jnp.mod(v, d)
    v = jnp.floor_divide(v, d) * n
    r = jnp.mod(B, n)
    B = jnp.floor_divide(B, n)
    return v + r, B


def _div_gamma(v, B, n, d):
    """_mul_gamma の逆"""
    r = jnp.mod(v, n)
    v = jnp.floor_divide(v - r, n)
    B = B * n + r
    v = v * d + jnp.mod(B, d)
    B = jnp.floor_divide(B, d)
    return v, B


class ReversibleSGD:

    SPILL_BITS = 32

    def __init__(self, loss_fn, lr=0.1, gamma=(9, 10), frac_bits=32,
                 batch_fn=None, compute_dtype=jnp.float64):
        if not jax.config.jax_enable_x64:  # pyright: ignore[reportAttributeAccessIssue]
            raise RuntimeError("ReversibleSGD には int64 が必要です。先に liouville2.enable_x64() を呼んでください")
        n, d = gamma
        if not (0 < n < d):
            raise ValueError("gamma=(n, d) は 0 < n < d を満たす整数の組")
        self.n, self.d = n, d
        self.lr = lr
        self.frac_bits = frac_bits
        self.scale = float(2 ** frac_bits)
        self.compute_dtype = compute_dtype
        self.spill_every = max(1, int(30 / math.log2(d / n)))

        self._stack = []            # 退避したバッファ
        self._log = []              # (区間のステップ数, 退避したか, 区間開始時の since)
        self._since = 0             # 前回の退避からのステップ数

        batch_fn = batch_fn or (lambda data, x: x)
        grad_fn = jax.grad(loss_fn)
        one_minus_gamma = 1.0 - n / d

        def qgrad(w, batch):
            params = tu.tree_map(self._decode, w)
            g = grad_fn(params, batch)
            return tu.tree_map(lambda x: self._encode(one_minus_gamma * x), g)

        def dw(v):
            return tu.tree_map(lambda x: jnp.round(lr * x.astype(jnp.float64)).astype(jnp.int64), v)

        def split(pairs):
            is_pair = lambda t: isinstance(t, tuple)
            return (tu.tree_map(lambda t: t[0], pairs, is_leaf=is_pair),
                    tu.tree_map(lambda t: t[1], pairs, is_leaf=is_pair))

        def fwd_step(carry, x):
            state, data = carry
            w, v, B = state
            g = qgrad(w, batch_fn(data, x))
            v, B = split(tu.tree_map(lambda v_, B_: _mul_gamma(v_, B_, n, d), v, B))
            v = tu.tree_map(jnp.subtract, v, g)
            w = tu.tree_map(jnp.add, w, dw(v))
            return (RevState(w, v, B), data), None

        def bwd_step(carry, x):
            state, data = carry
            w, v, B = state
            w = tu.tree_map(jnp.subtract, w, dw(v))
            g = qgrad(w, batch_fn(data, x))
            v = tu.tree_map(jnp.add, v, g)
            v, B = split(tu.tree_map(lambda v_, B_: _div_gamma(v_, B_, n, d), v, B))
            return (RevState(w, v, B), data), None

        @jax.jit
        def run_fwd(state, xs, data):
            return jax.lax.scan(fwd_step, (state, data), xs)[0][0]

        @jax.jit
        def run_bwd(state, xs, data):
            return jax.lax.scan(bwd_step, (state, data), xs, reverse=True)[0][0]

        self._run_fwd, self._run_bwd = run_fwd, run_bwd

    def _encode(self, x):
        return jnp.round(x.astype(jnp.float64) * self.scale).astype(jnp.int64)

    def _decode(self, q):
        return (q.astype(jnp.float64) / self.scale).astype(self.compute_dtype)

    def init(self, params):
        self._stack.clear()
        self._log.clear()
        self._since = 0
        w = tu.tree_map(self._encode, params)
        zeros = tu.tree_map(jnp.zeros_like, w)
        return RevState(w, zeros, zeros)

    def params(self, state, dtype=jnp.float32):
        return tu.tree_map(lambda q: (q.astype(jnp.float64) / self.scale).astype(dtype), state.w)

    def run(self, state, xs, data=None):
        T = tu.tree_leaves(xs)[0].shape[0]
        s = 0
        while s < T:
            seg = min(T - s, self.spill_every - self._since)
            chunk = tu.tree_map(lambda a: a[s:s + seg], xs)
            state = self._run_fwd(state, chunk, data)
            since_before = self._since
            self._since += seg
            spilled = self._since == self.spill_every
            if spilled:
                state = self._spill(state)
                self._since = 0
            self._log.append((seg, spilled, since_before))
            s += seg
        return state

    def reverse(self, state, xs, data=None):
        T = tu.tree_leaves(xs)[0].shape[0]
        e = T
        while e > 0:
            if not self._log:
                raise RuntimeError("戻すステップがログに残っていません")
            seg, spilled, since_before = self._log.pop()
            if seg > e:
                raise RuntimeError("xs の長さが run の区切りと一致しません")
            if spilled:
                state = self._unspill(state)
            chunk = tu.tree_map(lambda a: a[e - seg:e], xs)
            state = self._run_bwd(state, chunk, data)
            self._since = since_before
            e -= seg
        return state

    def _spill(self, state):
        mask = (1 << self.SPILL_BITS) - 1
        self._stack.append(tu.tree_map(lambda b: jnp.bitwise_and(b, mask), state.B))
        B = tu.tree_map(lambda b: jnp.right_shift(b, self.SPILL_BITS), state.B)
        return state._replace(B=B)

    def _unspill(self, state):
        low = self._stack.pop()
        B = tu.tree_map(lambda b, l: jnp.bitwise_or(jnp.left_shift(b, self.SPILL_BITS), l), state.B, low)
        return state._replace(B=B)

    def buffer_bits(self, state):
        leaves = tu.tree_leaves(state.B)
        n_param = sum(b.size for b in leaves)
        live = sum(float(jnp.sum(jnp.ceil(jnp.log2(b.astype(jnp.float64) + 1)))) for b in leaves)
        return live + len(self._stack) * self.SPILL_BITS * n_param

    def theoretical_bits(self, steps, n_param):
        return steps * n_param * math.log2(self.d / self.n)