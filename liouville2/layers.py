import zlib

import jax
import jax.numpy as jnp

import liouville2.functions as F

__all__ = ["Layer", "Linear"]

class _LazyParams:

    def __init__(self, key):
        self.key = key
        self.value = None
        self.children = {}

    def __getitem__(self, name):
        if name not in self.children:
            sub = jax.random.fold_in(self.key, zlib.crc32(str(name).encode()))
            self.children[name] = _LazyParams(sub)
        return self.children[name]

    def materialize(self):
        if self.value is not None:
            return self.value
        return {k: c.materialize() for k, c in self.children.items()}

class Layer:

    def init_params(self, key, *xs):
        return None

    def forward(self, params, *xs):
        raise NotImplementedError()

    def __call__(self, params, *xs):
        if isinstance(params, _LazyParams) and not params.children:
            p = self.init_params(params.key, *xs)
            if p is not None:              # 末端 Layer: その場で初期化して以降は実パラメータで計算
                params.value = p
                params = p
        return self.forward(params, *xs)

    def init(self, key, *xs):
        """サンプル入力 xs を使ってパラメータ pytree を作る。"""
        lazy = _LazyParams(key)
        self(lazy, *xs)
        return lazy.materialize()

class Linear(Layer):

    def __init__(self, out_size, nobias=False, dtype=jnp.float32, in_size=None):
        self.out_size = out_size
        self.nobias = nobias
        self.dtype = dtype
        self.in_size = in_size

    def init_params(self, key, x):
        I = self.in_size if self.in_size is not None else x.shape[-1]
        O = self.out_size
        W = jax.random.normal(key, (I, O), dtype=self.dtype) * jnp.sqrt(1.0 / I).astype(self.dtype)
        params = {"W": W}
        if not self.nobias:
            params["b"] = jnp.zeros(O, dtype=self.dtype)
        return params

    def forward(self, params, x):
        return F.linear(x, params["W"], params.get("b"))