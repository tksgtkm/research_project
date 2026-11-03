"""
jaxの基本データ型は配列であり、ここはnumpyと類似している。
https://docs.jax.dev/en/latest/101/arrays.html
"""

import jax
import jax.numpy as jnp
import numpy as np

x = jnp.arange(9.0).reshape(3, 3)
y = jnp.ones(3)

print(x @ y)
print(x.sum(axis=0))
print(x[1, :2])

# 配列はjax.Arrayインスタンスとして扱う
print(isinstance(x, jax.Array))

# jaxの型を調べるには
print(jax.typeof(x))

# jaxとnumpyの最も重要な違いは、jax配列は一度作成すると内容を変更できない点にある。
# numpyでは配列をその場で変更することもできる
x_np = np.arange(10)
x_np[0] = 10
print(x_np)

x = jnp.arange(10)
# ここでエラーになる
# x[0] = 10

# ただし以下のようにして更新された配列を新しく作ることはできる
y = x.at[0].set(10)
print(x)
print(y)

# setの操作以外にもadd, multiply, min, maxが使える
print(x.at[3].add(100))
print(x.at[:3].max(9))

# numpyはデフォルトで64ビットだがjaxでは32ビットとなっている。
# これはGPUやTPUではそっちのほうが望ましいため。
print(np.array([1.0, 2.0]).dtype)
print(jnp.array([1.0, 2.0]).dtype)

# jaxで64ビットの精度にするには
jax.config.update("jax_enable_x64", True)

# jaxは配列の範囲外のインデックスにアクセスしてもエラーが発生しない
x = jnp.arange(10)
print(x[11])

#jax.numpyは利便性レイヤーであり、その機能はjax.laxにより低レベルなAPIに基づいて実装されている。

def sin(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(float)
    return jax.lax.sin(x)

x = jnp.arange(10)
print(sin(x))