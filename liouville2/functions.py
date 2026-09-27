import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as _logsumexp

from liouville2.core import sum_to

__all__ = [
    "sin", "cos", "tanh", "exp", "log", "relu",
    "reshape", "transpose", "sum", "sum_to", "broadcast_to", "max",
    "matmul", "linear", "sigmoid", "softmax", "logsumexp",
    "mean_squared_error", "softmax_cross_entropy", "accuracy",
]

sin = jnp.sin
cos = jnp.cos
tanh = jnp.tanh
exp = jnp.exp
log = jnp.log

def relu(x):
    return jnp.maximum(x, 0)

def sigmoid(x):
    return jnp.tanh(x * 0.5) * 0.5 + 0.5

def reshape(x, shape):
    return jnp.reshape(x, shape)

def transpose(x, axes=None):
    return jnp.transpose(x, axes)

def sum(x, axis=None, keepdims=False):
    return jnp.sum(x, axis=axis, keepdims=keepdims)

def broadcast_to(x, shape):
    return jnp.broadcast_to(x, shape)

def max(x, axis=None, keepdims=False):
    return jnp.max(x, axis=axis, keepdims=keepdims)

def logsumexp(x, axis=1):
    return _logsumexp(x, axis=axis, keepdims=True)

def matmul(x, W):
    return x @ W

def linear(x, W, b=None):
    y = x @ W
    return y if b is None else y + b

def softmax(x, axis=1):
    return jax.nn.softmax(x, axis=axis)

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return jnp.sum(diff ** 2) / diff.shape[0]

def softmax_cross_entropy(x, t):
    N = x.shape[0]
    log_p = x - logsumexp(x, axis=1)
    return -jnp.sum(log_p[jnp.arange(N), t.ravel()]) / N

def accuracy(y, t):
    pred = jnp.argmax(y, axis=1).reshape(t.shape)
    return jnp.mean(pred == t)