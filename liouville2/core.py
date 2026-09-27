import jax
import jax.numpy as jnp
from jax import tree_util as tu
from jax.flatten_util import ravel_pytree

__all__ = [
    "grad", "value_and_grad", "hessian", "stop_gradient", "enable_x64",
    "Function", "sum_to",
    "tree_size", "tree_zeros_like", "tree_add", "tree_sub", "tree_scale",
    "tree_axpy", "tree_dot", "tree_norm", "ravel",
]

def enable_x64(flag=True):
    jax.config.update("jax_enable_x64", flag)

grad = jax.grad
value_and_grad = jax.value_and_grad
hessian = jax.hessian

def stop_gradient(tree):
    return tu.tree_map(jax.lax.stop_gradient, tree)

def sum_to(x, shape):
    shape = tuple(shape)
    if x.shape == shape:
        return x
    lead = x.ndim - len(shape)
    x = jnp.sum(x, axis=tuple(range(lead))) if lead > 0 else x
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and x.shape[i] != 1)
    if axes:
        x = jnp.sum(x, axis=axes, keepdims=True)
    return x.reshape(shape)

class Function:

    def forward(self, *xs):
        raise NotImplementedError()
    
    def backward(self, xs, y, gy):
        raise NotImplementedError()
    
    def _build(self):
        @jax.custom_vjp
        def f(*xs):
            return self.forward(*xs)
        
        def f_fwd(*xs):
            y = self.forward(*xs)
            return y, (xs, y)
        
        def f_bwd(res, gy):
            xs, y = res
            gxs = self.backward(xs, y, gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            return tuple(sum_to(gx, x.shape) for gx, x in zip(gxs, xs))
        
        f.defvjp(f_fwd, f_bwd)
        return f
    
    def __call__(self, *xs):
        if not hasattr(self, "_f"):
            self._f = self._build()
        xs = tuple(jnp.asarray(x) for x in xs)
        return self._f(*xs)

def tree_size(tree):
    return sum(x.size for x in tu.tree_leaves(tree))

def tree_zeros_like(tree):
    return tu.tree_map(jnp.zeros_like, tree)


def tree_add(a, b):
    return tu.tree_map(jnp.add, a, b)


def tree_sub(a, b):
    return tu.tree_map(jnp.subtract, a, b)


def tree_scale(tree, s):
    return tu.tree_map(lambda x: s * x, tree)


def tree_axpy(a, x, y):
    return tu.tree_map(lambda xi, yi: a * xi + yi, x, y)


def tree_dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(tu.tree_leaves(a), tu.tree_leaves(b)))


def tree_norm(tree):
    return jnp.sqrt(tree_dot(tree, tree))


def ravel(tree):
    return ravel_pytree(tree)