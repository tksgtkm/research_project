import jax.numpy as jnp

x = jnp.array(1)
print(x.ndim)

x = jnp.array([1, 2, 3])
print(x.ndim)

x = jnp.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(x.ndim)