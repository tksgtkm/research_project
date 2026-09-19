import jax.numpy as jnp

import liouville
from liouville import utils
from liouville.core import Function, Variable, as_variable, as_array

class Sin(Function):

    def forward(self, x):
        y = jnp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):

    def forward(self, x):
        y = jnp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):

    def forward(self, x):
        y = jnp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)