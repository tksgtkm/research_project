import unittest
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax import random

class Variable:

    def __init__(self, data):
        if data is not None:
            if not isinstance(data, jnp.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = jnp.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if jnp.isscalar(x):
        return jnp.array(x)
    return x

class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):

    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Variable(jnp.array(2.0))
        y = square(x)
        expected = jnp.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(jnp.array(3.0))
        y = square(x)
        y.backward()
        expected = jnp.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        key = random.key(42)
        key, subkey = random.split(key)
        a = random.uniform(subkey, (1,))
        x = Variable(a)
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = jnp.allclose(x.grad, num_grad)
        self.assertTrue(flg)

unittest.main()