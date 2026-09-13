import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data

class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, in_data):
        raise NotImplementedError()

class Square(Function):

    def forward(self, x):
        return x ** 2

class Exp(Function):

    def forward(self, x):
        return jnp.exp(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square()
x = Variable(jnp.array(2.0))
dy = numerical_diff(f, x)
print(dy)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(jnp.array(0.5))
dy = numerical_diff(f, x)
print(dy)