import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data

class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, in_data):
        raise NotImplementedError()

class Square(Function):

    def forward(self, x):
        return x ** 2

class Exp(Function):

    def forward(self, x):
        return jnp.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(jnp.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)