import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data
        # 微分した値を持つ
        self.grad = None

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

class Exp(Function):

    def forward(self, x):
        y = jnp.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = jnp.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(jnp.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = jnp.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)