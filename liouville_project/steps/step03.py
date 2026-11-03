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

# Exp関数の実装
class Exp(Function):

    def forward(self, x):
        return jnp.exp(x)

# Functionクラスの__call__メソッドの入力と出力はともにVariableインスタンス
# そのため関数を連続して使用することもできる
A = Square()
B = Exp()
C = Square()

x = Variable(jnp.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)