"""
再帰からループへ
"""

import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        """
        backwardメソッドの中でbackwardメソッドが呼ばれ、その呼ばれた先のbackwardメソッドでまた
        backwardメソッドが呼ばれ・・・という処理が続く。
        これは関数self.creatorがNoneになるまで続く
        """
        funcs = [self.creator]
        while funcs:
            # 関数を取得
            f = funcs.pop()
            # 関数の入出力を取得
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                # 1つ前の関数をリストに追加
                funcs.append(x.creator)

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
y.backward()
print(x.grad)