"""
逆誤差伝播法の自動化

逆伝播を自動化するに当たり、変数と関数の関係性について
関数から変数はどのように見えるかという点について考える。
関数から見ると変数は入力と出力として存在する。
逆に変数からは見ると、変数は関数から生み出されると考える。
そこで、変数からみた関数はcreatorとして扱う。
"""

import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data
        self.grad = None
        # creatorというインスタンス変数を追加
        self.creator = None

    # creatorを設定するためのメソッド
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 関数を取得
        f = self.creator
        if f is not None:
            # 関数の入力を取得
            x = f.input
            # 関数のbackwardメソッドを呼ぶ
            x.grad = f.backward(self.grad)
            # 自分より1つ前の変数のbackwardメソッドを呼ぶ(再帰)
            x.backward()

class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 出力変数に生みの親を覚えさせる
        output.set_creator(self)
        self.input = input
        # 出力も覚える
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