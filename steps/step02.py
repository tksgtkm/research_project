import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        self.data = data

"""
変数xとyをVariableインスタンスだと仮定して、
それらを処理できる関数fをFunctionクラスとして実装する

・Functionクラスで実装するメソッドはVariableインスタンスを入力とし、
  Variableインスタンスを出力とすること。
・Variableインスタンスの実際のデータはインスタンス変数んのdataに存在すること。
"""
class Function:
    # __call__メソッドはf = Function()としたとき
    # f(...)と書くことで__call__メソッドが呼び出せる。
    def __call__(self, input):
        # データを取り出す
        x = input.data
        # 実際の計算(forwardメソッドを使う)
        y = self.forward(x)
        # Variableとして返す
        output = Variable(y)
        return output

    def forward(self, in_data):
        raise NotImplementedError()

# Functionクラスは基底クラスとする
# そのため、SquareクラスはFunctionクラスを継承する
class Square(Function):

    def forward(self, x):
        return x ** 2

x = Variable(jnp.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)