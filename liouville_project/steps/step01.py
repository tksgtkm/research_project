"""
変数クラスの実装
"""
import jax.numpy as np

# Variableクラスが変数となるように実装する。
class Variable:

    def __init__(self, data):
        # 初期化で与えられた変数をインスタンス変数のdataに設定する。
        self.data = data

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)