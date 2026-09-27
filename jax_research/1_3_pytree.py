"""
jaxにはpytreeというネストされた構文をサポートする機能が含まれている。
"""
import operator
import collections

import numpy as np
import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

for pytree in example_trees:
    leaves = jax.tree.leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

# 概念的にはどのpytreeも2つの部分に分割できる。
# leaves(データ)とtreedef(構造)に分割する
params = {'W': jnp.zeros((2, 3)), 'b': jnp.zeros(3)}

# このフラット化/非フラット化の分解がjaxがpytreeをサポートする仕組みで、
# 内部では配列のフラットなリストに対して操作を行い、その結果に基づいて構造を構築する。
# 構造としてはツリー構造で、参照透過性を前提としている。
leaves, treedef = jax.tree.flatten(params)
print(leaves)
print(treedef)
print(jax.tree.unflatten(treedef, leaves))

# 一般的なpytree関数

# よく使うであろう関数はjax.tree.map()
# pytree全体に対してmap操作を行う
list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

print(jax.tree.map(lambda x: x * 2, list_of_lists))

# pytreeに対して一度に関数をマッピングすることもサポートしている。
another_list_of_lists = list_of_lists
print(jax.tree.map(lambda x, y: x + y, list_of_lists, another_list_of_lists))

# 他にはreduce関数もある
# print(jax.tree.reduce(operator.add, list_of_lists))
print(jax.tree.reduce(lambda x, y: x + y, list_of_lists))

# 機械学習で使うケースだと、パラメータはpytreeに格納され
# jax.grad勾配に対応するpytreeが生成され、jax.tree.mapが適用される。

def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
                biases=np.ones(shape=(n_out,))
            )
        )
    return params

params = init_mlp_params([1, 128, 128, 1])

print(jax.tree.map(lambda x: x.shape, params))

# 続いて、順伝播、損失、更新ステップを定義する
def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.001

def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(
        lambda p, g: p - LEARNING_RATE * g, params, grads
    )

x = np.random.normal(size=(128, 1))
y = x ** 2

for _ in range(100):
    params = update(params, x, y)

print(loss_fn(params, x, y))

# 明示的なキーパス
# pytreeの各リーフにはキーパスがある
ATuple = collections.namedtuple("ATuple", ('name',))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

for key_path, value in flattened:
    print(f'Value of tree {jax.tree_util.keystr(key_path)}: {value}')

# カスタムpytreeノード

# pytreeレジストリに登録されていない型はすべてリーフとして扱われる
class Special:

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

print(jax.tree.leaves([Special('a', 0, 1), Special('b', 2, 4)]))

# 独自のクラスをregister_pytree_node関数に登録し、インスタンスをフラット化する。
from jax.tree_util import register_pytree_node

class RegisteredSpecial(Special):

    def __repr__(self):
        return f"RegisteredSpecial({self.name!r}, x={self.x}, y={self.y})"

def special_flatten(v):
    children = (v.x, v.y)
    aux_data = v.name
    return children, aux_data

def special_unflatten(aux_data, children):
    return RegisteredSpecial(aux_data, *children)

print(register_pytree_node(RegisteredSpecial, special_flatten, special_unflatten))

print(jax.tree.map(lambda x: x + 1, [RegisteredSpecial('a', 0, 1), RegisteredSpecial('b', 2, 4)]))

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class RegisteredSpecial2(Special):

    def __repr__(self):
        return f"RegisteredSpecial2({self.name!r}, x={self.x}, y={self.y})"

    def tree_flatten(self):
        children = (self.x, self.y)
        aux_data = self.name
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)

print(jax.tree.map(lambda x: x + 1, [RegisteredSpecial2('a', 0, 1), RegisteredSpecial2('b', 2, 4)]))

from typing import NamedTuple, Any

class MyOtherContainer(NamedTuple):
    name: str
    a: any
    b: any

print(jax.tree.leaves([
    MyOtherContainer('Alice', 1, 2),
    MyOtherContainer('Bob', 3, 4)
]))

# データクラスの登録
from dataclasses import dataclass
import functools

@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['a', 'b'],
    meta_fields=['name']
)
@dataclass
class MyDataclassContainer:
    name: str
    a: Any
    b: Any

print(jax.tree.leaves([
    MyDataclassContainer('apple', 5.3, 4.7),
    MyDataclassContainer('banana', jnp.zeros(4), -1.0)
]))