import liouville2.functions as F
import liouville2.layers as L
from liouville2 import utils
from liouville2.layers import Layer

__all__ = ["Model", "MLP"]


class Model(Layer):

    def jaxpr(self, params, *xs):
        return utils.jaxpr(lambda p, *a: self(p, *a), params, *xs)


class MLP(Model):

    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        self.activation = activation
        self.layers = [L.Linear(out_size) for out_size in fc_output_sizes]

    def forward(self, params, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(params[f"l{i}"], x))
        last = len(self.layers) - 1
        return self.layers[-1](params[f"l{last}"], x)