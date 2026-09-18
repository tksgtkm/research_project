import jax.numpy as jnp

class Variable:

    def __init__(self, data):
        if data is not None:
            if not isinstance(data, jnp.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = jnp.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if jnp.isscalar(x):
        return jnp.array(x)
    return x

class Function:

    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):

    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

xs = [Variable(jnp.array(2)), Variable(jnp.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)