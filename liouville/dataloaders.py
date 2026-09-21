import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import jax
import jax.numpy as jnp

class DataLoader:

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.reset()

    def reset(self):
        self.iteration = 0
        key = jax.random.key(42)
        key, subkey = jax.random.split(key)
        if self.shuffle:
            self.index = jax.random.permutation(subkey, len(self.dataset))
        else:
            self.index = jnp.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        x = jnp.array([example[0] for example in batch])
        t = jnp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        self.__next__()
