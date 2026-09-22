import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import jax
import jax.numpy as jnp

class DataLoader:

    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.key = jax.random.key(seed)
        self._arrays = dataset.arrays() if hasattr(dataset, 'arrays') else None
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            self.index = jax.random.permutation(subkey, self.data_size)
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
        self.iteration += 1

        if self._arrays is not None:
            x_all, t_all = self._arrays
            return x_all[batch_index], t_all[batch_index]
        
        batch = [self.dataset[int(i)] for i in batch_index]
        x = jnp.stack([ex[0] for ex in batch])
        t = jnp.array([ex[1] for ex in batch])
        return x, t

    def next(self):
        return self.__next__()
