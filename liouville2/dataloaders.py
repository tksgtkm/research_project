import math

import jax
import jax.numpy as jnp

__all__ = ["DataLoader", "epoch_permutation"]


def epoch_permutation(seed, epoch, n, shuffle=True):
    if not shuffle:
        return jnp.arange(n)
    key = jax.random.fold_in(jax.random.key(seed), epoch)
    return jax.random.permutation(key, n)


class DataLoader:

    def __init__(self, dataset, batch_size, shuffle=True, seed=42, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.data = dataset.arrays()           # (x, t)
        self.data_size = len(dataset)
        if drop_last:
            self.max_iter = self.data_size // batch_size
        else:
            self.max_iter = math.ceil(self.data_size / batch_size)
        self.epoch = 0
        self.iteration = 0

    def __len__(self):
        return self.max_iter

    def indices(self, epoch):
        """そのエポックのインデックスを (steps, batch_size) で返す (端数は捨てる)。"""
        perm = epoch_permutation(self.seed, epoch, self.data_size, self.shuffle)
        steps = self.data_size // self.batch_size
        return perm[:steps * self.batch_size].reshape(steps, self.batch_size)

    @staticmethod
    def gather(data, idx):
        x, t = data
        return x[idx], (None if t is None else t[idx])

    def reset(self):
        self.iteration = 0
        self.epoch += 1

    def __iter__(self):
        self.iteration = 0
        self._perm = epoch_permutation(self.seed, self.epoch, self.data_size, self.shuffle)
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        i, bs = self.iteration, self.batch_size
        batch = self.gather(self.data, self._perm[i * bs:(i + 1) * bs])
        self.iteration += 1
        return batch

    def next(self):
        return self.__next__()