import gzip

import numpy as np
import jax
import jax.numpy as jnp

from liouville.utils import get_file
from liouville.transforms import Compose, Flatten, ToFloat, Normalize

__all__ = ["Dataset", "Spiral", "MNIST", "get_spiral"]


class Dataset:

    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform if transform is not None else (lambda x: x)
        self.target_transform = target_transform if target_transform is not None else (lambda x: x)
        self.data = None
        self.label = None
        self._cache = None
        self.prepare()

    def __getitem__(self, index):
        x = self.transform(self.data[index])
        if self.label is None:
            return x, None
        return x, self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

    def arrays(self):
        if self._cache is None:
            x = _apply_batched(self.transform, self.data)
            t = None if self.label is None else _apply_batched(self.target_transform, self.label)
            self._cache = (x, t)
        return self._cache


def _apply_batched(fn, arr):
    try:
        return jax.vmap(fn)(jnp.asarray(arr))
    except Exception:
        return jnp.stack([jnp.asarray(fn(a)) for a in arr])


# ----------------------------------------------------------------------
def get_spiral(train=True):
    seed = 1984 if train else 2026
    num_data, num_class = 100, 3
    data_size = num_class * num_data

    noise_key, perm_key = jax.random.split(jax.random.key(seed))

    j = jnp.repeat(jnp.arange(num_class), num_data)
    i = jnp.tile(jnp.arange(num_data), num_class)

    rate = i / num_data
    radius = 1.0 * rate
    theta = j * 4.0 + 4.0 * rate + jax.random.normal(noise_key, (data_size,)) * 0.2

    x = jnp.stack([radius * jnp.sin(theta), radius * jnp.cos(theta)], axis=1).astype(jnp.float32)
    t = j.astype(jnp.int32)

    indices = jax.random.permutation(perm_key, data_size)
    return x[indices], t[indices]


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz', 'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz', 'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        self.data = self._load_data(get_file(url + files['target']))
        self.label = self._load_label(get_file(url + files['label']))

    @staticmethod
    def _load_label(filepath):
        with gzip.open(filepath, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

    @staticmethod
    def _load_data(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 1, 28, 28)

    def show(self, row=10, col=10, seed=0):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = \
                    self.data[rng.integers(0, len(self.data))].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {i: str(i) for i in range(10)}