import os
import gzip
import tarfile
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from liouville.utils import get_file, cache_dir
from liouville.transforms import Compose, Flatten, ToFloat, Normalize

class Dataset:

    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert jnp.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

def get_spiral(train=True):
    seed = 1984 if train else 2026
    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data

    noise_key, perm_key = jax.random.split(jax.random.key(seed))

    j = jnp.repeat(jnp.arange(num_class), num_data)   # 0,0,...,1,1,...,2,2,...
    i = jnp.tile(jnp.arange(num_data), num_class)     # 0..99 を3回

    rate = i / num_data
    radius = 1.0 * rate
    theta = j * 4.0 + 4.0 * rate + jax.random.normal(noise_key, (data_size,)) * 0.2

    x = jnp.stack([radius * jnp.sin(theta),
                   radius * jnp.cos(theta)], axis=1).astype(jnp.float32)
    t = j.astype(jnp.int32)

    indices = jax.random.permutation(perm_key, data_size)
    return x[indices], t[indices]

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

class MNIST(Dataset):

    def __init__(self, train=True, transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]), target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            label = jnp.frombuffer(f.read(), jnp.uint8, offset=8)
        return label

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = jnp.frombuffer(f.read(), jnp.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def arrays(self):
        x = self.data.reshape(len(self.data), -1).astype(jnp.float32) / 255.0
        t = self.label.astype(jnp.int32)
        return x, t

    def show(self, row=10, col=10):
        H, W = 28, 28
        data = np.asarray(self.data)          # 表示用なのでNumPyに移す
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                idx = np.random.randint(0, len(data))
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = data[idx].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}