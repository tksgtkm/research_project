import os
import urllib.request

import numpy as np
import jax
from jax import tree_util as tu

__all__ = ["jaxpr", "get_file", "cache_dir", "pair", "save_params", "load_params"]


def jaxpr(fn, *args):
    """fn(*args) の計算グラフを jaxpr として返す。print すると読める。"""
    return jax.make_jaxpr(fn)(*args)

def _key_str(path):
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


def save_params(path, params):
    flat = {_key_str(p): np.asarray(v) for p, v in tu.tree_flatten_with_path(params)[0]}
    try:
        np.savez(path, **flat)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(path):
            os.remove(path)
        raise


def load_params(path, like):
    """like と同じ構造の pytree に読み込む"""
    npz = np.load(path)
    leaves = [jax.numpy.asarray(npz[_key_str(p)]) for p, _ in tu.tree_flatten_with_path(like)[0]]
    return tu.tree_unflatten(tu.tree_structure(like), leaves)


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    p = min(downloaded / total_size * 100, 100.0)
    i = min(int(downloaded / total_size * 30), 30)
    print("\r[{}] {:.2f}%".format("#" * i + "." * (30 - i), p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.liouville')


def get_file(url, file_name=None):
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")
    return file_path


def pair(x):
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    raise ValueError