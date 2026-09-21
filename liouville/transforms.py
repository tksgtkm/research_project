import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

from liouville.utils import pair


class Compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms is not None else []

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for PIL Image
# =============================================================================
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            return Image.merge('RGB', (b, g, r))
        return img.convert(self.mode)


class Resize:
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size
        mode (int): Desired interpolation.
    """
    def __init__(self, size, mode=Image.Resampling.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    """Crop the center of the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size.
    """
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    """Convert PIL Image to JAX array of shape (C, H, W)."""
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, jax.Array):
            return img
        if isinstance(img, np.ndarray):
            return jnp.asarray(img)
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.ndim == 2:            # grayscale: (H, W) -> (1, H, W)
                arr = arr[None, :, :]
            else:                        # (H, W, C) -> (C, H, W)
                arr = arr.transpose(2, 0, 1)
            return jnp.asarray(arr, dtype=self.dtype)
        raise TypeError(f'Unsupported type: {type(img)}')


class ToPIL:
    """Convert (C, H, W) array to PIL Image."""
    def __call__(self, array):
        data = np.asarray(array)         # PIL needs a NumPy array
        if data.shape[0] == 1:           # (1, H, W) -> (H, W)
            data = data[0]
        else:                            # (C, H, W) -> (H, W, C)
            data = data.transpose(1, 2, 0)
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


# =============================================================================
# Transforms for JAX arrays
# =============================================================================
class Normalize:
    """Normalize an array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence): std for all values or sequence of stds for
         each channel.
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        array = jnp.asarray(array)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            array = array.astype(jnp.float32)
        mean = self._per_channel(self.mean, array)
        std = self._per_channel(self.std, array)
        return (array - mean) / std

    @staticmethod
    def _per_channel(v, array):
        v = jnp.asarray(v, dtype=array.dtype)
        if v.ndim == 1:                  # (C,) -> (C, 1, 1, ...)
            v = v.reshape((-1,) + (1,) * (array.ndim - 1))
        return v


class Flatten:
    """Flatten an array."""
    def __call__(self, array):
        return jnp.ravel(array)


class AsType:
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

    def __call__(self, array):
        return jnp.asarray(array).astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=jnp.int32):
        self.dtype = dtype