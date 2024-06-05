import numpy as np


def mb_used(tensor: np.ndarray) -> float:
    return np.prod(tensor.shape) * tensor.dtype.size / 1024 / 1024
