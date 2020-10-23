import numpy as np


def sample_with_replacement(n: int, k: int, size: int) -> np.ndarray:
    return np.random.choice(n, size * k).reshape(size, k)


def sample_without_replacement_within_sets(n: int, k: int, size: int) -> np.ndarray:
    return np.stack([np.random.choice(n, k, replace=False) for _ in range(size)])


def sample_without_replacement(n: int, k: int, size: int) -> np.ndarray:
    return np.random.choice(n, size * k, replace=False).reshape(size, k)
