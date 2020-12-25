import numpy as np


def get_identity(num_classes: int) -> np.ndarray:
    return np.eye(num_classes, num_classes, dtype=float)


def get_symmetric_noise(num_classes: int, noise_rate: float) -> np.ndarray:
    if num_classes == 1:
        return np.ones((1, 1), dtype=float)
    else:
        m = np.ones((num_classes, num_classes), dtype=float) * noise_rate / (num_classes - 1)
        np.fill_diagonal(m, 1 - noise_rate)
        return m


def get_pairwise_noise(num_classes: int, noise_rate: float) -> np.ndarray:
    if num_classes == 1:
        return np.ones((1, 1), dtype=float)
    else:
        m = np.zeros((num_classes, num_classes), dtype=float)
        np.fill_diagonal(m, 1 - noise_rate)
        np.fill_diagonal(m[:-1, 1:], noise_rate)
        m[-1, 0] = noise_rate
        return m


def get_uniform_complement(num_classes: int) -> np.ndarray:
    return get_symmetric_noise(num_classes, 1.)
