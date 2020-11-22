import numpy as np


def cov2d(std: np.ndarray, corr: np.ndarray):
    num_classes, dim_features = std.shape
    assert dim_features == 2
    assert corr.shape == (num_classes,)

    cov = np.zeros((num_classes, dim_features, dim_features))
    for i in range(num_classes):
        cov[i] = np.diag(std[i]) @ np.array([[1, corr[i]],
                                             [corr[i], 1]]) @ np.diag(std[i])
    return cov


def gaussian_mixture(mean: np.ndarray, cov: np.ndarray, n: int, p: np.ndarray = None):
    num_classes, dim_features = mean.shape
    if p is None:
        p = np.ones(num_classes) / num_classes
    assert cov.shape == (num_classes, dim_features, dim_features)
    assert p.shape == (num_classes,)

    # components [num_points]
    z = np.random.choice(num_classes, size=n, p=p)
    # features [num_points, dim_features]
    x = mean[z] + (np.linalg.cholesky(cov)[z] @ np.random.randn(n, dim_features, 1)).squeeze()
    return x, z
