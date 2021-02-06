from typing import Iterable

import numpy as np

import torch

from lio.observations import sample_without_replacement, observe_categorical
from lio.utils.data import LioDataset, load_all_data
from lio.utils.metrics import confusion_matrix


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def diag_matrix(n: int, diagonal: float, off_diagonal: float) -> torch.Tensor:
    return off_diagonal * torch.ones(n, n) + (diagonal - off_diagonal) * torch.eye(n, n)


def synthetic_noise(dataset, transition_matrix: np.ndarray = None):
    _, z = load_all_data(dataset)
    indices = sample_without_replacement(n=len(z), k=1, size=len(z))
    observations = observe_categorical(torch.tensor(transition_matrix))(z, indices)
    dataset = LioDataset(dataset, indices, observations)
    return dataset


def take_cycle(n: int, xs: Iterable):
    # take n (cycle xs)
    it = iter(xs)
    for i in range(n):
        try:
            yield next(it)
        except StopIteration:
            it = iter(xs)
            yield next(it)


# ----------------------------------------------------------------------------------------------------------------------

def anchor_points_estimation(p: torch.Tensor, _):
    return p[p.argmax(dim=0)]


def confusion_matrix_estimation(p: torch.Tensor, y: torch.Tensor):
    m = confusion_matrix(p.argmax(dim=1), y)
    t1 = m / m.sum(dim=1, keepdim=True)
    t2 = p[p.argmax(dim=0)]
    return t1 @ t2


def quantile_estimation(p: torch.Tensor, _, q: float = 0.97):
    threshold = torch.quantile(p, q, dim=0)
    return p[torch.where(p >= threshold, torch.zeros_like(p), p).argmax(dim=0)]
