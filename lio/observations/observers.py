import operator
from functools import reduce
from typing import Callable, List

import numpy as np

import torch
from torch.distributions import Categorical


def observe(f: Callable):
    def observer(zs: torch.Tensor, indices: List[List[int]]) -> torch.Tensor:
        return torch.stack([f(*zs[i]) for i in indices])

    return observer


def observe_categorical(transition_matrix: np.ndarray):
    transition_matrix = torch.tensor(transition_matrix).float()
    dists = [Categorical(probs=p) for p in transition_matrix]
    return observe(lambda z: dists[z].sample())


# binary
observe_similarity = observe(operator.eq)
observe_difference = observe(operator.sub)
observe_rank = observe(operator.gt)

# ternary
observe_triplet = observe(lambda a, b, c: 1 if (a == b and a != c) else 2 if (a == c and a != b) else 0)

# variadic
observe_sum = observe(lambda *zs: reduce(operator.add, zs))
observe_mean = observe(lambda *zs: reduce(operator.add, zs) / len(zs))
observe_min = observe(lambda *zs: reduce(lambda a, b: a if (a < b) else b, zs))
observe_max = observe(lambda *zs: reduce(lambda a, b: a if (a > b) else b, zs))
observe_uncoupled = observe(lambda *zs: torch.stack(zs)[torch.randperm(len(zs))])
