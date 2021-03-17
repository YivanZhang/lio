from typing import Callable, List

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def direct_observation_loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(t, y)


def indirect_observation_loss(transition_matrix: torch.Tensor, activation: Callable = None) -> Callable:
    if activation is None:
        activation = lambda t: F.softmax(t, dim=1)

    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p_z = activation(t)
        p_y = p_z @ transition_matrix.to(y.device)
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    return loss


def pairwise_similarity_loss(activation: Callable = None) -> Callable:
    if activation is None:
        activation = lambda t: F.softmax(t, dim=1)

    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        t1, t2 = ts
        p_z1 = activation(t1)
        p_z2 = activation(t2)
        p_y1 = (p_z1 * p_z2).sum(dim=1)
        p_y0 = 1. - p_y1
        p_y = torch.stack((p_y0, p_y1), dim=1)
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    return loss


def triplet_comparison_loss(activation: Callable = None) -> Callable:
    if activation is None:
        activation = lambda t: F.softmax(t, dim=1)

    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        t1, t2, t3 = ts
        p_z1 = activation(t1)
        p_z2 = activation(t2)
        p_z3 = activation(t3)
        p_z12 = (p_z1 * p_z2).sum(dim=1)
        p_z13 = (p_z1 * p_z3).sum(dim=1)
        p_y1 = p_z12 * (1. - p_z13)
        p_y2 = (1. - p_z12) * p_z13
        p_y0 = 1. - p_y1 - p_y2
        p_y = torch.stack((p_y0, p_y1, p_y2), dim=1)
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    return loss


# ----------------------------------------------------------------------------------------------------------------------

def soft_bootstrapping_loss(beta: float) -> Callable:
    # https://arxiv.org/abs/1412.6596
    # entropy regularization

    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nll = F.cross_entropy(t, y)
        reg = -Categorical(probs=F.softmax(t, dim=1)).entropy().mean()
        return beta * nll + (1. - beta) * reg

    return loss


def hard_bootstrapping_loss(beta: float) -> Callable:
    # https://arxiv.org/abs/1412.6596
    # log-likelihood regularization

    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nll = F.cross_entropy(t, y)
        reg = F.cross_entropy(t, t.argmax(dim=1))
        return beta * nll + (1. - beta) * reg

    return loss


def focal_loss(gamma: float) -> Callable:
    # https://arxiv.org/abs/1708.02002

    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((1. - F.softmax(t, dim=1)[range(len(y)), y]) ** gamma *
                F.cross_entropy(t, y, reduction='none')).mean()

    return loss


def generalized_cross_entropy_loss(q: float) -> Callable:
    # https://arxiv.org/abs/1805.07836

    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1. - F.softmax(t, dim=1)[range(len(y)), y] ** q).mean() / q

    return loss
