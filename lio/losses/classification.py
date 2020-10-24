from typing import Callable, List

import torch
import torch.nn.functional as F


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
