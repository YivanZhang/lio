from typing import Callable

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
