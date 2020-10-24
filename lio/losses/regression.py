import math
from typing import Callable, List

import torch
import torch.nn.functional as F


def direct_gaussian_loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(t, y)


def direct_cauchy_loss(scale: float) -> Callable:
    def loss(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.log1p(((t - y) / scale).pow(2)).mean()

    return loss


# ----------------------------------------------------------------------------------------------------------------------

def mean_gaussian_loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    t = torch.cat(ts, dim=1).mean(dim=1, keepdim=True)
    return F.mse_loss(t, y)


def mean_cauchy_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        t = torch.cat(ts, dim=1).mean(dim=1, keepdim=True)
        return torch.log1p(((t - y) / scale).pow(2)).mean()

    return loss


# ----------------------------------------------------------------------------------------------------------------------

def rank_gaussian_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        p = 0.5 * (1 + torch.erf((ts[0] - ts[1]) / (2 * scale)))
        return F.binary_cross_entropy(p, y)

    return loss


def rank_cauchy_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        p = torch.atan((ts[0] - ts[1]) / (2 * scale)) / math.pi + 0.5
        return F.binary_cross_entropy(p, y)

    return loss


def rank_gumbel_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits((ts[0] - ts[1]) / scale, y)

    return loss


# ----------------------------------------------------------------------------------------------------------------------

def min_gaussian_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        t = torch.cat(ts, dim=1)
        t += torch.randn_like(t) * scale
        return F.mse_loss(t.min(dim=1, keepdim=True)[0], y)

    return loss


# ----------------------------------------------------------------------------------------------------------------------

def max_gaussian_loss(scale: float) -> Callable:
    def loss(ts: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        t = torch.cat(ts, dim=1)
        t += torch.randn_like(t) * scale
        return F.mse_loss(t.max(dim=1, keepdim=True)[0], y)

    return loss
