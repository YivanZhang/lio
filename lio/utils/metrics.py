from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    ts = []
    ys = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            ts.append(model(x.to(device)))
            ys.append(y.to(device))
    t = torch.cat(ts)
    y = torch.cat(ys)
    return t, y


def confusion_matrix(v1: torch.Tensor, v2: torch.Tensor,
                     n1: int = None, n2: int = None) -> torch.Tensor:
    if n1 is None:
        n1 = v1.max().item() + 1
    if n2 is None:
        n2 = v2.max().item() + 1
    matrix = torch.zeros(n1, n2).long().to(v1.device)
    pairs, counts = torch.unique(torch.stack((v1, v2)), dim=1, return_counts=True)
    matrix[pairs[0], pairs[1]] = counts
    return matrix


def expected_calibration_error(t: torch.Tensor, y: torch.Tensor, num_bins: int = 15) -> torch.Tensor:
    conf, y_ = F.softmax(t, dim=1).max(dim=1)
    acc = y.eq(y_).float()
    ece = torch.zeros(1).to(t.device)
    bins = torch.linspace(0, 1, num_bins + 1)
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        if (in_bin := conf.gt(bin_lower) & conf.le(bin_upper)).any():
            ece += (acc[in_bin].mean() - conf[in_bin].mean()).abs() * in_bin.float().mean()
    return ece
