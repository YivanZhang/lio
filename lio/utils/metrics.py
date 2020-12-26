from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ts = []
    ys = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            ts.append(model(x))
            ys.append(y)
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
