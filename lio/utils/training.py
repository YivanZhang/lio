from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ts = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            ts.append(model(x))
            ys.append(y)
    t = torch.cat(ts)
    y = torch.cat(ys)
    return t, y
