import inspect
from functools import wraps
from typing import Callable

import torch
from torch import autograd


def straight_through(func: Callable, num_args: int = None):
    if num_args is None:
        sig = inspect.signature(func)
        num_args = len(sig.parameters) - 1
    dct = {'forward': staticmethod(lambda ctx, *args: func(*args)),
           'backward': staticmethod(lambda ctx, grad_output: (grad_output, *[None] * num_args))}
    cls = type(func.__name__, (autograd.Function,), dct)

    @wraps(func)
    def func_st(*args):
        return cls.apply(*args)

    return func_st


def sparsemax_threshold(x: torch.Tensor, dim: int) -> torch.Tensor:
    z = x.detach().sort(dim=dim, descending=True)[0]
    arange = torch.arange(start=1, end=z.shape[dim] + 1).to(z.device)
    cumsum = z.cumsum(dim=dim) - 1
    kz = (arange * z.transpose(dim, -1)).transpose(dim, -1)
    k = (kz > cumsum).sum(dim=dim, keepdim=True)
    t = cumsum.gather(dim=dim, index=k - 1) / k
    return t


def sparsemax(x: torch.Tensor, dim: int) -> torch.Tensor:
    t = sparsemax_threshold(x, dim)
    p = torch.clamp(x - t, min=0.)
    return p
