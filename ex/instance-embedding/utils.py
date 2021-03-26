from collections import Iterable

import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def take_cycle(n: int, xs: Iterable):
    # take n (cycle xs)
    it = iter(xs)
    for i in range(n):
        try:
            yield next(it)
        except StopIteration:
            it = iter(xs)
            yield next(it)
