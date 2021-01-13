import torch
from torch.utils.data import Sampler


class InfiniteSampler(Sampler):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            yield torch.randint(high=self.num_samples, size=(1,), dtype=torch.int64).item()

    def __len__(self):
        return float('inf')
