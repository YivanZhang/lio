from typing import Sequence

from torch.utils.data import Dataset


class LioDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 indices: Sequence[Sequence[int]],
                 targets: Sequence,
                 ):
        super(LioDataset, self).__init__()
        assert len(indices) == len(targets)
        self.dataset = dataset
        self.indices = indices
        self.targets = targets

    def __getitem__(self, index):
        xs = [self.dataset[i][0] for i in self.indices[index]]
        y = self.targets[index]
        if len(xs) == 1:
            return xs[0], y
        else:
            return xs, y

    def __len__(self):
        return len(self.indices)


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)
