from torch import nn

from .layers import FlattenLayer


def linear(dim_output: int = 10):
    return nn.Sequential(
        FlattenLayer(),
        nn.Linear(28 ** 2, dim_output),
    )


def cnn(dim_output: int = 10):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        FlattenLayer(),
        nn.Linear(64 * 10 ** 2, 128),
        nn.Dropout(0.5),
        nn.ReLU(inplace=True),
        nn.Linear(128, dim_output),
    )
