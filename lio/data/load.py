from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def load_all_data(dataset: Dataset):
    return next(iter(DataLoader(dataset, batch_size=len(dataset))))


def load_mnist(data_dir: str, dataset_name: str) -> Tuple[Dataset, Dataset]:
    dataset = {
        'mnist': datasets.MNIST,
        'fashion-mnist': datasets.FashionMNIST,
        'kmnist': datasets.KMNIST,
    }[dataset_name]
    transform = transforms.ToTensor()
    dataset_tr = dataset(data_dir, train=True, transform=transform, download=True)
    dataset_ts = dataset(data_dir, train=False, transform=transform)
    return dataset_tr, dataset_ts


def load_cifar(data_dir: str, dataset_name: str) -> Tuple[Dataset, Dataset]:
    dataset = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
    }[dataset_name]
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
    }[dataset_name]
    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
    }[dataset_name]
    dataset_tr = dataset(data_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std),
                         ]))
    dataset_ts = dataset(data_dir, train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std),
                         ]))
    classes = {
        'cifar10': [
            'airplane', 'ship',
            'automobile', 'truck',
            'bird', 'deer', 'frog',
            'cat', 'dog', 'horse',
        ],
        'cifar100': [
            'beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
            'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',
        ],
    }[dataset_name]
    blocks = {
        'cifar10': [2, 2, 3, 3],
        'cifar100': [5] * 20,
    }[dataset_name]
    permutation = [classes.index(c) for c in dataset_tr.classes]
    for dataset in [dataset_tr, dataset_ts]:
        dataset.targets = torch.tensor(dataset.targets).apply_(lambda i: permutation[i])
        dataset.classes = classes
        dataset.blocks = blocks
    return dataset_tr, dataset_ts
