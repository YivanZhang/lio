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


def load_svhn(data_dir: str) -> Tuple[Dataset, Dataset]:
    transform = transforms.ToTensor()
    dataset_tr = datasets.SVHN(data_dir, split='train', transform=transform, download=True)
    dataset_ts = datasets.SVHN(data_dir, split='test', transform=transform, download=True)
    return dataset_tr, dataset_ts


# permuted CIFAR classes
cifar10_classes = [
    # large vehicle
    'airplane', 'ship',
    # small vehicle
    'automobile', 'truck',
    # wild animal
    'bird', 'deer', 'frog',
    # domestic-animal
    'cat', 'dog', 'horse',
]

cifar100_classes = [
    # aquatic mammals
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    # fish
    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
    # flowers
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    # food containers
    'bottle', 'bowl', 'can', 'cup', 'plate',
    # fruit and vegetables
    'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
    # household electrical devices
    'clock', 'keyboard', 'lamp', 'telephone', 'television',
    # household furniture
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    # insects
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    # large carnivores
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    # large man-made outdoor things
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    # large natural outdoor scenes
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    # large omnivores and herbivores
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    # medium-sized mammals
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    # non-insect invertebrates
    'crab', 'lobster', 'snail', 'spider', 'worm',
    # people
    'baby', 'boy', 'girl', 'man', 'woman',
    # reptiles
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    # small mammals
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    # trees
    'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    # vehicles 1
    'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    # vehicles 2
    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',
]


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
        'cifar10': cifar10_classes,
        'cifar100': cifar100_classes,
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
