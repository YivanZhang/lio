from datetime import datetime
from functools import lru_cache

import numpy as np
from sacred import Experiment

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from lio.models import mnist, resnet
from lio.observations import get_identity, get_symmetric_noise, get_pairwise_noise, get_random_noise
from lio.utils.data import load_all_data, load_mnist, load_cifar
from lio.utils.metrics import predict

from core import (
    NoTransition, CategoricalTransition, DirichletTransition,
    no_regularization, tv_regularization,
    get_train_step,
)
from utils import get_device, diag_matrix, synthetic_noise, take_cycle

ex = Experiment('transition-matrix')


@ex.config
def config():
    # data
    dataset_name = 'mnist'
    data_dir = 'data/mnist' if 'mnist' in dataset_name else 'data/cifar'
    noise_type = 'symm'
    num_classes = 10 if dataset_name != 'cifar100' else 100

    # loaders
    batch_size = 512
    num_workers = 4 * torch.cuda.device_count()

    # iterations
    num_iter_total = 2000 if 'mnist' in dataset_name else 4000
    num_iter_warmup = 400
    num_iter_test = int(num_iter_total / 10)

    # model
    device = get_device()
    mnist = dict(
        device=device,
        lr=1e-3, lr_decay=0.1 ** (1 / num_iter_total),
    )
    cifar = dict(
        device=device, num_classes=num_classes,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
        lr=0.1, momentum=0.9, weight_decay=1e-4,
    )

    # transition
    transition_type = 'none'
    categorical = dict(
        device=device, num_classes=num_classes,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
        diagonal=np.log(0.5), off_diagonal=np.log(0.5 / (num_classes - 1)),
        lr=5e-3,
    )
    dirichlet = dict(
        device=device, num_classes=num_classes,
        diagonal=10. if 'mnist' in dataset_name else 100., off_diagonal=0.,
        betas=(0.999, 0.01),
    )

    # regularization
    regularization_type = 'none'
    num_pairs = batch_size
    gamma = 0.1


@ex.capture
@lru_cache(maxsize=1)
def get_datasets(data_dir, dataset_name):
    if 'mnist' in dataset_name:
        dataset_tr, dataset_ts = tuple(TensorDataset(*load_all_data(dataset))
                                       for dataset in load_mnist(data_dir, dataset_name))
    elif 'cifar' in dataset_name:
        dataset_tr, dataset_ts = load_cifar(data_dir=data_dir, dataset_name=dataset_name)
    else:
        raise ValueError
    return dataset_tr, dataset_ts


@ex.capture
@lru_cache(maxsize=1)
def get_transition_matrix(noise_type, num_classes, _log):
    # clean
    if noise_type == 'none':
        transition_matrix = get_identity(num_classes)
    # noisy
    elif noise_type == 'symm':
        transition_matrix = get_symmetric_noise(num_classes, 0.5)
    elif noise_type == 'pair':
        transition_matrix = get_pairwise_noise(num_classes, 0.4)
    elif noise_type == 'pair2':
        transition_matrix = get_pairwise_noise(num_classes, 0.3) @ get_pairwise_noise(num_classes, 0.2)
    elif noise_type == 'trid':
        transition_matrix = get_pairwise_noise(num_classes, 0.3) @ get_pairwise_noise(num_classes, 0.3).T
    elif noise_type == 'rand':
        transition_matrix = get_random_noise(num_classes, 0.5, concentration=0.5)
    else:
        raise ValueError
    with np.printoptions(precision=2, suppress=True):
        _log.info(f'\n{transition_matrix}')
    return transition_matrix


@ex.capture
@lru_cache(maxsize=1)
def add_noise(dataset, noise_type):
    if noise_type == 'clean':
        return dataset
    else:
        transition_matrix = get_transition_matrix()
        return synthetic_noise(dataset, transition_matrix)


@ex.capture
def get_loaders(dataset_tr, dataset_ts,
                batch_size, num_workers):
    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    loader_tr = DataLoader(dataset_tr, shuffle=True, **loader_kwargs)
    loader_ts = DataLoader(dataset_ts, **loader_kwargs)
    return loader_tr, loader_ts


@ex.capture(prefix='mnist')
def model_mnist(device, lr, lr_decay):
    model = mnist.cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    return model, optimizer, scheduler


@ex.capture(prefix='cifar')
def model_cifar(device, num_classes, num_iter_warmup, num_iter_total, lr, momentum, weight_decay, _log):
    model = resnet.resnet18(dim_output=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        _log.info(f'using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return model, optimizer, scheduler


@ex.capture
def get_model(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist', 'kmnist']:
        model, optimizer, scheduler = model_mnist()
    elif dataset_name in ['cifar10', 'cifar100']:
        model, optimizer, scheduler = model_cifar()
    else:
        raise ValueError
    return model, optimizer, scheduler


@ex.capture(prefix='categorical')
def categorical_transition(device, num_classes, num_iter_warmup, num_iter_total, diagonal, off_diagonal, lr):
    init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).to(device)
    optim_matrix = lambda params: optim.Adam(params, lr=lr)
    lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
    sched_matrix = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return CategoricalTransition(init_matrix, optim_matrix, sched_matrix)


@ex.capture(prefix='dirichlet')
def dirichlet_transition(device, num_classes, diagonal, off_diagonal, betas):
    init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).to(device)
    return DirichletTransition(init_matrix, betas)


@ex.capture
def get_transition(transition_type):
    transition = {
        'none': NoTransition,
        'categorical': categorical_transition,
        'dirichlet': dirichlet_transition
    }[transition_type]()
    return transition


@ex.capture
def get_regularization(regularization_type, num_pairs, gamma):
    regularization = {
        'none': no_regularization,
        'tv': tv_regularization(num_pairs),
    }[regularization_type]
    return regularization, gamma


@ex.capture
def run(model, transition, scheduler, loader_tr, loader_ts, train_step,  # train & test
        num_iter_total, num_iter_test,  # configs
        _run, _log):
    transition_matrix = get_transition_matrix()
    accuracy, tv = 0., 1.

    start = toc = datetime.now()
    for it, (x, y) in enumerate(take_cycle(num_iter_total + 1, loader_tr)):
        # test
        if it % num_iter_test == 0 or it == num_iter_total:
            t, z = predict(model, loader_ts)
            accuracy = z.eq(t.argmax(dim=1)).sum().item() / len(z)

            # log
            if (m := transition.matrix) is not None:
                m = m.detach().cpu().numpy()
                tv = 0.5 * np.abs(m - transition_matrix).sum(axis=1).mean()
                with np.printoptions(precision=2, suppress=True):
                    _log.info(f' total variation {tv:.4f}\n{m}')

            tic, toc = toc, datetime.now()
            _run.log_scalar('accuracy', accuracy, it)
            _run.log_scalar('tv', tv, it)
            _log.info(
                f' iter [{it:>4}/{num_iter_total:>4}]'
                f' time [{str(toc - tic)[2:-7]}/{str(toc - start)[2:-7]}]'
                f' lr {scheduler.get_last_lr()[0]:.8f}'
                f' accuracy {accuracy:.2%}'
            )

        # train
        train_step(x, y)
        scheduler.step()

    # final matrix
    if (m := transition.matrix) is not None:
        m = m.detach().cpu().numpy()
        _run.log_scalar('matrix', m.tolist())


@ex.main
def main(_log):
    tic = datetime.now()
    dataset_tr, dataset_ts = get_datasets()
    dataset_tr = add_noise(dataset_tr)
    toc = datetime.now()
    _log.info(f'loaded dataset [{str(toc - tic)[2:-7]}]')

    loader_tr, loader_ts = get_loaders(dataset_tr, dataset_ts)
    model, optimizer, scheduler = get_model()
    transition = get_transition()
    regularization, gamma = get_regularization()

    # train & test
    train_step = get_train_step(model, transition, optimizer, regularization, gamma)
    run(model, transition, scheduler, loader_tr, loader_ts, train_step)


if __name__ == '__main__':
    ex.run_commandline()
