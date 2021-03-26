from datetime import datetime
from functools import lru_cache

import numpy as np
from sacred import Experiment

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from lio.models import mnist, resnet
from lio.observations import get_identity, get_symmetric_noise, get_random_noise, observe_categorical
from lio.utils.data import IndexDataset, load_all_data, load_mnist, load_svhn, load_cifar, LioDataset
from lio.utils.metrics import predict

from core import (
    get_embedding, get_train_step,
    no_transformation, linear_interpolation, power_transformation,
)
from utils import get_device, take_cycle

ex = Experiment('instance-embedding-image')


@ex.config
def config():
    # data
    dataset_name = 'mnist'
    if 'mnist' in dataset_name:
        data_dir = 'data/mnist'
    elif 'svhn' in dataset_name:
        data_dir = 'data/svhn'
    elif 'cifar' in dataset_name:
        data_dir = 'data/cifar'
    split_ratio = 0.8
    noise_type = 'symm'
    num_classes = 10 if dataset_name != 'cifar100' else 100
    dim_output = num_classes

    # loaders
    batch_size = 512
    num_workers = 4 * torch.cuda.device_count()

    # iterations
    if 'cifar' in dataset_name:
        num_iter_total = 4000
    elif 'mnist' in dataset_name:
        num_iter_total = 2000
    else:
        num_iter_total = 1000
    num_iter_warmup = 400
    num_iter_test = int(num_iter_total / 10)

    # model
    device = get_device()
    mnist = dict(
        device=device,
        lr=1e-3, lr_decay=0.1 ** (1 / num_iter_total),
    )
    resnet = dict(
        device=device, num_classes=num_classes,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
        lr=0.1, momentum=0.9, weight_decay=1e-4,
    )

    # embedding
    embed = dict(
        device=device,
        lr=500,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
    )
    transformation_type = 'power'


@ex.capture
@lru_cache(maxsize=1)
def get_datasets(data_dir, dataset_name):
    if 'mnist' in dataset_name:
        dataset_tr, dataset_ts = tuple(TensorDataset(*load_all_data(dataset))
                                       for dataset in load_mnist(data_dir, dataset_name))
    elif 'svhn' in dataset_name:
        dataset_tr, dataset_ts = tuple(TensorDataset(*load_all_data(dataset))
                                       for dataset in load_svhn(data_dir))
    elif 'cifar' in dataset_name:
        dataset_tr, dataset_ts = load_cifar(data_dir, dataset_name)
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
    else:
        raise ValueError
    with np.printoptions(precision=2, suppress=True):
        _log.info(f'\n{transition_matrix}')
    return transition_matrix


@ex.capture
def add_noise(dataset, transition_matrix,
              noise_type, _run):
    if noise_type == 'clean':
        return dataset
    else:
        z = load_all_data(dataset)[1]
        indices = torch.arange(len(dataset))[:, None].long()
        y = observe_categorical(torch.tensor(transition_matrix))(z, indices)
        _run.log_scalar('z', z.numpy().tolist())
        _run.log_scalar('y', y.numpy().tolist())
        return LioDataset(dataset, indices, y)


@ex.capture
def split(dataset,
          split_ratio, _run):
    size_tr = int(split_ratio * len(dataset))
    size_vl = len(dataset) - size_tr
    dataset_tr, dataset_vl = random_split(dataset, [size_tr, size_vl])
    _run.log_scalar('idx_tr', dataset_tr.indices)
    _run.log_scalar('idx_vl', dataset_vl.indices)
    return dataset_tr, dataset_vl


@ex.capture
def get_loaders(dataset_tr, dataset_vl, dataset_ts,
                batch_size, num_workers):
    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    loader_tr = DataLoader(dataset_tr, shuffle=True, **loader_kwargs)
    loader_vl = DataLoader(dataset_vl, **loader_kwargs)
    loader_ts = DataLoader(dataset_ts, **loader_kwargs)
    return loader_tr, loader_vl, loader_ts


@ex.capture(prefix='mnist')
def model_mnist(dim_output,
                device, lr, lr_decay):
    model = mnist.cnn(dim_output=dim_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    return model, optimizer, scheduler


@ex.capture(prefix='resnet')
def model_resnet(dim_output,
                 device, lr, momentum, weight_decay, num_iter_warmup, num_iter_total, _log):
    model = resnet.resnet18(dim_output=dim_output).to(device)
    if torch.cuda.device_count() > 1:
        _log.info(f'using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return model, optimizer, scheduler


@ex.capture
def get_model(dataset_name, dim_output, device):
    if dataset_name in ['mnist', 'fashion-mnist', 'kmnist']:
        model, optimizer, scheduler = model_mnist(dim_output)
    elif dataset_name in ['svhn', 'cifar10', 'cifar100']:
        model, optimizer, scheduler = model_resnet(dim_output)
    else:
        raise ValueError
    model = nn.Sequential(model, nn.BatchNorm1d(dim_output).to(device))
    return model, optimizer, scheduler


@ex.capture(prefix='embed')
def get_embed(size,
              device, lr, num_iter_warmup, num_iter_total):
    embed = get_embedding(size).to(device)
    optimizer = optim.SGD(embed.parameters(), lr=lr)
    lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return embed, optimizer, scheduler


@ex.capture
def get_transformation(transformation_type):
    transformation = {
        'none': no_transformation,
        'linear': linear_interpolation,
        'power': power_transformation,
    }[transformation_type]
    return transformation


@ex.capture
def run(model, scheduler_model,
        embed, scheduler_embed,
        loader_tr, loader_vl, loader_ts, train_step, decision_function,
        # configs
        device, num_iter_total, num_iter_test,
        _run, _log):
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    indices = torch.arange(len(loader_tr.dataset)).long().to(device)

    start = toc = datetime.now()
    for it, (i, (x, y)) in enumerate(take_cycle(num_iter_total + 1, loader_tr)):
        # test
        if it % num_iter_test == 0 or it == num_iter_total:
            t, z = predict(model, loader_vl)
            acc_vl = z.eq(decision_function(t)).sum().item() / len(z)
            t, z = predict(model, loader_ts)
            acc_ts = z.eq(decision_function(t)).sum().item() / len(z)
            confidence = embed(indices).detach().cpu().numpy().squeeze()

            # log
            tic, toc = toc, datetime.now()
            _run.log_scalar('acc_vl', acc_vl, it)
            _run.log_scalar('acc_ts', acc_ts, it)
            _run.log_scalar('confidence', confidence.tolist(), it)
            _log.info(
                f' iter [{it:>4}/{num_iter_total:>4}]'
                f' time [{str(toc - tic)[2:-7]}/{str(toc - start)[2:-7]}]'
                f' lr {scheduler_model.get_last_lr()[0]:.2e} {scheduler_embed.get_last_lr()[0]:.2e}'
                f' acc {acc_vl:.2%} {acc_ts:.2%}'
            )
            _log.info(' quantile ' + ' '.join(f'{q:.2%}' for q in np.quantile(confidence, quantiles)))

        # train
        train_step(i, x, y)
        scheduler_model.step()
        scheduler_embed.step()

    # final prediction
    dataset_tr = loader_tr.dataset.dataset
    t = predict(model, DataLoader(dataset_tr, batch_size=1024))[0]
    prediction = t.argmax(dim=1).cpu().numpy()
    _run.log_scalar('prediction', prediction.tolist())


def datasets():
    dataset_tr, dataset_ts = get_datasets()
    transition_matrix = get_transition_matrix()
    dataset_tr = add_noise(dataset_tr, transition_matrix)
    dataset_tr, dataset_vl = split(dataset_tr)
    return dataset_tr, dataset_vl, dataset_ts


@ex.main
def main(_run):
    dataset_tr, dataset_vl, dataset_ts = datasets()
    loader_tr, loader_vl, loader_ts = get_loaders(IndexDataset(dataset_tr), dataset_vl, dataset_ts)
    model, optimizer_model, scheduler_model = get_model()
    embed, optimizer_embed, scheduler_embed = get_embed(len(dataset_tr))

    transformation = get_transformation()
    loss = F.cross_entropy
    decision_function = lambda t: t.argmax(dim=1)

    train_step = get_train_step(model, optimizer_model,
                                embed, optimizer_embed,
                                transformation, loss)
    run(model, scheduler_model,
        embed, scheduler_embed,
        loader_tr, loader_vl, loader_ts,
        train_step, decision_function)


if __name__ == '__main__':
    ex.run_commandline()
