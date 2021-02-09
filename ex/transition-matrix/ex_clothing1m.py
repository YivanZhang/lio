import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

import torch
from torch import nn, optim, distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms

from core import (
    NoTransition, DirichletTransition,
    no_regularization, tv_regularization,
)
from utils import diag_matrix, take_cycle

ex = Experiment('clothing1m')


@ex.config
def config():
    data_dir = 'data/clothing1m'
    model_dir = 'model/clothing1m'

    folder_tr = 'noisy_train'
    folder_ts = 'clean_test'

    num_classes = 14
    batch_size = 32
    num_workers = 4 * torch.cuda.device_count()

    num_iter_total = 5000
    num_iter_warmup = 500
    num_iter_test = int(num_iter_total / 10)

    # model
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-3
    step_size = int(num_iter_total / 2)

    # transition
    transition_type = 'none'
    diagonal = 1.
    off_diagonal = 0.
    betas = (0.999, 0.01)

    # regularization
    regularization_type = 'none'
    num_pairs = batch_size
    gamma = 0.1

    # load & save
    load_name = 'pretrained.pth' if transition_type == 'none' else f'none.pth'
    save_name = f'{transition_type}.pth'
    load_path = f'{model_dir}/{load_name}'
    save_path = f'{model_dir}/{save_name}'


@ex.command
def download_model(model_dir):
    from torch.hub import load_state_dict_from_url
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    state_dict = load_state_dict_from_url(url, model_dir=model_dir, progress=True)
    for k in ['fc.weight', 'fc.bias']:
        state_dict.pop(k)
    torch.save(state_dict, Path(model_dir) / 'pretrained.pth')
    os.remove(Path(model_dir) / 'resnet50-19c8e357.pth')


@ex.capture
def get_datasets(data_dir, folder_tr, folder_ts):
    transform_tr = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_ts = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset_tr = ImageFolder(Path(data_dir) / folder_tr, transform=transform_tr)
    dataset_ts = ImageFolder(Path(data_dir) / folder_ts, transform=transform_ts)
    return dataset_tr, dataset_ts


@ex.capture
def get_loaders(dataset_tr, dataset_ts,
                batch_size, num_workers):
    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    loader_tr = DataLoader(dataset_tr, sampler=DistributedSampler(dataset_tr), **loader_kwargs)
    loader_ts = DataLoader(dataset_ts, sampler=DistributedSampler(dataset_ts), **loader_kwargs)
    return loader_tr, loader_ts


@ex.capture
def get_model(num_classes, load_path,  # model
              lr, momentum, weight_decay, step_size,  # optimizer & scheduler
              rank, local_rank,  # rank
              _log):
    model = resnet50()
    model.fc = nn.Linear(2048, num_classes)
    # load
    if load_path is not None and rank == 0:
        _log.info(f'loading {load_path}')
        model.load_state_dict(torch.load(load_path), strict=False)
    # data parallel
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return model, optimizer, scheduler


@ex.capture
def get_transition(transition_type, num_classes, diagonal, off_diagonal, betas):
    if transition_type == 'none':
        transition = NoTransition()
    elif transition_type == 'dirichlet':
        init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).cuda()
        transition = DirichletTransition(init_matrix, betas)
    else:
        raise ValueError
    return transition


@ex.capture
def get_regularization(regularization_type, num_pairs, gamma):
    regularization = {
        'none': no_regularization,
        'tv': tv_regularization(num_pairs),
    }[regularization_type]
    return regularization, gamma


def get_train_step(model, transition, optimizer, regularization=None, gamma=0.):
    if regularization is None or gamma == 0.:
        regularization, gamma = no_regularization, 0.

    def step(x, y):
        model.train()
        # device
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        # forward
        t = model(x)
        loss = transition.loss(t, y) - gamma * regularization(t)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # optimization
        optimizer.step()  # optimize model
        transition.update(t, y)  # optimize transition
        if (params := transition.params) is not None:
            dist.all_reduce(params.data)
            params.data /= dist.get_world_size()  # average

    return step


def get_accuracy(model, loader):
    model.eval()
    num_correct = torch.tensor(0).cuda()
    with torch.no_grad():
        for x, z in loader:
            x = x.cuda(non_blocking=True)
            z = z.cuda(non_blocking=True)
            num_correct += z.eq(model(x).argmax(dim=1)).sum()
    dist.all_reduce(num_correct)
    accuracy = num_correct.item() / len(loader.dataset)
    return accuracy


@ex.capture
def save(model,
         save_path, rank, _log):
    if save_path is not None and rank == 0:
        _log.info(f'saving {save_path}')
        torch.save(model.module.state_dict(), save_path)


@ex.capture
def run(model, transition, scheduler, loader_tr, loader_ts, train_step,  # train & test
        num_iter_total, num_iter_test,  # configs
        rank,
        _run, _log):
    start = toc = datetime.now()
    for it, (x, y) in enumerate(take_cycle(num_iter_total + 1, loader_tr)):
        # test
        if it % num_iter_test == 0 or it == num_iter_total:
            accuracy = get_accuracy(model, loader_ts)

            # log
            if rank == 0:
                if (m := transition.matrix) is not None:
                    m = m.detach().cpu().numpy()
                    with np.printoptions(precision=2, suppress=True):
                        _log.info(f'\n{m}')

                tic, toc = toc, datetime.now()
                _run.log_scalar('accuracy', accuracy, it)
                _log.info(
                    f' iter [{it:>4}/{num_iter_total:>4}]'
                    f' time [{str(toc - tic)[2:-7]}/{str(toc - start)[2:-7]}]'
                    f' lr {scheduler.get_last_lr()[0]:.8f}'
                    f' accuracy {accuracy:.2%}'
                )

        # train
        train_step(x, y)
        scheduler.step()

    if rank == 0:
        # save
        save(model)

        # final matrix
        if (m := transition.matrix) is not None:
            m = m.detach().cpu().numpy()
            _run.log_scalar('matrix', m.tolist())


@ex.main
def main():
    dataset_tr, dataset_ts = get_datasets()
    loader_tr, loader_ts = get_loaders(dataset_tr, dataset_ts)
    model, optimizer, scheduler = get_model()
    transition = get_transition()
    regularization, gamma = get_regularization()

    # train & test
    train_step = get_train_step(model, transition, optimizer, regularization, gamma)
    run(model, transition, scheduler, loader_tr, loader_ts, train_step)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--log_dir', type=str, default='log-c/clothing1m')
    args, other_args = parser.parse_known_args()
    ex.add_config({'log_dir': args.log_dir})

    if args.local_rank is not None:
        # initialization
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        # add config
        rank = dist.get_rank()
        ex.add_config({'local_rank': args.local_rank, 'rank': rank})
        # observer
        if rank == 0:
            ex.observers.append(FileStorageObserver(args.log_dir))
        else:
            other_args.append('--unobserved')
    # run
    ex.run_commandline([parser.prog] + other_args)
