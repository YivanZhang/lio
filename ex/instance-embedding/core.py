import torch
import torch.nn.functional as F
from torch import nn


def get_embedding(size: int):
    embed = nn.Embedding(size, 1, sparse=True)
    embed.weight.data.fill_(0.)
    embed = nn.Sequential(embed, nn.Sigmoid())
    return embed


def no_transformation(_, t):
    return t


def linear_interpolation(c, t):
    return torch.log(c * F.softmax(t, dim=1) + (1 - c) * 1 / t.shape[1])


def power_transformation(c, t):
    return c * t


# ----------------------------------------------------------------------------------------------------------------------

def get_train_step(
        model, optimizer_model,
        embed, optimizer_embed,
        transformation=None,
        loss=None,
):
    if transformation is None:
        transformation = no_transformation
    if loss is None:
        loss = F.cross_entropy
    device = next(model.parameters()).device

    def step(i, x, y):
        model.train()
        # device
        i = i.to(device)
        x = x.to(device)
        y = y.to(device)
        # forward
        c = embed(i)
        t = model(x)
        l = loss(transformation(c, t), y)
        # backward
        optimizer_model.zero_grad()
        optimizer_embed.zero_grad()
        l.backward()
        # optimization
        optimizer_model.step()  # optimize model
        optimizer_embed.step()  # optimize embed

    return step
