from typing import Optional, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Categorical

from lio.losses.classification import indirect_observation_loss
from lio.utils.metrics import confusion_matrix

default_activation = lambda t: F.softmax(t, dim=1)


class Transition:
    params: Optional[torch.Tensor] = None
    matrix: Optional[torch.Tensor] = None
    update: Callable[[torch.Tensor, torch.Tensor], None] = lambda *_: None
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class NoTransition(Transition):
    def __init__(self, loss: Callable = None):
        self.loss = F.cross_entropy if loss is None else loss


class FixedTransition(Transition):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.loss = indirect_observation_loss(matrix)


class CategoricalTransition(Transition):
    def __init__(self,
                 init_matrix: torch.Tensor,
                 optimizer: Callable,
                 scheduler: Callable = None,
                 activation_output: Callable = None,
                 activation_matrix: Callable = None,
                 ):
        self.logits = nn.Parameter(init_matrix, requires_grad=True)
        self.optimizer = optimizer([self.logits])
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None
        self.activation_output = default_activation if activation_output is None else activation_output
        self.activation_matrix = default_activation if activation_matrix is None else activation_matrix

    @property
    def params(self):
        return self.logits

    @property
    def matrix(self):
        return self.activation_matrix(self.logits)

    def loss(self, t, y):
        self.optimizer.zero_grad()  # no accumulated gradient
        p_z = self.activation_output(t)
        p_y = p_z @ self.matrix
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    def update(self, *_):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()


class DirichletTransition(Transition):
    def __init__(self,
                 init_matrix: torch.Tensor,
                 betas: Tuple[float, float],
                 activation_output: Callable = None,
                 ):
        self.concentrations = init_matrix
        self.betas = betas
        self.activation_output = default_activation if activation_output is None else activation_output

    @property
    def params(self):
        return self.concentrations

    @property
    def matrix(self):
        return self.concentrations / self.concentrations.sum(dim=1, keepdim=True)

    def _sample(self):
        return torch.stack([Dirichlet(c).sample() for c in self.concentrations])

    def loss(self, t, y):
        p_z = self.activation_output(t)
        p_y = p_z @ self._sample()
        return F.nll_loss(torch.log(p_y + 1e-32), y)

    def update(self, t, y):
        num_classes = self.concentrations.shape[0]
        # z = t.detach().argmax(dim=1)  # simplified version using argmax
        z = Categorical(probs=self.activation_output(t.detach())).sample()
        m = confusion_matrix(z, y, n1=num_classes, n2=num_classes)
        self.concentrations *= self.betas[0]  # decay
        self.concentrations += self.betas[1] * m  # update


# ----------------------------------------------------------------------------------------------------------------------

def no_regularization(*_):
    return 0


def tv_regularization(num_pairs: int, activation_output: Callable = None):
    activation_output = default_activation if activation_output is None else activation_output

    def reg(t: torch.Tensor):
        p = activation_output(t)
        idx1, idx2 = torch.randint(0, t.shape[0], (2, num_pairs)).to(t.device)
        tv = 0.5 * (p[idx1] - p[idx2]).abs().sum(dim=1).mean()
        return tv

    return reg


# ----------------------------------------------------------------------------------------------------------------------

def get_train_step(model, transition, optimizer, regularization=None, gamma=0.):
    if regularization is None or gamma == 0.:
        regularization, gamma = no_regularization, 0.
    device = next(model.parameters()).device

    def step(x, y):
        model.train()
        # device
        x = x.to(device)
        y = y.to(device)
        # forward
        t = model(x)
        l = transition.loss(t, y) - gamma * regularization(t)
        # backward
        optimizer.zero_grad()
        l.backward()
        # optimization
        optimizer.step()  # optimize model
        transition.update(t, y)  # optimize transition

    return step
