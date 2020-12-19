import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection

aspect_ratio = np.sqrt(3) / 2
# simplex [3, 2]
Ts = np.array([[1 / 2, np.sqrt(3) / 2], [0, 0], [1, 0]])
# color [3, 3]
Tc = np.array(list(map(mpl.colors.to_rgb, ['xkcd:red', 'xkcd:green', 'xkcd:blue'])))


def grid(k: int, n: int) -> [[int]]:
    # grid on a (k-1)-simplex
    # x_1 + x_2 + ... + x_k = n
    if k == 1:
        return [[n]]
    else:
        return [[x] + y for x in range(n + 1) for y in grid(k - 1, n - x)]


def init(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(*Ts.T, alpha=0)
    ax.set_aspect(aspect_ratio / ax.get_data_ratio(), adjustable='box')


def lines(ax, segments, **kwargs):
    default = dict(linewidths=3, colors='k')
    ax.add_collection(LineCollection(segments, **{**default, **kwargs}))


def boundary(ax, **kwargs):
    default = dict(linewidths=5)
    lines(ax, [np.concatenate((Ts, Ts[:2]))], **{**default, **kwargs})


def scatter(ax, p, **kwargs):
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = p @ Tc
    default = dict(s=40, edgecolor='none')
    ax.scatter(*(p @ Ts).T, **{**default, **kwargs})


def tricontour(ax, p, z, **kwargs):
    default = dict(linewidths=3)
    ax.tricontour(*(p @ Ts).T, z, **{**default, **kwargs})


def tricontourf(ax, p, z, **kwargs):
    ax.tricontourf(*(p @ Ts).T, z, **kwargs)
