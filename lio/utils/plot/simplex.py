import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection, PolyCollection

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
    return ax.add_collection(LineCollection([s @ Ts for s in segments], **{**default, **kwargs}))


def polygon(ax, p, **kwargs):
    default = dict(linewidths=3, edgecolors='k', facecolors='None')
    return ax.add_collection(PolyCollection([p @ Ts], **{**default, **kwargs}))


def boundary(ax, **kwargs):
    default = dict(linewidths=5)
    return polygon(ax, np.eye(3, 3), **{**default, **kwargs})


def arrow(ax, start, end, **kwargs):
    start = start @ Ts
    end = end @ Ts
    default = dict(width=0.01, head_length=0.03, head_width=0.03, length_includes_head=True,
                   edgecolor='None', facecolor='k')
    return ax.arrow(*start, *(end - start), **{**default, **kwargs})


def scatter(ax, p, **kwargs):
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = p @ Tc
    if 'c' in kwargs:
        kwargs['cmap'] = 'magma'
    default = dict(s=40, edgecolor='none')
    return ax.scatter(*(p @ Ts).T, **{**default, **kwargs})


def text(ax, p, s, **kwargs):
    default = dict(fontsize=20, ha='center', va='center')
    return ax.text(*(p @ Ts), s, **{**default, **kwargs})


def tricontour(ax, p, z, **kwargs):
    if 'colors' not in kwargs and 'cmap' not in kwargs:
        kwargs['cmap'] = 'magma'
    default = dict(linewidths=3)
    return ax.tricontour(*(p @ Ts).T, z, **{**default, **kwargs})


def tricontourf(ax, p, z, **kwargs):
    if 'colors' not in kwargs and 'cmap' not in kwargs:
        kwargs['cmap'] = 'magma'
    default = dict()
    return ax.tricontourf(*(p @ Ts).T, z, **{**default, **kwargs})


def quiver(ax, p, d, **kwargs):
    Td = Ts - np.array([1 / 2, np.sqrt(3) / 6])
    default = dict()
    return ax.quiver(*(p @ Ts).T, *(d @ Td).T, **{**default, **kwargs})
