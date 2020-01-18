from functools import wraps

import lab as B
import matplotlib.pyplot as plt
from plum import Dispatcher

__all__ = ['patch', 'tex', 'tweak']

_dispatch = Dispatcher()

plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'  # Use CM for math font.
plt.rcParams['figure.autolayout'] = True  # Use tight layouts.


@_dispatch(object)
def _convert(x):
    return x


@_dispatch(B.Numeric)
def _convert(x):
    return B.squeeze(B.to_numpy(x))


@_dispatch(tuple)
def _convert(xs):
    return tuple(_convert(x) for x in xs)


@_dispatch(list)
def _convert(xs):
    return [_convert(x) for x in xs]


@_dispatch(dict)
def _convert(d):
    return {k: _convert(v) for k, v in d.items()}


def patch(f):
    """Decorator to patch a function to automatically convert arguments that
    are of a framework type, like PyTorch, to NumPy.
    """

    @wraps(f)
    def patched_f(*args, **kw_args):
        return f(*_convert(args), **_convert(kw_args))

    return patched_f


# Patch common plotting functions.
plt.plot = patch(plt.plot)
plt.scatter = patch(plt.scatter)
plt.fill_between = patch(plt.fill_between)
plt.errorbar = patch(plt.errorbar)
plt.xlim = patch(plt.xlim)
plt.ylim = patch(plt.ylim)


def tex():
    """Use TeX for rendering."""
    plt.rcParams['text.usetex'] = True  # Use TeX for rendering.


def tweak(grid=True, legend=True, spines=True, ticks=True):
    """Tweak a plot.

    Args:
        grid (bool, optional): Show grid. Defaults to `True`.
        legend (bool, optional): Show legend. Defaults to `True`.
        spines (bool, optional): Hide top and right spine. Defaults to `True`.
        ticks (bool, optional): Hide top and right ticks. Defaults to `True`.
    """
    ax = plt.gca()

    if grid:
        ax.set_axisbelow(True)  # Show grid lines below other elements.
        plt.grid(which='major', c='#c0c0c0', alpha=.5, lw=1)

    if legend:
        leg = plt.legend(facecolor='#eeeeee',
                         framealpha=0.7,
                         loc='upper right',
                         labelspacing=0.25)
        leg.get_frame().set_linewidth(0)

    if spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_lw(1)
        ax.spines['left'].set_lw(1)

    if ticks:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(width=1)
