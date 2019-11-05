import logging

import matplotlib.pyplot as plt

__all__ = ['tex', 'tweak']

log = logging.getLogger(__name__)

plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'  # Use CM for math font.
plt.rcParams['figure.autolayout'] = True  # Use tight layouts.


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
