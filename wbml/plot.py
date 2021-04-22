from functools import wraps

import subprocess
import lab as B
import matplotlib.pyplot as plt
from plum import Dispatcher

__all__ = ["patch", "tex", "tweak", "style", "pdfcrop"]

_dispatch = Dispatcher()

plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.
plt.rcParams["figure.autolayout"] = True  # Use tight layouts.


@_dispatch
def _convert(x):
    return x


@_dispatch
def _convert(x: B.Number):
    return x


@_dispatch
def _convert(x: B.Numeric):
    return B.squeeze(B.to_numpy(x))


@_dispatch
def _convert(xs: tuple):
    return tuple(_convert(x) for x in xs)


@_dispatch
def _convert(xs: list):
    return [_convert(x) for x in xs]


@_dispatch
def _convert(d: dict):
    return {k: _convert(v) for k, v in d.items()}


def patch(f, kind=None):
    """Decorator to patch a function to automatically convert arguments that
    are of a framework type, like PyTorch, to NumPy. Also allows a keyword
    argument `style` to automatically expand the dictionary given by
    :func:`.plot.style` into the keyword arguments.
    """

    @wraps(f)
    def patched_f(*args, **kw_args):
        # Automatically expand style configuration.
        if kind and "style" in kw_args:
            for k, v in style(kw_args["style"], kind).items():
                if k not in kw_args:
                    kw_args[k] = v
            del kw_args["style"]

        return f(*_convert(args), **_convert(kw_args))

    return patched_f


# Patch common plotting functions.
plt.plot = patch(plt.plot, kind="line")
plt.scatter = patch(plt.scatter, kind="scatter")
plt.fill_between = patch(plt.fill_between, kind="fill")
plt.errorbar = patch(plt.errorbar)
plt.xlim = patch(plt.xlim)
plt.ylim = patch(plt.ylim)


def tex():
    """Use TeX for rendering."""
    plt.rcParams["text.usetex"] = True  # Use TeX for rendering.


def tweak(
    grid=True, legend=None, legend_loc="upper right", spines=True, ticks=True, ax=None
):
    """Tweak a plot.

    Args:
        grid (bool, optional): Show grid. Defaults to `True`.
        legend (bool, optional): Show legend. Automatically shows a legend if any labels
            are set.
        legend_loc (str, optional): Position of the legend. Defaults to
            "upper right".
        spines (bool, optional): Hide top and right spine. Defaults to `True`.
        ticks (bool, optional): Hide top and right ticks. Defaults to `True`.
        ax (axis, optional): Axis to tune. Defaults to `plt.gca()`.
    """
    if ax is None:
        ax = plt.gca()

    if grid:
        ax.set_axisbelow(True)  # Show grid lines below other elements.
        ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)

    if legend is None:
        legend = len(ax.get_legend_handles_labels()[0]) > 0

    if legend:
        leg = ax.legend(
            facecolor="#eeeeee", framealpha=0.7, loc=legend_loc, labelspacing=0.25
        )
        leg.get_frame().set_linewidth(0)

    if spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_lw(1)
        ax.spines["left"].set_lw(1)

    if ticks:
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(width=1)


scheme = [
    "#000000",  # Black
    "#F5793A",  # Orange
    "#4BA6FB",  # Modified blue (original was #85C0F9)
    "#A95AA1",  # Pink
]
"""list[str]: Color scheme."""

colour_map = {
    "train": scheme[0],
    "test": scheme[1],
    "pred": scheme[2],
    "pred2": scheme[3],
}
"""dict[str, str]: Name to colour mapping."""

line_style_map = {"train": "-", "test": "-", "pred": "--", "pred2": "-."}
"""dict[str, str]: Name to line style mapping."""

scatter_style_map = {"train": "o", "test": "x", "pred": "s", "pred2": "D"}
"""dict[str, str]: Name to scatter style mapping."""


def style(name, kind="line"):
    """Generate style setting for functions in :mod:`matplotlib.pyplot`.

    Args:
        name (str): Name of style.
        kind ('line', 'scatter', or 'fill'): Kind of settings.

    Returns:
        dict: Style settings.
    """
    if kind == "line":
        return {"c": colour_map[name], "ls": line_style_map[name]}
    elif kind == "scatter":
        return {"c": colour_map[name], "marker": scatter_style_map[name], "s": 8}
    elif kind == "fill":
        return {"facecolor": colour_map[name], "alpha": 0.25}
    else:
        return ValueError(f'Unknown kind "{kind}".')


def pdfcrop(path):
    """Run pdfcrop on a PDF.

    Args:
        path (str): Path of PDF.
    """
    subprocess.call(["pdfcrop", path, path])