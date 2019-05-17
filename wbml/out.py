# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import sys

import lab as B
from plum import Dispatcher

__all__ = ['Section', 'out', 'kv']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

indent = 0  #: Indentation level.
indent_width = 4  #: Number of indentation characters.
indent_char = ' '  #: Indentation character.
stream = sys.stdout  #: Output stream.
key_width = 10  #: Default width for printing keys.


def _print(msg):
    """Print a line with proper indentation.

    Args:
        msg (str): Message to print.
    """
    global indent
    global indent_width
    global indent_char
    global stream
    stream.write('{}{}\n'.format(indent_char * indent_width * indent, msg))


class Section(object):
    """Create a section in which output is indented.

    Args:
        name (str): Name of the section.
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        # Print the section name, if one is given.
        if self.name is not None:
            _print('{}:'.format(self.name))

        # Increase indentation.
        global indent
        indent += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Decrease indentation.
        global indent
        indent -= 1


def out(msg):
    """Print a message.

    Args:
        msg (str): Message to print.
    """
    _print(msg)


@_dispatch(object, object)
def kv(key, value):
    """Print a key-value pair.

    Args:
        key (str): Key.
        value (object): Value.
    """
    global key_width
    f = '{{key:{width}}} {{value}}'.format(width=key_width + 1)
    _print(f.format(key=_format(key) + ':', value=_format(value)))


@_dispatch(dict)
def kv(dict_):
    for k, v in dict_.items():
        kv(k, v)


@_dispatch(object, dict)
def kv(key, dict_):
    with Section(key):
        for k, v in dict_.items():
            kv(k, v)


@_dispatch(object)
def _format(x):
    """Format an object.

    Args:
        x (object): Object to format.

    Returns:
        str: `x` as a string.
    """
    return str(x)


@_dispatch(B.Number)
def _format(x):
    out = '{:.3e}'.format(x)

    # If `x` is not too large, print it as a float instead.
    if 1e-3 < B.abs(x) < 1e3:
        return str(float(out))

    return out


@_dispatch(B.Int)
def _format(x):
    return str(x)


@_dispatch(list)
def _format(xs):
    return '[{}]'.format(', '.join([_format(x) for x in xs]))


@_dispatch(tuple)
def _format(xs):
    return '({})'.format(', '.join([_format(x) for x in xs]))


@_dispatch(set)
def _format(xs):
    return '{{{}}}'.format(', '.join([_format(x) for x in xs]))
