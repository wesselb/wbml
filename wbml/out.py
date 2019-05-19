# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import sys

import lab as B
import numpy as np
from plum import Dispatcher

__all__ = ['Section', 'out', 'kv', 'format', 'Counter']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

indent_level = 0  #: Indentation level.
indent_width = 4  #: Number of indentation characters per level.
indent_char = ' '  #: Indentation character.
key_width = 10  #: Minimum width for printing keys, for alignment.
stream = sys.stdout  #: Output stream.
tf_session = None  #: TensorFlow session to run tensors, for formatting.


def _print(msg, line_end='\n'):
    """Print a line with proper indentation.

    Args:
        msg (str): Message to print.
        line_end (str, optional): String to print at the end of the line.
            Defaults to "\n".
    """
    global indent_level
    global indent_width
    global indent_char
    global stream

    # Construct total indentation.
    indent = indent_char * indent_width * indent_level

    # Indent the message and all newlines therein.
    msg = indent + msg.replace('\n', '\n' + indent)

    # Write the message plus line end.
    stream.write(msg + line_end)


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
        global indent_level
        indent_level += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Decrease indentation.
        global indent_level
        indent_level -= 1


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

    # Construct the format.
    f = '{{key:{width}}} {{value}}'.format(width=key_width + 1)

    # Format the key and value.
    formatted_key = format(key)
    formatted_value = format(value)

    # If the value contains a newline, print it within a section.
    if '\n' in formatted_value:
        with Section(name=formatted_key):
            out(formatted_value)
    else:
        _print(f.format(key=formatted_key + ':', value=formatted_value))


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
def format(x):
    """Format an object.

    Args:
        x (object): Object to format.

    Returns:
        str: `x` as a string.
    """
    return str(x)


@_dispatch(B.Number)
def format(x):
    out = '{:.3e}'.format(x)

    # If `x` is not too large, print it as a float instead.
    if 1e-3 < B.abs(x) < 1e3:
        return str(float(out))

    return out


@_dispatch(B.Int)
def format(x):
    return str(x)


@_dispatch(list)
def format(xs):
    return '[{}]'.format(', '.join([format(x) for x in xs]))


@_dispatch(tuple)
def format(xs):
    return '({})'.format(', '.join([format(x) for x in xs]))


@_dispatch(set)
def format(xs):
    return '{{{}}}'.format(', '.join([format(x) for x in xs]))


@_dispatch(np.ndarray)
def format(x):
    return np.array_str(x, precision=3)


@_dispatch(B.TorchNumeric)
def format(x):
    return format(x.detach().numpy())


@_dispatch(B.TFNumeric)
def format(x):
    global tf_session

    if tf_session:
        return format(tf_session.run(x))
    else:
        return str(x)


class Counter(object):
    """A counter.

    Args:
        name (str, optional): Name of the counter. Defaults to no name.
        total (int, optional): Total number of counts. Defaults to no total.
    """

    def __init__(self, name=None, total=None):
        self.iteration = 0
        self.name = name
        self.total = total

    def __enter__(self):
        # Print the name, if one is given.
        title = self.name if self.name else 'Counting'

        # Print the total, if one is given.
        if self.total:
            title += ' (total: {})'.format(self.total)

        _print(title + ':', line_end='')
        stream.flush()  # Flush, because there is no newline.

        return self

    def count(self):
        """Count once."""
        self.iteration += 1
        stream.write(' {}'.format(self.iteration))
        stream.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        stream.write('\n')
        stream.flush()

    @staticmethod
    def map(f, xs, name=None):
        """Perform a mapping operation that is counted.

        Args:
            f (function): Function to apply.
            xs (iterable): Arguments to apply.
            name (str, optional): Name of the mapping operation. Defaults to
                "mapping".

        Returns:
            list: Result.
        """
        # Set default name.
        if name is None:
            name = 'Mapping'

        # Perform mapping operation.
        results = []
        with Counter(name=name, total=len(xs)) as counter:
            for x in xs:
                counter.count()
                results.append(f(x))
        return results
