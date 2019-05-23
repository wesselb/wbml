# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
import numpy as np

import lab as B
import tensorflow as tf
import torch
import wbml.out as out

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, approx


class RecordingStream(object):
    """A stream that records its writes."""

    def __init__(self):
        self.writes = []

    def write(self, msg):
        self.writes.append(msg)

    def flush(self):
        self.writes.append('[flush]')

    def __str__(self):
        return ''.join(self.writes)

    def __len__(self):
        return len(self.writes)


class Mock(object):
    """Mock the stream that `wbml.out` uses."""

    def __enter__(self):
        self.stream = RecordingStream()
        out.stream = self.stream  # Mock stream.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        out.stream = sys.stdout  # Unmock stream.

    def __str__(self):
        return str(self.stream)

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, i):
        return self.stream.writes[i]


def test_section():
    with Mock() as mock:
        out.out('before')

        with out.Section():
            out.out('message1')

        with out.Section('name'):
            out.out('message2')

            with out.Section():
                out.out('message3')

        out.out('after')

    yield eq, len(mock), 6
    yield eq, mock[0], 'before\n'
    yield eq, mock[1], '    message1\n'
    yield eq, mock[2], 'name:\n'
    yield eq, mock[3], '    message2\n'
    yield eq, mock[4], '        message3\n'
    yield eq, mock[5], 'after\n'


def test_out():
    with Mock() as mock:
        out.out('message')

    yield eq, len(mock), 1
    yield eq, str(mock), 'message\n'

    # Test that newlines are correctly indented.
    with Mock() as mock:
        out.out('a\nb')

        with out.Section():
            out.out('c\nd')

    yield eq, len(mock), 2
    yield eq, mock[0], 'a\nb\n'
    yield eq, mock[1], '    c\n    d\n'


def test_kv():
    with Mock() as mock:
        out.kv('key', 'value')
        out.kv(1.0, 1.0)
        out.kv(10.0, 10.0)
        out.kv(1000.0, 1000.0)
        out.kv(1000, 1000)

    yield eq, len(mock), 5
    yield eq, mock[0], 'key:        value\n'
    yield eq, mock[1], '1.0:        1.0\n'
    yield eq, mock[2], '10.0:       10.0\n'
    yield eq, mock[3], '1.000e+03:  1.000e+03\n'
    yield eq, mock[4], '1000:       1000\n'

    # Test giving a dictionary.

    with Mock() as mock:
        out.kv({'level1': {'level2': {1: 1}}})

    yield eq, len(mock), 3
    yield eq, mock[0], 'level1:\n'
    yield eq, mock[1], '    level2:\n'
    yield eq, mock[2], '        1:          1\n'

    # Test giving a key and a dictionary.

    with Mock() as mock:
        out.kv('dict', {1: 1})

    yield eq, len(mock), 2
    yield eq, mock[0], 'dict:\n'
    yield eq, mock[1], '    1:          1\n'

    # Test values with newlines.

    with Mock() as mock:
        out.kv('a', 'b\nc')

    yield eq, len(mock), 2
    yield eq, mock[0], 'a:\n'
    yield eq, mock[1], '    b\n    c\n'


def test_format():
    class A(object):
        def __str__(self):
            return 'FormattedA()'

    yield eq, out.format(A()), 'FormattedA()'

    # Test formatting of floats.
    yield eq, out.format(0.000012), '1.200e-05'
    yield eq, out.format(0.00012), '1.200e-04'
    yield eq, out.format(0.0012), '0.0012'
    yield eq, out.format(0.012), '0.012'
    yield eq, out.format(0.12), '0.12'
    yield eq, out.format(1.2), '1.2'
    yield eq, out.format(12.0), '12.0'
    yield eq, out.format(120.0), '120.0'
    yield eq, out.format(1200.0), '1.200e+03'
    yield eq, out.format(12000.0), '1.200e+04'

    # Test formatting of integers.
    yield eq, out.format(1), '1'
    yield eq, out.format(1200), '1200'
    yield eq, out.format(12000), '12000'

    # Test formatting of containers.
    yield eq, out.format([1, 1.0, 1000.0]), '[1, 1.0, 1.000e+03]'
    yield eq, out.format((1, 1.0, 1000.0)), '(1, 1.0, 1.000e+03)'
    yield eq, out.format({1}), '{1}'
    yield eq, out.format({1.0}), '{1.0}'
    yield eq, out.format({1000.0}), '{1.000e+03}'

    # Test formatting of NumPy objects.
    yield eq, out.format(B.ones(int, 3)), '[1 1 1]'
    yield eq, out.format(B.ones(int, 3, 3)), \
          '(3x3 array of data type int64)\n[[1 1 1]\n [1 1 1]\n [1 1 1]]'

    # Test formatting of PyTorch objects.
    yield eq, out.format(B.ones(torch.int, 3)), '[1 1 1]'

    # Test formatting of TensorFlow objects.
    ones = B.ones(tf.int32, 3)
    yield eq, out.format(ones), str(ones)
    out.tf_session = tf.Session()
    yield eq, out.format(ones), '[1 1 1]'
    out.tf_session.close()
    out.tf_session = None


def test_counter():
    def yield_counts(mock_):
        yield eq, len(mock_), 8
        yield eq, mock_[1], '[flush]'
        yield eq, mock_[2], ' 1'
        yield eq, mock_[3], '[flush]'
        yield eq, mock_[4], ' 2'
        yield eq, mock_[5], '[flush]'
        yield eq, mock_[6], '\n'
        yield eq, mock_[7], '[flush]'

    # Test normal application.

    with Mock() as mock:
        with out.Counter() as counter:
            counter.count()
            counter.count()

    yield eq, mock[0], 'Counting:'
    for x in yield_counts(mock):
        yield x

    with Mock() as mock:
        with out.Counter(name='name', total=3) as counter:
            counter.count()
            counter.count()

    yield eq, mock[0], 'name (total: 3):'
    for x in yield_counts(mock):
        yield x

    # Test mapping.

    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3])

    yield eq, res, [4, 9]
    yield eq, mock[0], 'Mapping (total: 2):'
    for x in yield_counts(mock):
        yield x

    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3], name='name')

    yield eq, res, [4, 9]
    yield eq, mock[0], 'name (total: 2):'
    for x in yield_counts(mock):
        yield x


def test_compute_alpha():
    lags = 8
    alpha = out._compute_alpha(lags)
    x = np.sin(2 * np.pi / lags * B.range(10000))

    # Perform filtering.
    y = [x[0]]
    for xi in x:
        y.append(alpha * xi + (1 - alpha) * y[-1])
    y = np.array(y)

    # Check damping in dB.
    ratio = 10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2))
    yield approx, ratio, -3, 2

    # Check not setting the cut-off.
    yield eq, out._compute_alpha(None), 1


def test_progress():
    # Test a simple case.

    with Mock() as mock:
        with out.Progress() as progress:
            progress(a=1)

    yield eq, len(mock), 4
    yield eq, mock[0], 'Progress:\n'
    yield eq, mock[1], '    Iteration 1:\n'
    yield eq, mock[2], '        a:          1\n'
    yield eq, mock[3], '    Done!\n'

    # Test setting the total number of iterations and report interval as int.

    with Mock() as mock:
        with out.Progress(name='name', total=4, interval=3) as progress:
            progress(a='a')
            progress(a='b')
            progress(a='c')
            # Change time per iteration to a second to show a non-zero time
            # left.
            progress.values['_time_per_it'] = 1
            progress(a='d')

    yield eq, len(mock), 11
    yield eq, mock[0], 'name:\n'
    yield eq, mock[1], '    Iteration 1/4:\n'  # First is always shown.
    yield ok, 'Time left' in mock[2]
    yield eq, mock[3], '        a:          a\n'
    yield eq, mock[4], '    Iteration 3/4:\n'
    yield ok, 'Time left' in mock[5]
    yield eq, mock[6], '        a:          c\n'
    yield eq, mock[7], '    Iteration 4/4:\n'  # Last is also always shown.
    yield ok, 'Time left' in mock[8]
    yield ok, '0.0 s' not in mock[8]
    yield eq, mock[9], '        a:          d\n'
    yield eq, mock[10], '    Done!\n'

    # Test filters, report interval as float, and giving a dictionary.

    with Mock() as mock:
        with out.Progress(name='name',
                          interval=1e-10,
                          filter={'a': None},
                          filter_global=np.inf) as progress:
            progress({'a': 1, 'b': 1})
            progress({'a': 2, 'b': 2})

    yield eq, len(mock), 8
    yield eq, mock[0], 'name:\n'
    yield eq, mock[1], '    Iteration 1:\n'
    yield eq, mock[2], '        a:          1\n'
    yield eq, mock[3], '        b:          1\n'
    yield eq, mock[4], '    Iteration 2:\n'
    yield eq, mock[5], '        a:          2\n'  # Filter should be off.
    yield eq, mock[6], '        b:          1.0\n'  # Filter should be maximal.
    yield eq, mock[7], '    Done!\n'

    # Test mapping.

    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3])

    yield eq, res, [4, 9]
    yield eq, len(mock), 6
    yield eq, mock[0], 'Mapping:\n'

    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3], name='name')

    yield eq, res, [4, 9]
    yield eq, len(mock), 6
    yield eq, mock[0], 'name:\n'
