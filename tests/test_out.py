# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys

import wbml.out as out

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok


class RecordingStream(object):
    """A stream that records its writes."""

    def __init__(self):
        self.writes = []

    def write(self, msg):
        self.writes.append(msg)

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


def test_Section():
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

    yield eq, str(mock), 'message\n'


def test_format():
    class A(object):
        def __str__(self):
            return 'FormattedA()'

    yield eq, out._format(A()), 'FormattedA()'

    # Test formatting of floats.
    yield eq, out._format(0.000012), '1.200e-05'
    yield eq, out._format(0.00012), '1.200e-04'
    yield eq, out._format(0.0012), '0.0012'
    yield eq, out._format(0.012), '0.012'
    yield eq, out._format(0.12), '0.12'
    yield eq, out._format(1.2), '1.2'
    yield eq, out._format(12.0), '12.0'
    yield eq, out._format(120.0), '120.0'
    yield eq, out._format(1200.0), '1.200e+03'
    yield eq, out._format(12000.0), '1.200e+04'

    # Test formatting of integers.
    yield eq, out._format(1), '1'
    yield eq, out._format(1200), '1200'
    yield eq, out._format(12000), '12000'

    # Test formatting of containers.
    yield eq, out._format([1, 1.0, 1000.0]), '[1, 1.0, 1.000e+03]'
    yield eq, out._format((1, 1.0, 1000.0)), '(1, 1.0, 1.000e+03)'
    yield eq, out._format({1}), '{1}'
    yield eq, out._format({1.0}), '{1.0}'
    yield eq, out._format({1000.0}), '{1.000e+03}'


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
