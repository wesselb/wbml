import lab as B
import numpy as np
import tensorflow as tf
import torch

import wbml.out as out
from .util import approx


class RecordingStream:
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


class Mock:
    """Mock the stream that `wbml.out` uses."""

    def __enter__(self):
        self.stream = RecordingStream()
        self.saved_streams = list(out.streams)
        out.streams = [self.stream]  # Mock stream.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        out.streams = self.saved_streams  # Unmock stream.

    def __str__(self):
        return str(self.stream)

    def __repr__(self):
        return '<Mock: recorded=' + str(self.stream) + '>'

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

    assert len(mock) == 6
    assert mock[0] == 'before\n'
    assert mock[1] == '    message1\n'
    assert mock[2] == 'name:\n'
    assert mock[3] == '    message2\n'
    assert mock[4] == '        message3\n'
    assert mock[5] == 'after\n'


def test_out():
    with Mock() as mock:
        out.out('message')

    assert len(mock) == 1
    assert str(mock) == 'message\n'

    # Test that newlines are correctly indented.
    with Mock() as mock:
        out.out('a\nb')

        with out.Section():
            out.out('c\nd')

    assert len(mock) == 2
    assert mock[0] == 'a\nb\n'
    assert mock[1] == '    c\n    d\n'


def test_kv():
    with Mock() as mock:
        out.kv('key', 'value')
        out.kv(1.0, 1.0)
        out.kv(10.0, 10.0)
        out.kv(1000.0, 1000.0)
        out.kv(1000, 1000)

    assert len(mock) == 5
    assert mock[0] == 'key:        value\n'
    assert mock[1] == '1.0:        1.0\n'
    assert mock[2] == '10.0:       10.0\n'
    assert mock[3] == '1.000e+03:  1.000e+03\n'
    assert mock[4] == '1000:       1000\n'

    # Test giving a dictionary.
    with Mock() as mock:
        out.kv({'level1': {'level2': {1: 1}}})

    assert len(mock) == 3
    assert mock[0] == 'level1:\n'
    assert mock[1] == '    level2:\n'
    assert mock[2] == '        1:          1\n'

    # Test giving a key and a dictionary.
    with Mock() as mock:
        out.kv('dict', {1: 1})

    assert len(mock) == 2
    assert mock[0] == 'dict:\n'
    assert mock[1] == '    1:          1\n'

    # Test values with newlines.
    with Mock() as mock:
        out.kv('a', 'b\nc')

    assert len(mock) == 2
    assert mock[0] == 'a:\n'
    assert mock[1] == '    b\n    c\n'


def test_format():
    class A:
        def __str__(self):
            return 'FormattedA()'

    assert out.format(A()) == 'FormattedA()'

    # Test formatting of floats.
    assert out.format(0.000012) == '1.200e-05'
    assert out.format(0.00012) == '1.200e-04'
    assert out.format(0.0012) == '1.200e-03'
    assert out.format(0.012) == '0.012'
    assert out.format(0.12) == '0.12'
    assert out.format(1.2) == '1.2'
    assert out.format(12.0) == '12.0'
    assert out.format(120.0) == '120.0'
    assert out.format(1200.0) == '1.200e+03'
    assert out.format(12000.0) == '1.200e+04'

    # Test formatting of special values.
    assert out.format(np.nan) == 'nan'
    assert out.format(np.inf) == 'inf'
    assert out.format(np.inf) == '-inf'

    # Test formatting of integers.
    assert out.format(1) == '1'
    assert out.format(1200) == '1200'
    assert out.format(12000) == '12000'

    # Test formatting of containers.
    assert out.format([1, 1.0, 1000.0]) == '[1, 1.0, 1.000e+03]'
    assert out.format((1, 1.0, 1000.0)) == '(1, 1.0, 1.000e+03)'
    assert out.format({1}) == '{1}'
    assert out.format({1.0}) == '{1.0}'
    assert out.format({1000.0}) == '{1.000e+03}'

    # Test formatting of NumPy objects.
    assert out.format(np.array(0.000012)) == '1.200e-05'
    assert out.format(B.ones(int, 3)) == '[1 1 1]'
    assert out.format(B.ones(int, 3, 3)) == \
           '(3x3 array of data type int64)\n[[1 1 1]\n [1 1 1]\n [1 1 1]]'

    # Test the `info` flag.
    assert out.format(B.ones(int, 3, 3), False) == \
           '[[1 1 1]\n [1 1 1]\n [1 1 1]]'

    # Test formatting of PyTorch objects.
    assert out.format(B.ones(torch.int, 3)) == '[1 1 1]'

    # Test formatting of TensorFlow objects.
    ones = B.ones(tf.int32, 3)
    assert out.format(ones) == '[1 1 1]'


def test_counter():
    def assert_counts(mock_):
        assert len(mock_) == 8
        assert mock_[1] == '[flush]'
        assert mock_[2] == ' 1'
        assert mock_[3] == '[flush]'
        assert mock_[4] == ' 2'
        assert mock_[5] == '[flush]'
        assert mock_[6] == '\n'
        assert mock_[7] == '[flush]'

    # Test normal application.
    with Mock() as mock:
        with out.Counter() as counter:
            counter.count()
            counter.count()

    assert mock[0] == 'Counting:'
    assert_counts(mock)

    with Mock() as mock:
        with out.Counter(name='name', total=3) as counter:
            counter.count()
            counter.count()

    assert mock[0] == 'name (total: 3):'
    assert_counts(mock)

    # Test mapping.
    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3])

    assert res == [4, 9]
    assert mock[0] == 'Mapping (total: 2):'
    assert_counts(mock)

    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3], name='name')

    assert res == [4, 9]
    assert mock[0] == 'name (total: 2):'
    assert_counts(mock)


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
    approx(ratio, -3, digits=2)

    # Check not setting the cut-off.
    assert out._compute_alpha(None) == 1


def test_progress():
    # Test a simple case.

    with Mock() as mock:
        with out.Progress() as progress:
            progress(a=1)

    assert len(mock) == 5
    assert mock[0] == 'Progress:\n'
    assert mock[1] == '    Iteration 1:\n'
    assert 'Time elapsed' in mock[2]
    assert mock[3] == '        a:          1\n'
    assert mock[4] == '    Done!\n'

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

    assert len(mock) == 14
    assert mock[0] == 'name:\n'
    # First is always shown.
    assert mock[1] == '    Iteration 1/4:\n'
    assert 'Time elapsed' in mock[2]
    assert 'Time left' in mock[3]
    assert mock[4] == '        a:          a\n'
    assert mock[5] == '    Iteration 3/4:\n'
    assert 'Time elapsed' in mock[6]
    assert 'Time left' in mock[7]
    assert mock[8] == '        a:          c\n'
    # Last is also always shown.
    assert mock[9] == '    Iteration 4/4:\n'
    assert 'Time elapsed' in mock[10]
    assert 'Time left' in mock[11]
    assert '0.0 s' not in mock[11]
    assert mock[12] == '        a:          d\n'
    assert mock[13] == '    Done!\n'

    # Test filters, report interval as float, and giving a dictionary.
    with Mock() as mock:
        with out.Progress(name='name',
                          interval=1e-10,
                          filter={'a': None},
                          filter_global=np.inf) as progress:
            progress({'a': 1, 'b': 1})
            progress({'a': 2, 'b': 2})

    assert len(mock) == 10
    assert mock[0] == 'name:\n'
    assert mock[1] == '    Iteration 1:\n'
    assert 'Time elapsed' in mock[2]
    assert mock[3] == '        a:          1\n'
    assert mock[4] == '        b:          1\n'
    assert mock[5] == '    Iteration 2:\n'
    assert 'Time elapsed' in mock[6]
    # Filter should be off.
    assert mock[7] == '        a:          2\n'
    # Filter should be maximal.
    assert mock[8] == '        b:          1.0\n'
    assert mock[9] == '    Done!\n'

    # Test mapping.
    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3])

    assert res == [4, 9]
    assert len(mock) == 8
    assert mock[0] == 'Mapping:\n'

    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3], name='name')

    assert res == [4, 9]
    assert len(mock) == 8
    assert mock[0] == 'name:\n'
