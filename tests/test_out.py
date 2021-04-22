import pytest

import lab as B
import numpy as np
import tensorflow as tf
import torch
import time

import wbml.out as out
from .util import approx


class RecordingStream:
    """A stream that records its writes."""

    def __init__(self):
        self.writes = []

    def write(self, msg):
        self.writes.append(msg)

    def flush(self):
        self.writes.append("[flush]")

    def __str__(self):
        return "".join(self.writes)

    def __len__(self):
        return len(self.writes)


class Mock:
    """Mock the stream that `wbml.out` uses. Also reset
    `wbml.out._time_start`."""

    def __enter__(self):
        self.stream = RecordingStream()

        # Save current stream and time started.
        self.saved_streams = list(out.streams)
        self.saved_time_start = out._time_start

        # Mock stream and time started.
        out.streams = [self.stream]
        out._time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unmock stream and time started.
        out.streams = self.saved_streams
        out._time_start = self.saved_time_start

    def __str__(self):
        return str(self.stream)

    def __repr__(self):
        return "<Mock: recorded=" + str(self.stream) + ">"

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, i):
        return self.stream.writes[i]


def test_section():
    with Mock() as mock:
        out.out("before")

        with out.Section():
            out.out("message1")

        with out.Section("name"):
            out.out("message2")

            with out.Section():
                out.out("message3")

        out.out("after")

    assert len(mock) == 6
    assert mock[0] == "before\n"
    assert mock[1] == "    message1\n"
    assert mock[2] == "name:\n"
    assert mock[3] == "    message2\n"
    assert mock[4] == "        message3\n"
    assert mock[5] == "after\n"


def test_out():
    with Mock() as mock:
        out.out("message")

    assert len(mock) == 1
    assert str(mock) == "message\n"


def test_out_newlines():
    # Test that newlines are correctly indented.
    with Mock() as mock:
        out.out("a\nb")

        with out.Section():
            out.out("c\nd")

    assert len(mock) == 2
    assert mock[0] == "a\nb\n"
    assert mock[1] == "    c\n    d\n"


def test_kv():
    with Mock() as mock:
        out.kv("key", "value")
        out.kv(1.0, 1.0)
        out.kv(10.0, 10.0)
        out.kv(1000.0, 1000.0)
        out.kv(1000, 1000)

    assert len(mock) == 5
    assert mock[0] == "key:        value\n"
    assert mock[1] == "1.0:        1.0\n"
    assert mock[2] == "10.0:       10.0\n"
    assert mock[3] == "1.000e+03:  1.000e+03\n"
    assert mock[4] == "1000:       1000\n"


def test_kv_unit():
    with Mock() as mock:
        out.kv("key", 1, unit="s")

    assert len(mock) == 1
    assert mock[0] == "key:        1 s\n"


def test_kv_fmt():
    with Mock() as mock:
        out.kv("key", 1, fmt=".4f")

    assert len(mock) == 1
    assert mock[0] == "key:        1.0000\n"


def test_kv_dict():
    # Test giving a dictionary.
    with Mock() as mock:
        out.kv({"level1": {"level2": {1: 1}}})

    assert len(mock) == 3
    assert mock[0] == "level1:\n"
    assert mock[1] == "    level2:\n"
    assert mock[2] == "        1:          1\n"


def test_kv_dict_as_value():
    # Test giving a key and a dictionary.
    with Mock() as mock:
        out.kv("dict", {1: 1})

    assert len(mock) == 2
    assert mock[0] == "dict:\n"
    assert mock[1] == "    1:          1\n"


def test_kv_newlines():
    # Test values with newlines.
    with Mock() as mock:
        out.kv("a", "b\nc")

    assert len(mock) == 2
    assert mock[0] == "a:\n"
    assert mock[1] == "    b\n    c\n"


def test_format_object():
    class A:
        def __str__(self):
            return "FormattedA()"

    assert out.format(A()) == "FormattedA()"


@pytest.mark.parametrize(
    "x, y",
    [
        # Floats:
        (0.000012, "1.200e-05"),
        (0.00012, "1.200e-04"),
        (0.0012, "1.200e-03"),
        (0.012, "0.012"),
        (0.12, "0.12"),
        (1.2, "1.2"),
        (12.0, "12.0"),
        (120.0, "120.0"),
        (1200.0, "1.200e+03"),
        (12000.0, "1.200e+04"),
        # Integers:
        (1, "1"),
        (1200, "1200"),
        (12000, "12000"),
        # NaN and infinity:
        (np.nan, "nan"),
        (np.inf, "inf"),
        (-np.inf, "-inf"),
        # Containers:
        ([1, 1.0, 1000.0], "[1, 1.0, 1.000e+03]"),
        ((1, 1.0, 1000.0), "(1, 1.0, 1.000e+03)"),
        ({1}, "{1}"),
        ({1.0}, "{1.0}"),
        ({1000.0}, "{1.000e+03}"),
        # NumPy:
        (np.array(0.000012), "1.200e-05"),
        (B.ones(int, 3), "[1 1 1]"),
        (
            B.ones(int, 3, 3),
            "(3x3 array of data type int64)\n[[1 1 1]\n [1 1 1]\n [1 1 1]]",
        ),
        # PyTorch:
        (B.ones(torch.int, 3), "[1 1 1]"),
        # TensorFlow:
        (B.ones(tf.int32, 3), "[1 1 1]"),
    ],
)
def test_format(x, y):
    assert out.format(x) == y


def test_format_info_flag():
    assert out.format(B.ones(int, 3, 3), False) == "[[1 1 1]\n [1 1 1]\n [1 1 1]]"


def _assert_counts(mock):
    assert len(mock) == 8
    assert mock[1] == "[flush]"
    assert mock[2] == " 1"
    assert mock[3] == "[flush]"
    assert mock[4] == " 2"
    assert mock[5] == "[flush]"
    assert mock[6] == "\n"
    assert mock[7] == "[flush]"


def test_counter():
    # Test normal application.
    with Mock() as mock:
        with out.Counter() as counter:
            counter.count()
            counter.count()

    assert mock[0] == "Counting:"
    _assert_counts(mock)


def test_counter_total():
    with Mock() as mock:
        with out.Counter(name="name", total=3) as counter:
            counter.count()
            counter.count()

    assert mock[0] == "name (total: 3):"
    _assert_counts(mock)


def test_counter_map():
    # Test mapping.
    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3])

    assert res == [4, 9]
    assert mock[0] == "Mapping (total: 2):"
    _assert_counts(mock)


def test_counter_map2():
    with Mock() as mock:
        res = out.Counter.map(lambda x: x ** 2, [2, 3], name="name")

    assert res == [4, 9]
    assert mock[0] == "name (total: 2):"
    _assert_counts(mock)


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
    approx(ratio, -3, atol=1e-3)

    # Check not setting the cut-off.
    assert out._compute_alpha(None) == 1


def test_progress():
    # Test a simple case.
    with Mock() as mock:
        with out.Progress() as progress:
            progress(a=1)

    assert len(mock) == 5
    assert mock[0] == "Progress:\n"
    assert mock[1] == "    Iteration 1:\n"
    assert "Time elapsed" in mock[2]
    assert mock[3] == "        a:          1\n"
    assert mock[4] == "    Done!\n"


def test_progress_interval():
    # Test setting the total number of iterations and report interval as int.
    with Mock() as mock:
        with out.Progress(name="name", total=4, interval=3) as progress:
            progress(a="a")
            progress(a="b")
            progress(a="c")
            # Change time per iteration to a second to show a non-zero time
            # left.
            progress.values["_time_per_it"] = 1
            progress(a="d")

    assert len(mock) == 14
    assert mock[0] == "name:\n"
    # First is always shown.
    assert mock[1] == "    Iteration 1/4:\n"
    assert "Time elapsed" in mock[2]
    assert "Time left" in mock[3]
    assert mock[4] == "        a:          a\n"
    assert mock[5] == "    Iteration 3/4:\n"
    assert "Time elapsed" in mock[6]
    assert "Time left" in mock[7]
    assert mock[8] == "        a:          c\n"
    # Last is also always shown.
    assert mock[9] == "    Iteration 4/4:\n"
    assert "Time elapsed" in mock[10]
    assert "Time left" in mock[11]
    assert "0.0 s" not in mock[11]
    assert mock[12] == "        a:          d\n"
    assert mock[13] == "    Done!\n"


def test_progress_filters():
    # Test filters, report interval as float, and giving a dictionary.
    with Mock() as mock:
        with out.Progress(
            name="name", interval=1e-10, filter={"a": None}, filter_global=np.inf
        ) as progress:
            progress({"a": 1, "b": 1})
            progress({"a": 2, "b": 2})

    assert len(mock) == 10
    assert mock[0] == "name:\n"
    assert mock[1] == "    Iteration 1:\n"
    assert "Time elapsed" in mock[2]
    assert mock[3] == "        a:          1\n"
    assert mock[4] == "        b:          1\n"
    assert mock[5] == "    Iteration 2:\n"
    assert "Time elapsed" in mock[6]
    # Filter should be off.
    assert mock[7] == "        a:          2\n"
    # Filter should be maximal.
    assert mock[8] == "        b:          1.0\n"
    assert mock[9] == "    Done!\n"


def test_progress_map():
    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3])

    assert res == [4, 9]
    assert len(mock) == 8
    assert mock[0] == "Mapping:\n"


def test_progress_map2():
    with Mock() as mock:
        res = out.Progress.map(lambda x: x ** 2, [2, 3], name="name")

    assert res == [4, 9]
    assert len(mock) == 8
    assert mock[0] == "name:\n"


def test_time_report_interval(monkeypatch):
    monkeypatch.setattr(out, "report_time", True)

    # Test that time stamp is not repeated unnecessarily.
    with Mock() as mock:
        out.out("a")
        out.out("b")
        time.sleep(1.0)
        out.out("c")

    assert len(mock) == 3
    assert mock[0] == "00:00:00 | a\n"
    assert mock[1] == "         | b\n"
    assert mock[2] == "00:00:01 | c\n"


def test_time_report_calculation(monkeypatch):
    monkeypatch.setattr(out, "report_time", True)

    # Test that time is correctly calculated.
    with Mock() as mock:
        out._time_start = time.time() - 2 * 60 * 60 - 2 * 60 - 2
        out.out("a")

    assert len(mock) == 1
    assert mock[0] == "02:02:02 | a\n"


def test_time_report_with_progress(monkeypatch):
    monkeypatch.setattr(out, "report_time", True)

    # Test normal application of counting, as above.
    with Mock() as mock:
        with out.Counter() as counter:
            counter.count()
            counter.count()

    assert len(mock) == 8
    assert mock[0] == "00:00:00 | Counting:"
    assert mock[1] == "[flush]"
    assert mock[2] == " 1"
    assert mock[3] == "[flush]"
    assert mock[4] == " 2"
    assert mock[5] == "[flush]"
    assert mock[6] == "\n"
    assert mock[7] == "[flush]"
