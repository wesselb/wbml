# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, eprint
from wbml import feedforward, Vars


def test_construction():
    vars32 = Vars(np.float32)
    # Note: the layers are interleaved with batch norm layers.
    nn = feedforward([10, 100, 200, 20], vars=vars32)

    yield eq, nn.input_size, 10
    yield eq, len(nn.layers), 5
    yield eq, nn.layers[0].width, 100
    yield eq, nn.layers[3].width, 200
    yield eq, nn.layers[4].width, 20

    s = tf.Session()
    vars32.init(s)
    y = s.run(nn(np.random.randn(10, 5).astype(np.float32)))
    yield eq, y.shape, (20, 5)

    # Test float64 compatibility
    vars64 = Vars(np.float64)
    nn = feedforward([10, 100, 20], vars=vars64)
    vars64.init(s)
    y = s.run(nn(np.random.randn(10, 5).astype(np.float64)))
    yield eq, y.shape, (20, 5)
