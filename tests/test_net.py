# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from lab.tf import B

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, eprint
from wbml import ff, Vars, rnn


def test_construction():
    vars32 = Vars(np.float32)
    # Note: the layers are interleaved with batch norm layers.
    nn = ff(10, 20, (100, 200))
    nn.initialise(vars32)

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
    nn = ff(10, 20, (100,))
    nn.initialise(vars64)
    vars64.init(s)
    y = s.run(nn(np.random.randn(10, 5).astype(np.float64)))
    yield eq, y.shape, (20, 5)


def test_num_weights():
    vars32 = Vars(np.float32)
    nn = ff(10, 20, (30, 40))
    nn.initialise(vars32)

    yield eq, B.shape_int(nn.weights())[0], nn.num_weights(), 'ff'

    vars32 = Vars(np.float32)
    nn = rnn(10, 20, ((30, 40), (40, 50)))
    nn.initialise(vars32)

    yield eq, B.shape_int(nn.weights())[0], nn.num_weights(), 'rnn'
