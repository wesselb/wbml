# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import lab as B
from . import Normalise, Linear, Activation, Recurrent, GRU, Elman, ff, rnn

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, approx, allclose
from varz import Vars


def check_batch_consistency(layer, xs):
    outs = []
    for x in xs:
        outs.append(layer(x))

    # Test consistency.
    allclose(layer(xs), B.stack(*outs, axis=0))

    # Test exceptions.
    raises(ValueError, lambda: layer(B.ones(5)))


def test_normalise():
    layer = Normalise(epsilon=0)
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    yield eq, layer.num_weights(10), 0
    yield eq, layer.width, 10

    # Check initialisation and width.
    layer.initialise(3, None)
    yield eq, layer.width, 3

    # Check batch consistency.
    yield check_batch_consistency, layer, x

    # Check correctness
    out = layer(x)
    yield approx, B.std(out, axis=2), B.ones(10, 5)
    yield approx, B.mean(out, axis=2), B.zeros(10, 5)


def test_linear():
    layer = Linear(20)
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    yield eq, layer.num_weights(3), 3 * 20 + 20
    yield eq, layer.width, 20

    # Check initialisation and width.
    vs = Vars(np.float64)
    layer.initialise(3, vs)
    yield eq, layer.width, 20

    # Check batch consistency.
    yield check_batch_consistency, layer, x

    # Check correctness.
    yield allclose, layer(x), \
          B.matmul(x, layer.A[None, :, :]) + layer.b[None, :, :]


def test_activation():
    layer = Activation()
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    yield eq, layer.num_weights(10), 0
    yield eq, layer.width, 10

    # Check initialisation and width.
    layer.initialise(3, None)
    yield eq, layer.width, 3

    # Check batch consistency.
    yield check_batch_consistency, layer, x

    # Check correctness
    yield allclose, layer(x), B.relu(x)


def test_recurrent():
    vs = Vars(np.float32)

    # Test setting the initial hidden state.

    layer = Recurrent(GRU(10), B.zeros(1, 10))
    layer.initialise(5, vs)
    yield allclose, layer.h0, B.zeros(1, 10)

    layer = Recurrent(GRU(10))
    layer.initialise(5, vs)
    yield ok, layer.h0 is not None

    # Check batch consistency.
    yield check_batch_consistency, B.randn(30, 20, 5)

    # Test preservation of rank upon calls.
    yield eq, B.shape(layer(B.randn(20, 5))), (20, 10)
    yield eq, B.shape(layer(B.randn(30, 20, 5))), (30, 20, 10)


def test_ff():
    vs = Vars(np.float32)

    nn = ff(10, (20, 30), normalise=True)
    nn.initialise(5, vs)
    x = B.randn(2, 3, 5)

    # Check number of weights and width.
    yield eq, B.length(vs.get_vector()), nn.num_weights(5)
    yield eq, nn.width(10)

    # Test batch consistency.
    yield check_batch_consistency, nn, x

    # Check composition.
    yield eq, len(nn.layers), 7
    yield eq, type(nn.layers[0]), Linear
    yield eq, nn.layers[0].A.shape[0], 5
    yield eq, nn.layers[0].width, 20
    yield eq, type(nn.layers[1]), Activation
    yield eq, nn.layers[1].width, 20
    yield eq, type(nn.layers[2]), Normalise
    yield eq, nn.layers[2].width, 20
    yield eq, type(nn.layers[3]), Linear
    yield eq, nn.layers[3].width, 30
    yield eq, type(nn.layers[4]), Activation
    yield eq, nn.layers[4].width, 30
    yield eq, type(nn.layers[5]), Normalise
    yield eq, nn.layers[5].width, 30
    yield eq, type(nn.layers[6]), Linear
    yield eq, nn.layers[6].width, 10

    # Check normalisation layers disappear.
    yield eq, len(ff(10, (20, 30), normalise=False).layers), 5


def test_rnn():
    for final_dense, gru, nn in \
            [(True, False,
              rnn(10, (20, 30), normalise=True, gru=False, final_dense=True)),
             (False, True,
              rnn(10, (20, 30), normalise=True, gru=True, final_dense=False))]:
        vs = Vars(np.float32)
        nn.initialise(5, vs)
        x = B.randn(2, 3, 5)

        # Check number of weights and width.
        yield eq, B.length(vs.get_vector()), nn.num_weights(5)
        yield eq, nn.width, 10

        # Test batch consistency.
        yield check_batch_consistency, nn, x

        # Check composition.
        yield eq, len(nn.layers), 9 if final_dense else 7
        yield eq, type(nn.layers[0]), Recurrent
        yield eq, type(nn.layers[0].cell), GRU if gru else Elman
        yield eq, nn.layers[0].width, 20
        yield eq, type(nn.layers[1]), Activation
        yield eq, nn.layers[1].width, 20
        yield eq, type(nn.layers[2]), Normalise
        yield eq, nn.layers[2].width, 20
        yield eq, type(nn.layers[3]), Recurrent
        yield eq, type(nn.layers[3].cell), GRU if gru else Elman
        yield eq, nn.layers[3].width, 30
        yield eq, type(nn.layers[4]), Activation
        yield eq, nn.layers[4].width, 30
        yield eq, type(nn.layers[5]), Normalise
        yield eq, nn.layers[5].width, 30
        if final_dense:
            yield eq, type(nn.layers[6]), Linear
            yield eq, nn.layers[6].width, 10
            yield eq, type(nn.layers[7]), Activation
            yield eq, nn.layers[7].width, 10
            yield eq, type(nn.layers[8]), Linear
            yield eq, nn.layers[8].width, 10
        else:
            yield eq, type(nn.layers[6]), Linear
            yield eq, nn.layers[6].width, 10

    # Check normalisation layers disappear.
    yield eq, len(rnn(10, (20, 30),
                      normalise=False,
                      gru=True,
                      final_dense=False).layers), 5
