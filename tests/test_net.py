import lab as B
import numpy as np
import pytest
from varz import Vars

from wbml.net import Normalise, Linear, Activation, Recurrent, GRU, Elman, ff, rnn
from .util import approx, approx


def check_batch_consistency(layer, xs):
    outs = []
    for x in xs:
        outs.append(layer(x))

    # Test consistency. This tests that rank 2 and 3 tensors are of the correct
    # shape.
    approx(layer(xs), B.stack(*outs, axis=0))


def test_normalise():
    layer = Normalise(epsilon=0)
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    assert layer.num_weights(10) == 0
    assert layer.width == 10

    # Check initialisation and width.
    layer.initialise(3, None)
    assert layer.width == 3

    # Check correctness
    out = layer(x)
    approx(B.std(out, axis=2), B.ones(10, 5), rtol=1e-4)
    approx(B.mean(out, axis=2), B.zeros(10, 5), atol=1e-4)


def test_linear():
    layer = Linear(20)
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    assert layer.num_weights(3) == 3 * 20 + 20
    assert layer.width == 20

    # Check initialisation and width.
    vs = Vars(np.float64)
    layer.initialise(3, vs)
    assert layer.width == 20

    # Check batch consistency.
    check_batch_consistency(layer, x)

    # Check correctness.
    approx(layer(x), B.matmul(x, layer.A[None, :, :]) + layer.b[None, :, :])


def test_activation():
    layer = Activation()
    x = B.randn(10, 5, 3)

    # Check number of weights and width.
    assert layer.num_weights(10) == 0
    assert layer.width == 10

    # Check initialisation and width.
    layer.initialise(3, None)
    assert layer.width == 3

    # Check correctness
    approx(layer(x), B.relu(x))


def test_recurrent():
    vs = Vars(np.float32)

    # Test setting the initial hidden state.
    layer = Recurrent(GRU(10), B.zeros(1, 10))
    layer.initialise(5, vs)
    approx(layer.h0, B.zeros(1, 10))

    layer = Recurrent(GRU(10))
    layer.initialise(5, vs)
    assert layer.h0 is not None

    # Check batch consistency.
    check_batch_consistency(layer, B.randn(30, 20, 5))

    # Test preservation of rank upon calls.
    assert B.shape(layer(B.randn(20, 5))) == (20, 10)
    assert B.shape(layer(B.randn(30, 20, 5))) == (30, 20, 10)

    # Check that zero-dimensional calls fail.
    with pytest.raises(ValueError):
        layer(0)


def test_ff():
    vs = Vars(np.float32)

    nn = ff(10, (20, 30), normalise=True)
    nn.initialise(5, vs)
    x = B.randn(2, 3, 5)

    # Check number of weights and width.
    assert B.length(vs.get_vector()) == nn.num_weights(5)
    assert nn.width == 10

    # Test batch consistency.
    check_batch_consistency(nn, x)

    # Check composition.
    assert len(nn.layers) == 7
    assert type(nn.layers[0]) == Linear
    assert nn.layers[0].A.shape[0] == 5
    assert nn.layers[0].width == 20
    assert type(nn.layers[1]) == Activation
    assert nn.layers[1].width == 20
    assert type(nn.layers[2]) == Normalise
    assert nn.layers[2].width == 20
    assert type(nn.layers[3]) == Linear
    assert nn.layers[3].width == 30
    assert type(nn.layers[4]) == Activation
    assert nn.layers[4].width == 30
    assert type(nn.layers[5]) == Normalise
    assert nn.layers[5].width == 30
    assert type(nn.layers[6]) == Linear
    assert nn.layers[6].width == 10

    # Check that one-dimensional calls are okay.
    vs = Vars(np.float32)
    nn.initialise(1, vs)
    approx(nn(B.linspace(0, 1, 10)), nn(B.linspace(0, 1, 10)[:, None]))

    # Check that zero-dimensional calls fail.
    with pytest.raises(ValueError):
        nn(0)

    # Check normalisation layers disappear.
    assert len(ff(10, (20, 30), normalise=False).layers) == 5


def test_rnn():
    for final_dense, gru, nn in [
        (True, False, rnn(10, (20, 30), normalise=True, gru=False, final_dense=True)),
        (False, True, rnn(10, (20, 30), normalise=True, gru=True, final_dense=False)),
    ]:
        vs = Vars(np.float32)
        nn.initialise(5, vs)
        x = B.randn(2, 3, 5)

        # Check number of weights and width.
        assert B.length(vs.get_vector()) == nn.num_weights(5)
        assert nn.width == 10

        # Test batch consistency.
        check_batch_consistency(nn, x)

        # Check composition.
        assert len(nn.layers) == 9 if final_dense else 7
        assert type(nn.layers[0]) == Recurrent
        assert type(nn.layers[0].cell) == GRU if gru else Elman
        assert nn.layers[0].width == 20
        assert type(nn.layers[1]) == Activation
        assert nn.layers[1].width == 20
        assert type(nn.layers[2]) == Normalise
        assert nn.layers[2].width == 20
        assert type(nn.layers[3]) == Recurrent
        assert type(nn.layers[3].cell) == GRU if gru else Elman
        assert nn.layers[3].width == 30
        assert type(nn.layers[4]) == Activation
        assert nn.layers[4].width == 30
        assert type(nn.layers[5]) == Normalise
        assert nn.layers[5].width == 30
        if final_dense:
            assert type(nn.layers[6]) == Linear
            assert nn.layers[6].width == 10
            assert type(nn.layers[7]) == Activation
            assert nn.layers[7].width == 10
            assert type(nn.layers[8]) == Linear
            assert nn.layers[8].width == 10
        else:
            assert type(nn.layers[6]) == Linear
            assert nn.layers[6].width == 10

    # Check that normalisation layers disappear.
    assert (
        len(rnn(10, (20, 30), normalise=False, gru=True, final_dense=False).layers) == 5
    )
