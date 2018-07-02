# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

from lab import B

from .util import vars32, inv_perm, identity


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def initialise(self, input_size, vars):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def num_weights(self, input_size):
        pass


class Identity(Layer):
    def __init__(self):
        pass

    def initialise(self, *arg):
        pass

    def weights(self):
        return []

    def __call__(self, x):
        return identity(x)

    def num_weights(self, input_size):
        return 0


class FeedForwardLayer(Layer):
    __metaclass__ = ABCMeta

    def __call__(self, *xs):
        # Stack inputs.
        x = B.concat(xs, axis=0)

        if B.rank(x) == 2:
            return self._apply(x)
        else:
            # Shape into one big matrix for efficient application.
            x = B.transpose(x, [1, 2, 0])
            shape = B.shape(x)
            x = B.reshape(x, [B.shape(x)[0], -1])

            # Apply.
            x = self._apply(x)

            # Shape back.
            x = B.reshape(x, [self.width, shape[1], shape[2]])
            return B.transpose(x, inv_perm([1, 2, 0]))

    @abstractmethod
    def _apply(self, x):
        pass


class Normalise(FeedForwardLayer):
    def __init__(self):
        self.width = None

    def initialise(self, input_size, vars):
        self.width = input_size

    def weights(self):
        return []

    def _apply(self, x):
        mean = B.mean(x, axis=0)[None, :]
        std = B.mean((x - mean) ** 2, axis=0)[None, :] ** .5
        return (x - mean) / (std + B.epsilon)

    def num_weights(self, input_size):
        self.width = input_size
        return 0


class Dense(FeedForwardLayer):
    def __init__(self, width, nonlinearity=None):
        self.width = width
        self.A = None
        self.b = None
        self.nonlinearity = B.relu if nonlinearity is None else nonlinearity

    def initialise(self, input_size, vars):
        self.A = vars.get(shape=[self.width, input_size])
        self.b = vars.get(shape=[self.width, 1])

    def _apply(self, x):
        return self.nonlinearity(B.matmul(self.A, x) + self.b)

    def weights(self):
        return B.concat([B.reshape(self.A, [-1]),
                         B.reshape(self.b, [-1])], axis=0)

    def num_weights(self, input_size):
        return self.width * (input_size + 1)


class Linear(Dense):
    def __init__(self, width):
        Dense.__init__(self, width, nonlinearity=identity)


class Net(object):
    def __init__(self, input_size, layers):
        self.input_size = input_size
        self.layers = layers
        self.is_initialised = False

    def initialise(self, vars=vars32):
        # Initialise layers.
        cur_size = self.input_size
        for layer in self.layers:
            layer.initialise(cur_size, vars)
            cur_size = layer.width
        self.is_initialised = True

    def weights(self):
        return B.concat([x.weights() for x in self.layers], axis=0)

    def __call__(self, x):
        # Make sure the net is initialised.
        if not self.is_initialised:
            self.initialise()

        # Apply layers.
        for layer in self.layers:
            x = layer(x)
        return x

    def num_weights(self):
        num_weights = 0
        cur_size = self.input_size
        for layer in self.layers:
            num_weights += layer.num_weights(cur_size)
            cur_size = layer.width
        return num_weights


class GRU(Layer):
    def __init__(self, width, nonlinearity=None):
        self.nonlinearity = B.tahn if nonlinearity is None else nonlinearity
        self.width = width
        self.f_z = Dense(self.width, nonlinearity=B.sigmoid)
        self.f_r = Dense(self.width, nonlinearity=B.sigmoid)
        self.f_h = Dense(self.width, nonlinearity=self.nonlinearity)
        self.h0 = None

    def initialise(self, input_size, vars):
        self.h0 = vars.get(shape=[self.width, 1])
        self.f_z.initialise(self.width + input_size, vars)
        self.f_r.initialise(self.width + input_size, vars)
        self.f_h.initialise(self.width + input_size, vars)

    def __call__(self, xs):
        batch_size = B.shape(xs)[2]

        def loop(prev, x):
            prev_h, prev_y = prev

            # Gate logic:
            z = self.f_z(prev_h, x)
            r = self.f_r(prev_h, x)
            h = (1 - z) * prev_h + z * self.f_h(r * prev_h, x)
            y = h

            return h, y

        y0 = B.zeros((self.width, batch_size), dtype=B.dtype(xs))
        h0 = B.tile(self.h0, (1, batch_size))
        return B.scan(loop, xs, initializer=(h0, y0))[1]

    def weights(self):
        return B.concat([B.reshape(self.h0, [-1]),
                         self.f_z.weights(),
                         self.f_r.weights(),
                         self.f_h.weights()], axis=0)

    def num_weights(self, input_size):
        return self.width + \
               self.f_z.num_weights(self.width + input_size) + \
               self.f_r.num_weights(self.width + input_size) + \
               self.f_h.num_weights(self.width + input_size)


class Elman(Layer):
    def __init__(self, width, nonlinearity=None):
        self.nonlinearity = B.sigmoid if nonlinearity is None else nonlinearity
        self.width = width
        self.f_h = Dense(self.width, nonlinearity=self.nonlinearity)
        self.h0 = None

    def initialise(self, input_size, vars):
        self.h0 = vars.get(shape=[self.width, 1])
        self.f_h.initialise(self.width + input_size, vars)

    def __call__(self, xs):
        batch_size = B.shape(xs)[2]

        def loop(prev, x):
            prev_h, prev_y = prev

            # Gate logic:
            h = self.f_h(prev_h, x)
            y = h

            return h, y

        y0 = B.zeros((self.width, batch_size), dtype=B.dtype(xs))
        h0 = B.tile(self.h0, (1, batch_size))
        return B.scan(loop, xs, initializer=(h0, y0))[1]

    def weights(self):
        return B.concat([B.reshape(self.h0, [-1]), self.f_h.weights()], axis=0)

    def num_weights(self, input_size):
        return self.width + \
               self.f_h.num_weights(self.width + input_size)


def ff(input_size, output_size, widths, nonlinearity=None, normalise=True):
    """A standard feed-forward neural net.

    Args:
        input_size (int): Size of input.
        output_size (int): Size of output.
        widths (tuple of int): Widths of the layers.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            ReLUs.
        normalise (bool, optional): Interleave with normalisation layers.
            Defaults to `True`.
    """
    nonlinearity = B.relu if nonlinearity is None else nonlinearity
    layers = []
    for width in widths:
        layers.append(Dense(width, nonlinearity=nonlinearity))
        if normalise:
            layers.append(Normalise())
    layers.append(Linear(output_size))
    return Net(input_size, layers)


def rnn(input_size, output_size, widths,
        nonlinearity=None,
        normalise=False,
        gru=True,
        final_dense=False):
    """A standard recurrent neural net.

    Args:
        input_size (int): Size of input.
        output_size (int): Size of output.
        widths (tuple of int): Widths of the layers.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            sigmoid.
        normalise (bool, optional): Interleave with normalisation layers.
            Defaults to `False`.
        gru (bool, optional): Use GRU layers instead of Elman layers. Defaults
            to `True`.
        final_dense (bool, optional): Append a final dense layer. Defaults to
            `False`.
    """
    nonlinearity = B.sigmoid if nonlinearity is None else nonlinearity
    layer_type = GRU if gru else Elman
    layers = []
    for width in widths:
        layers.append(layer_type(width, nonlinearity=nonlinearity))
        if normalise:
            layers.append(Normalise())
    if final_dense:
        layers.append(Dense(output_size, nonlinearity=nonlinearity))
    layers.append(Linear(output_size))
    return Net(input_size, layers)
