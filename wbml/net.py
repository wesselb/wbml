# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

import tensorflow as tf
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

    def __call__(self, x):
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
            x = B.reshape(x, shape)
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
    def __init__(self, width, nonlinearity=tf.nn.relu):
        self.width = width
        self.A = None
        self.b = None
        self.nonlinearity = nonlinearity

    def initialise(self, input_size, vars):
        self.A = vars.get(shape=[self.width, input_size])
        self.b = vars.get(shape=[self.width, 1])

    def _apply(self, x):
        return self.nonlinearity(B.matmul(self.A, x) + self.b)

    def weights(self):
        return B.concat([B.reshape(self.A, [-1]),
                         B.reshape(self.b, [-1])], axis=0)

    def num_weights(self, input_size):
        return self.width * input_size + self.width


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


class Elman(Layer):
    def __init__(self, width, hidden_width, nonlinearity=tf.nn.sigmoid):
        self.nonlinearity = nonlinearity
        self.width = width
        self.hidden_width = hidden_width
        self.f_h = Dense(self.hidden_width, nonlinearity=self.nonlinearity)
        self.f_y = Dense(self.width, nonlinearity=self.nonlinearity)
        self.h0 = None

    def initialise(self, input_size, vars):
        self.h0 = vars.get(shape=[self.hidden_width, 1])
        self.f_h.initialise(self.hidden_width + input_size, vars)
        self.f_y.initialise(self.hidden_width, vars)

    def __call__(self, xs):
        batch_size = B.shape(xs)[2]

        def loop(prev, x):
            h = self.f_h(B.concat([prev[0], x], axis=0))
            y = self.f_y(h)
            return h, y

        y0 = B.zeros((self.width, batch_size), dtype=B.dtype(xs))
        h0 = B.tile(self.h0, (1, batch_size))
        return tf.scan(loop, xs, initializer=(h0, y0))[1]

    def weights(self):
        return B.concat([B.reshape(self.h0, [-1]),
                         self.f_h.weights(),
                         self.f_y.weights()], axis=0)

    def num_weights(self, input_size):
        return self.hidden_width + self.f_h(input_size) + self.f_y(input_size)


def ff(input_size, output_size, widths,
       nonlinearity=tf.nn.relu, normalise=True):
    """A standard feed-forward neural net.

    Args:
        input_size (int): Size of input.
        output_size (int): Size of output.
        widths (tuple of int): Widths of the layers.
        nonlinearity (function): Nonlinearity to use.
        normalise (bool): Interleave with normalisation layers.
    """
    layers = []
    for width in widths:
        layers.append(Dense(width, nonlinearity=nonlinearity))
        if normalise:
            layers.append(Normalise())
    layers.append(Linear(output_size))
    return Net(input_size, layers)


def rnn(input_size, output_size, widths, nonlinearity=tf.nn.sigmoid):
    """A standard recurrent neural net.

    Args:
        input_size (int): Size of input.
        output_size (int): Size of output.
        widths (tuple of tuple): Widths of the layers.
        nonlinearity (function): Nonlinearity to use.
    """
    layers = []
    for width, hidden_width in widths:
        layers.append(Elman(width, hidden_width, nonlinearity=nonlinearity))
    layers.append(Linear(output_size))
    return Net(input_size, layers)
