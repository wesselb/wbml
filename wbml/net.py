# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

import tensorflow as tf
from lab import B
from wbml import vars32


def identity(x):
    return x


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


class Identity(Layer):
    def __init__(self):
        pass

    def initialise(self, *arg):
        pass

    def weights(self):
        return []

    def __call__(self, x):
        return identity(x)


class Normalise(Layer):
    def __init__(self):
        self.width = None

    def initialise(self, input_size, vars):
        self.width = input_size

    def weights(self):
        return []

    def __call__(self, x):
        mean = B.mean(x, axis=0)[None, :]
        std = B.mean((x - mean) ** 2, axis=0)[None, :] ** .5
        return (x - mean) / (std + B.epsilon)


class Dense(Layer):
    def __init__(self, width, nonlinearity=tf.nn.relu):
        self.width = width
        self.A = None
        self.b = None
        self.nonlinearity = nonlinearity

    def initialise(self, input_size, vars):
        self.A = vars.get(shape=[self.width, input_size])
        self.b = vars.get(shape=[self.width, 1])

    def __call__(self, x):
        return self.nonlinearity(B.matmul(self.A, x) + self.b)

    def weights(self):
        return B.concat([B.reshape(self.A, [-1]), B.reshape(self.b, [-1])])


class AbstractNet(object):
    __metaclass__ = ABCMeta

    def __init__(self, input_size, layers, vars=vars32):
        self.input_size = input_size
        self.layers = layers

        # Initialise layers.
        cur_size = input_size
        for layer in layers:
            layer.initialise(cur_size, vars)
            cur_size = layer.width

    @property
    def output_size(self):
        """Size of the output."""
        return self.layers[-1].width

    def weights(self):
        return B.concat([x.weights() for x in self.layers], axis=0)


class Net(AbstractNet):
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def feedforward(widths, vars=vars32):
    """A standard feedforward neural net.

    Args:
        widths (list of int): Widths of the layers. Must give at least two
            widths.
        vars (instance of :class:`.util.Vars`, optional): Variable storage.
            Defaults to the global `vars32`.
    """
    if len(widths) < 2:
        raise ValueError('Must specify at least width of input and output.')
    layers = []
    for w in widths[1:-1]:
        layers.append(Dense(w))
        layers.append(Normalise())
    layers.append(Dense(widths[-1], nonlinearity=identity))
    return Net(widths[0], layers, vars=vars)
