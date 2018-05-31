# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

import tensorflow as tf
from lab import B


def identity(x):
    return x


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def initialise(self, input_size):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def vars(self):
        pass


class Identity(Layer):
    def __init__(self):
        pass

    def initialise(self, *arg):
        pass

    def weights(self):
        return []

    def vars(self):
        return []

    def __call__(self, x):
        return identity(x)


class Normalise(Layer):
    def __init__(self):
        self.width = None

    def initialise(self, input_size):
        self.width = input_size

    def weights(self):
        return []

    def vars(self):
        return []

    def __call__(self, x):
        mean = B.mean(x, axis=0)[None, :]
        std = B.mean((x - mean) ** 2, axis=0)[None, :] ** .5
        return (x - mean) / (std + B.epsilon)


class Dense(Layer):
    def __init__(self, width, s2=1., nonlinearity=tf.nn.relu):
        self.width = width
        self._A = None
        self._b = None
        self._s2 = s2
        self._nonlinearity = nonlinearity

    def initialise(self, input_size):
        self._A = B.Variable(self._s2 ** .5 * B.randn([self.width, input_size]))
        self._b = B.Variable(self._s2 ** .5 * B.randn([self.width, 1]))

    def __call__(self, x):
        return self._nonlinearity(B.matmul(self._A, x) + self._b)

    def weights(self):
        return B.concat([B.reshape(self._A, [-1]),
                         B.reshape(self._b, [-1])], axis=0)

    def vars(self):
        return [self._A, self._b]


class AbstractNet(object):
    __metaclass__ = ABCMeta

    def __init__(self, input_size, *layers):
        self._input_size = input_size
        self._layers = layers

        # Initialise layers.
        cur_size = input_size
        for layer in layers:
            layer.initialise(cur_size)
            cur_size = layer.width

    def weights(self):
        return B.concat([x.weights() for x in self._layers], axis=0)

    def vars(self):
        return sum([layer.vars() for layer in self._layers], [])


class Net(AbstractNet):
    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
