# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

import lab.tensorflow as B
import six
import tensorflow as tf


class Layer(six.with_metaclass(ABCMeta, object)):
    """A layer."""

    @abstractmethod
    def initialise(self, input_size, vs):
        """Initialise the weights of the layer.

        Args:
            input_size (int): Size of the input.
            vs (:class:`varz.Vars`): Variable container.
        """

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def num_weights(self, input_size):
        """Get the number of weights used in the layer.

        Args:
            input_size (int): Size of the input.

        Returns:
            int: Number of weights.
        """


class AutoBatchLayer(six.with_metaclass(ABCMeta, Layer)):
    """A layer that automatically handles batched calls."""

    def __call__(self, x):
        # Handle the rank one case.
        if B.rank(x) == 1:
            x = B.expand_dims(x, 1)

        if B.rank(x) == 2:
            return self._apply(x)
        elif B.rank(x) == 3:
            # Shape into one big matrix for efficient application.
            batch_size, n, m = B.shape(x)
            x = B.reshape(x, batch_size * n, m)

            # Apply.
            x = self._apply(x)

            # Shape back and return.
            return B.reshape(x, batch_size, n, m)
        else:
            raise ValueError('I do not know how to handle inputs of rank {}.'
                             ''.format(B.rank(x)))

    @abstractmethod
    def _apply(self, x):
        pass


class Normalise(AutoBatchLayer):
    """Batch-norm layer."""

    def __init__(self):
        self.width = None

    def initialise(self, input_size, vs):
        self.width = input_size

    def _apply(self, x):
        mean = B.mean(x, axis=0)[None, :]
        std = B.std(x, axis=0)[None, :]
        return (x - mean) / (std + B.epsilon)

    def num_weights(self, input_size):
        self.width = input_size
        return 0


class Linear(AutoBatchLayer):
    """Linear layer.

    Args:
        width (int): Width of the layer.
    """

    def __init__(self, width):
        self.width = width
        self.A = None
        self.b = None

    def initialise(self, input_size, vs):
        self.A = vs.get(shape=[input_size, self.width])
        self.b = vs.get(shape=[1, self.width])

    def _apply(self, x):
        return B.matmul(x, self.A) + self.b

    def num_weights(self, input_size):
        return input_size * self.width + self.width


class Activation(AutoBatchLayer):
    """Activation layer.

    Args:
        nonlinearity (function, optional): Nonlinearity. Defaults to `B.relu`.
    """

    def __init__(self, nonlinearity=B.relu):
        self.width = None
        self.nonlinearity = nonlinearity

    def initialise(self, input_size, vs):
        self.width = input_size

    def _apply(self, x):
        return self.nonlinearity(x)

    def num_weights(self, input_size):
        self.width = input_size
        return 0


class Net(Layer):
    """A composition of layers.

    Args:
        layers (:class:`.net.Layer`): List of the layers to compose.
    """

    def __init__(self, *layers):
        self.layers = layers

    def initialise(self, input_size, vs):
        # Initialise layers.
        for layer in self.layers:
            layer.initialise(input_size, vs)
            input_size = layer.width

    def __call__(self, x):
        # Apply layers one by one.
        for layer in self.layers:
            x = layer(x)
        return x

    def num_weights(self, input_size):
        num_weights = 0
        for layer in self.layers:
            num_weights += layer.num_weights(input_size)
            input_size = layer.width
        return num_weights


class Recurrent(Layer):
    """An recurrent layer.

    Args:
        cell (:class:`.net.Cell`): Cell of the recurrent layer.
        h0 (tensor, optional): Initial hidden state. Defaults to a variable
            that will be learned.
    """

    def __init__(self, cell, h0=None):
        self.cell = cell
        self.h0 = h0

    def initialise(self, input_size, vs):
        if self.h0 is None:
            self.h0 = vs.get(shape=[1, self.cell.width])
        self.cell.initialise(input_size, vs)

    def __call__(self, x):
        # Put the batch dimension second.
        if B.rank(x) == 1:
            x = x[:, None, None]
        elif B.rank(x) == 2:
            x = x[:, None, :]
        elif B.rank(x) == 3:
            x = B.transpose(x, perm=(1, 0, 2))
        else:
            raise ValueError('Cannot handle inputs of rank {}.'
                             ''.format(B.rank(x)))

        # Recurrently apply the cell.
        n, batch_size, m = B.shape(x)
        y0 = B.zeros(B.dtype(x), batch_size, self.cell.width)
        h0 = tf.tile(self.h0, [batch_size, 1])
        res = tf.scan(self.cell, x, initializer=(h0, y0))[1]

        # Put the batch dimension first again.
        return B.transpose(res, perm=(1, 0, 2))

    def num_weights(self, input_size):
        return self.cell.width + self.cell.num_weights(input_size)


class Cell(six.with_metaclass(ABCMeta, object)):
    """An abstract cell of a recurrent layer."""

    @abstractmethod
    def initialise(self, input_size, vs):
        """Initialise the weights of the cell.

        Args:
            input_size (int): Size of the input.
            vs (:class:`varz.Vars`): Variable container.
        """

    @abstractmethod
    def __call__(self, prev, x):
        pass

    @abstractmethod
    def num_weights(self, input_size):
        """Get the number of weights used in the cell.

        Args:
            input_size (int): Size of the input.

        Returns:
            int: Number of weights.
        """


class GRU(Cell):
    """A gated recurrent unit.

    Args:
        width (int): Width of the hidden.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.tanh`.
    """

    def __init__(self, width, nonlinearity=B.tanh):
        self.width = width
        self.f_z = Net(Linear(self.width), Activation(B.sigmoid))
        self.f_r = Net(Linear(self.width), Activation(B.sigmoid))
        self.f_h = Net(Linear(self.width), Activation(nonlinearity))

    def initialise(self, input_size, vs):
        self.f_z.initialise(self.width + input_size, vs)
        self.f_r.initialise(self.width + input_size, vs)
        self.f_h.initialise(self.width + input_size, vs)

    def __call__(self, prev, x):
        prev_h, _ = prev

        # Gate logic:
        z = self.f_z(B.concat(prev_h, x, axis=1))
        r = self.f_r(B.concat(prev_h, x, axis=1))
        h = (1 - z) * prev_h + z * self.f_h(B.concat(r * prev_h, x, axis=1))
        y = h

        return h, y

    def num_weights(self, input_size):
        return self.f_z.num_weights(self.width + input_size) + \
               self.f_r.num_weights(self.width + input_size) + \
               self.f_h.num_weights(self.width + input_size)


class Elman(Cell):
    """An recurrent layer of the Elman type.

    Args:
        width (int): Width of the layer.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.tanh`.
    """

    def __init__(self, width, nonlinearity=B.tanh):
        self.width = width
        self.f_h = Net(Linear(self.width), Activation(nonlinearity))

    def initialise(self, input_size, vs):
        self.f_h.initialise(self.width + input_size, vs)

    def __call__(self, prev, x):
        prev_h, prev_y = prev

        # Gate logic:
        h = self.f_h(B.concat(prev_h, x, axis=1))
        y = h

        return h, y

    def num_weights(self, input_size):
        return self.f_h.num_weights(self.width + input_size)


def ff(output_size, widths, nonlinearity=B.relu, normalise=True):
    """A standard feed-forward neural net.

    Args:
        output_size (int): Size of output.
        widths (tuple of int): Widths of the layers.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.relu`.
        normalise (bool, optional): Interleave with normalisation layers.
            Defaults to `True`.

    Returns:
        :class:`.net.Net`: Feed-forward NN.
    """
    layers = []

    for width in widths:
        layers.append(Linear(width))
        layers.append(Activation(nonlinearity))

        # Add a batch-norm layer, if asked for.
        if normalise:
            layers.append(Normalise())

    # Add a final linear layer to scale outputs.
    layers.append(Linear(output_size))
    return Net(*layers)


def rnn(output_size,
        widths,
        nonlinearity=B.relu,
        normalise=False,
        gru=True,
        final_dense=False):
    """A standard recurrent neural net.

    Args:
        output_size (int): Size of output.
        widths (tuple of int): Widths of the layers.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.relu`.
        normalise (bool, optional): Interleave with normalisation layers.
            Defaults to `False`.
        gru (bool, optional): Use GRU layers instead of Elman layers. Defaults
            to `True`.
        final_dense (bool, optional): Append a final dense layer. Defaults to
            `False`.

    Returns:
        :class:`.net.Net`: RNN.
    """
    layer_type = GRU if gru else Elman
    layers = []

    for width in widths:
        layers.append(Recurrent(layer_type(width)))
        layers.append(Activation(nonlinearity))

        # Add a batch-norm layer, if asked for.
        if normalise:
            layers.append(Normalise())

    # Add a final dense layer, if asked for.
    if final_dense:
        layers.append(Linear(output_size))
        layers.append(Activation(nonlinearity))

    # Add a final linear layer to scale outputs.
    layers.append(Linear(output_size))
    return Net(layers)
