from abc import ABCMeta, abstractmethod

import lab as B

__all__ = [
    "Normalise",
    "Linear",
    "Activation",
    "Net",
    "Recurrent",
    "GRU",
    "Elman",
    "ff",
    "rnn",
]


class Layer(metaclass=ABCMeta):
    """A layer."""

    @abstractmethod
    def initialise(self, input_size, vs):
        """Initialise the weights of the layer.

        Args:
            input_size (int): Size of the input.
            vs (:class:`varz.Vars`): Variable container.
        """

    @abstractmethod
    def __call__(self, x):  # pragma: no cover
        pass

    @abstractmethod
    def num_weights(self, input_size):
        """Get the number of weights used in the layer.

        Args:
            input_size (int): Size of the input.

        Returns:
            int: Number of weights.
        """

    @property
    @abstractmethod
    def width(self):
        """Width of the layer."""


class Normalise(Layer):
    """Normalisation layer.

    Args:
        epsilon (scalar): Small value to add to the standard deviation before
            dividing. Defaults to `B.epsilon`.
    """

    def __init__(self, epsilon=B.epsilon):
        self._width = None
        self.epsilon = epsilon

    def initialise(self, input_size, vs):
        self._width = input_size

    def __call__(self, x):
        mean = B.mean(x, axis=2)[:, :, None]
        std = B.std(x, axis=2)[:, :, None]
        return (x - mean) / (std + self.epsilon)

    def num_weights(self, input_size):
        self._width = input_size
        return 0

    @property
    def width(self):
        return self._width


class Linear(Layer):
    """Linear layer.

    Args:
        width (int): Width of the layer.
    """

    def __init__(self, width):
        self._width = width
        self.A = None
        self.b = None

    def initialise(self, input_size, vs):
        self.A = vs.get(shape=[input_size, self.width])
        self.b = vs.get(shape=[1, self.width])

    def __call__(self, x):
        return B.matmul(x, self.A) + self.b

    def num_weights(self, input_size):
        return input_size * self.width + self.width

    @property
    def width(self):
        return self._width


class Activation(Layer):
    """Activation layer.

    Args:
        nonlinearity (function, optional): Nonlinearity. Defaults to `B.relu`.
    """

    def __init__(self, nonlinearity=B.relu):
        self._width = None
        self.nonlinearity = nonlinearity

    def initialise(self, input_size, vs):
        self._width = input_size

    def __call__(self, x):
        return self.nonlinearity(x)

    def num_weights(self, input_size):
        self._width = input_size
        return 0

    @property
    def width(self):
        return self._width


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
        x_rank = B.rank(x)
        if x_rank == 1:
            x = x[None, :, None]
        elif x_rank == 2:
            x = x[None, :, :]
        elif x_rank == 3:
            pass
        else:
            raise ValueError("Cannot handle inputs of rank {}.".format(x_rank))

        # Apply layers one by one.
        for layer in self.layers:
            x = layer(x)

        # Remove batch dimension, if that wasn't specified initially.
        if x_rank != 3:
            x = x[0, :, :]

        return x

    def num_weights(self, input_size):
        num_weights = 0
        for layer in self.layers:
            num_weights += layer.num_weights(input_size)
            input_size = layer.width
        return num_weights

    @property
    def width(self):
        return self.layers[-1].width


class Recurrent(Layer):
    """A recurrent layer.

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
        x_rank = B.rank(x)
        if x_rank == 2:
            x = x[:, None, :]
        elif x_rank == 3:
            x = B.transpose(x, perm=(1, 0, 2))
        else:
            raise ValueError(f"Cannot handle inputs of rank {B.rank(x)}.")

        # Recurrently apply the cell.
        n, batch_size, m = B.shape(x)
        y0 = B.zeros(B.dtype(x), batch_size, self.cell.width)
        h0 = B.tile(self.h0, batch_size, 1)
        res = B.scan(self.cell, x, h0, y0)[1]

        # Put the batch dimension first again.
        res = B.transpose(res, perm=(1, 0, 2))

        # Remove the batch dimension, if that didn't exist before.
        if x_rank == 2:
            res = res[0, :, :]

        return res

    def num_weights(self, input_size):
        return self.cell.width + self.cell.num_weights(input_size)

    @property
    def width(self):
        return self.cell.width


class Cell(metaclass=ABCMeta):
    """An abstract cell of a recurrent layer."""

    @abstractmethod
    def initialise(self, input_size, vs):
        """Initialise the weights of the cell.

        Args:
            input_size (int): Size of the input.
            vs (:class:`varz.Vars`): Variable container.
        """

    @abstractmethod
    def __call__(self, prev, x):  # pragma: no cover
        pass

    @abstractmethod
    def num_weights(self, input_size):
        """Get the number of weights used in the cell.

        Args:
            input_size (int): Size of the input.

        Returns:
            int: Number of weights.
        """

    @property
    @abstractmethod
    def width(self):
        """Width of the layer"""


class GRU(Cell):
    """A gated recurrent unit.

    Args:
        width (int): Width of the hidden.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.tanh`.
    """

    def __init__(self, width, nonlinearity=B.tanh):
        self._width = width
        self.f_z = Net(Linear(width), Activation(B.sigmoid))
        self.f_r = Net(Linear(width), Activation(B.sigmoid))
        self.f_h = Net(Linear(width), Activation(nonlinearity))

    def initialise(self, input_size, vs):
        self.f_z.initialise(self._width + input_size, vs)
        self.f_r.initialise(self._width + input_size, vs)
        self.f_h.initialise(self._width + input_size, vs)

    def __call__(self, prev, x):
        prev_h, _ = prev

        # Gate logic:
        z = self.f_z(B.concat(prev_h, x, axis=1))
        r = self.f_r(B.concat(prev_h, x, axis=1))
        h = (1 - z) * prev_h + z * self.f_h(B.concat(r * prev_h, x, axis=1))
        y = h

        return h, y

    def num_weights(self, input_size):
        return (
            self.f_z.num_weights(self._width + input_size)
            + self.f_r.num_weights(self._width + input_size)
            + self.f_h.num_weights(self._width + input_size)
        )

    @property
    def width(self):
        return self._width


class Elman(Cell):
    """Elman cell.

    Args:
        width (int): Width of the layer.
        nonlinearity (function, optional): Nonlinearity to use. Defaults to
            `B.tanh`.
    """

    def __init__(self, width, nonlinearity=B.tanh):
        self._width = width
        self.f_h = Net(Linear(width), Activation(nonlinearity))

    def initialise(self, input_size, vs):
        self.f_h.initialise(self._width + input_size, vs)

    def __call__(self, prev, x):
        prev_h, prev_y = prev

        # Gate logic:
        h = self.f_h(B.concat(prev_h, x, axis=1))
        y = h

        return h, y

    def num_weights(self, input_size):
        return self.f_h.num_weights(self._width + input_size)

    @property
    def width(self):
        return self._width


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

        # Add a normalisation layer, if asked for.
        if normalise:
            layers.append(Normalise())

    # Add a final linear layer to scale outputs.
    layers.append(Linear(output_size))
    return Net(*layers)


def rnn(
    output_size,
    widths,
    nonlinearity=B.relu,
    normalise=False,
    gru=True,
    final_dense=False,
):
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

        # Add a normalisation layer, if asked for.
        if normalise:
            layers.append(Normalise())

    # Add a final dense layer, if asked for.
    if final_dense:
        layers.append(Linear(output_size))
        layers.append(Activation(nonlinearity))

    # Add a final linear layer to scale outputs.
    layers.append(Linear(output_size))
    return Net(*layers)
