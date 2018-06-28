# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul
from itertools import product

import numpy as np
import tensorflow as tf
from lab.tf import B

__all__ = ['Packer', 'Vars', 'vars32', 'vars64', 'VarsFrom', 'inv_perm',
           'identity', 'map_cols', 'Initialiser']


class Packer(object):
    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]
        self._lengths = [B.length(obj) for obj in objs]

    def pack(self, *objs):
        return tf.concat([B.reshape(obj, [-1]) for obj in objs], axis=0)

    def unpack(self, package):
        i, outs = 0, []
        for shape, length in zip(self._shapes, self._lengths):
            outs.append(B.reshape(package[i:i + length], shape))
            i += length
        return outs


class Initialiser(object):
    """Variable initialiser."""

    def __init__(self):
        self._assignments = {}

    def assign(self, name, values):
        """Assign values to a particular variable.

        Args:
            name (str): Name of variables.
            values (list[tensor]): List of values to assign.
        """
        self._assignments[name] = values

    def generate(self, vs):
        """Generate initialisers.

        Args:
            vs (:class:`.util.Vars`): Variable storage.

        Returns:
            list[list[tensor]]: List of initialisers. Run a initialiser with a
                TensorFlow session to set the initialisation.
        """
        names, values = zip(*self._assignments.items())
        return [[vs.assign(name, val) for name, val in zip(names, value)]
                for value in product(*values)]


class Vars(object):
    """Variable storage manager.

    Args:
        dtype (data type, optional): Data type of the variables. Defaults to
            `np.float32`.
        epsilon (float, optional): Epsilon for various things. Defaults to
            `1e-6`.
    """

    def __init__(self, dtype=np.float32, epsilon=1e-6):
        self.vars = []
        self.dtype = dtype
        self.epsilon = epsilon
        self.vars_by_name = {}
        self.assigners = {}

    def init(self, session):
        """Initialise the variables.

        Args:
            session (:class:`tf.Session`): TensorFlow session.
        """
        session.run(tf.variables_initializer(self.vars))

    def get(self, init=None, shape=(), dtype=None, name=None):
        """Get a variable.

        Args:
            init (tensor, optional): Initialisation of the variable. Defaults
                to a N(0, 1) draw.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """
        # Resolve data type.
        dtype = self.dtype if dtype is None else dtype

        # Generate initialisation if necessary.
        init = B.randn(shape, dtype=dtype) if init is None else init

        # Generate the latent variable and store it.
        latent = B.Variable(B.cast(init, dtype=dtype), dtype=dtype)
        self.vars.append(latent)

        # Contruct the observed variable.
        observed = latent

        # If a name is given, generate an assignment method and store it by
        # name.
        if name is not None:
            def assign(value):
                return B.assign(latent, value)

            self.assigners[name] = assign
            self.vars_by_name[name] = observed

        return observed

    def assign(self, name, value):
        """Assign a value to a variable.

        Args:
            name (str): Name of variable to assign value to.
            value (tensor): Value to assign.

        Returns:
            tensor: TensorFlow tensor that can be run to perform the assignment.
        """
        return self.assigners[name](value)

    def positive(self, init=None, shape=(), dtype=None, name=None):
        """Get a positive variable.

        Args:
            init (tensor, optional): Initialisation of the variable. Defaults
                to a Uniform(0, 1) draw.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """
        # Resolve data type.
        dtype = self.dtype if dtype is None else dtype

        # Generate initialisation if necessary.
        if init is None:
            init = B.log(B.rand(shape, dtype=dtype))
        else:
            init = B.log(init)

        # Generate the latent variable and store it.
        latent = B.Variable(B.cast(init, dtype=dtype), dtype=dtype)
        self.vars.append(latent)

        # Construct the observed variable.
        observed = B.exp(latent)

        # If a name is given, generate an assignment method and store it by
        # name.
        if name is not None:
            def assign(value):
                return B.assign(latent, B.log(value))

            self.assigners[name] = assign
            self.vars_by_name[name] = observed

        return observed

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.util.Vars.positive`."""
        return self.positive(*args, **kw_args)

    def __getitem__(self, name):
        """Get a variable by name.

        Args:
            name (str): Name of variable.

        Returns:
            tensor: Variable.
        """
        return self.vars_by_name[name]


class VarsFrom(object):
    def __init__(self, source):
        self._source = source
        self._i = 0

    def get(self, shape):
        length = reduce(mul, shape, 1)
        out = B.reshape(self._source[self._i: self._i + length], shape)
        self._i += length
        return out


vars32 = Vars(np.float32)
vars64 = Vars(np.float64)


def inv_perm(perm):
    out = [0 for _ in range(len(perm))]
    for i, p in enumerate(perm):
        out[p] = i
    return out


def identity(x):
    return x


def map_cols(f, xs):
    return tf.map_fn(lambda x: f(x[:, None]), B.transpose(xs))
