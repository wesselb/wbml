# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
from lab.tf import B

__all__ = ['Packer', 'Vars', 'vars32', 'vars64', 'VarsFrom', 'inv_perm',
           'identity', 'map_cols']


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
        self.names = dict()

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
        dtype = self._resolve_dtype(dtype)
        init = B.randn(shape, dtype=dtype) if init is None else init
        return self._generate(init, dtype, name)

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
        dtype = self._resolve_dtype(dtype)
        init = B.rand(shape, dtype=dtype) if init is None else init
        return B.exp(self._generate(B.log(init), dtype, name)) + \
               B.cast(self.epsilon, dtype=dtype)

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.util.Vars.positive`."""
        return self.positive(*args, **kw_args)

    def _generate(self, init, dtype, name):
        latent = B.Variable(B.cast(init, dtype=dtype), dtype=dtype)
        self.vars.append(latent)
        if name is not None:
            self.names[name] = latent
        return latent

    def _resolve_dtype(self, dtype):
        return self.dtype if dtype is None else dtype

    def __getitem__(self, name):
        """Get a variable by name.

        Args:
            name (str): Name of variable.

        Returns:
            tensor: Variable.
        """
        return self.names[name]


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
