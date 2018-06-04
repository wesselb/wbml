# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
from lab.tf import B

__all__ = ['Packer', 'Vars', 'vars32', 'vars64', 'VarsFrom']


class Packer(object):
    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]
        self._ranks = [B.rank(obj) for obj in objs]
        self._lengths = [Packer.calc_length(shape, rank)
                         for shape, rank in zip(self._shapes, self._ranks)]

    @staticmethod
    def calc_length(shape, rank):
        length = 1
        for i in range(rank):
            length *= shape[i]
        return length

    def pack(self, *objs):
        return tf.concat([B.reshape(obj, [-1]) for obj in objs], axis=0)

    def unpack(self, package):
        i, outs = 0, []
        for shape, length in zip(self._shapes, self._lengths):
            outs.append(B.reshape(package[i:i + length], shape))
            i += length
        return outs


class Vars(object):
    def __init__(self, dtype=np.float32, epsilon=1e-6):
        self.latents = []
        self.dtype = dtype
        self.epsilon = epsilon

    def init(self, session):
        session.run(tf.variables_initializer(self.latents))

    def get(self, init=None, shape=(), dtype=None):
        dtype = self._resolve_dtype(dtype)
        init = B.randn(shape, dtype=dtype) if init is None else init
        return self._generate(init, dtype)

    def positive(self, init=None, shape=(), dtype=None):
        dtype = self._resolve_dtype(dtype)
        init = B.rand(shape, dtype=dtype) if init is None else init
        return B.exp(self._generate(B.log(init), dtype)) + \
               B.cast(self.epsilon, dtype=dtype)

    def pos(self, *args, **kw_args):
        return self.positive(*args, **kw_args)

    def _generate(self, init, dtype):
        latent = B.Variable(B.cast(init, dtype=dtype), dtype=dtype)
        self.latents.append(latent)
        return latent

    def _resolve_dtype(self, dtype):
        return self.dtype if dtype is None else dtype


class VarsFrom(object):
    def __init__(self, resource):
        self._resource = resource
        self._i = 0

    def get(self, shape):
        length = reduce(mul, shape, 1)
        out = B.reshape(self._resource[self._i: self._i + length], shape)
        self._i += length
        return out


vars32 = Vars(np.float32)
vars64 = Vars(np.float64)
