# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
from lab import B

__all__ = ['Packer', 'Vars', 'vars32', 'vars64']


class Packer(object):
    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]

    def pack(self, *objs):
        return tf.concat([B.reshape(obj, [-1]) for obj in objs], axis=0)

    def unpack(self, package):
        i, outs = 0, []
        for shape in self._shapes:
            length = reduce(mul, shape, 1)
            outs.append(B.reshape(package[i:i + length], shape))
            i += length
        return outs


class Vars(object):
    def __init__(self, dtype=None):
        self.latents = []
        self.dtype = dtype

    def get(self, init=None, shape=(), dtype=None):
        return self._generate(init, shape, dtype)

    def positive(self, init=None, shape=(), dtype=None):
        return B.exp(self._generate(init, shape, dtype, np.log))

    def _generate(self, init, shape, dtype, f=lambda x: x):
        init = B.randn(shape) if init is None else f(init)
        latent = B.Variable(init, dtype=dtype)
        self.latents.append(latent)
        return latent


vars32 = Vars(np.float32)
vars64 = Vars(np.float64)
