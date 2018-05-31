# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul

import tensorflow as tf
from lab import B


def create_var(shape):
    return B.Variable(B.randn(shape))


class Packer:
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
