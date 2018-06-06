# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from wbml import Vars, Packer
from lab.tf import B

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, eprint


def test_vars():
    s = tf.Session()

    vs = Vars(np.float32)
    a = vs.get(1.)
    b = vs.get(2.)
    yield eq, len(vs.latents), 2
    yield eq, a.dtype.as_numpy_dtype, np.float32
    yield eq, b.dtype.as_numpy_dtype, np.float32
    vs.init(s)

    vs = Vars(np.float64)
    a = vs.get(1.)
    b = vs.get(2.)
    yield eq, len(vs.latents), 2
    yield eq, a.dtype.as_numpy_dtype, np.float64
    yield eq, b.dtype.as_numpy_dtype, np.float64
    vs.init(s)


def test_vars_positive():
    s = tf.Session()
    vs = Vars()
    xs = [vs.positive() for _ in range(10)]
    vs.init(s)
    yield ok, all([x > 0 for x in s.run(xs)])


def test_packer():
    s = tf.Session()
    a, b, c = tf.ones((5, 10)), tf.ones((20)), tf.ones((5, 1, 15))
    packer = Packer(a, b, c)

    # Test packing.
    packed = packer.pack(a, b, c)
    yield eq, B.rank(packed), 1

    # Test unpacking.
    a, b, c = packer.unpack(packed)
    yield eq, B.shape_int(a), (5, 10)
    yield eq, B.shape_int(b), (20,)
    yield eq, B.shape_int(c), (5, 1, 15)
