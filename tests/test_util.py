# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from wbml import Vars

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
