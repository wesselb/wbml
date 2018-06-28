# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from wbml import Vars, Packer, Initialiser
from lab.tf import B
from itertools import product

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, eprint


def test_vars():
    s = tf.Session()

    vs = Vars(np.float32)
    a = vs.get(1.)
    b = vs.get(2.)
    yield eq, len(vs.vars), 2
    yield eq, a.dtype.as_numpy_dtype, np.float32
    yield eq, b.dtype.as_numpy_dtype, np.float32
    vs.init(s)

    vs = Vars(np.float64)
    a = vs.get(1.)
    b = vs.get(2.)
    yield eq, len(vs.vars), 2
    yield eq, a.dtype.as_numpy_dtype, np.float64
    yield eq, b.dtype.as_numpy_dtype, np.float64
    vs.init(s)


def test_vars_positive():
    s = tf.Session()
    vs = Vars()
    xs = [vs.positive() for _ in range(10)]
    vs.init(s)
    yield ok, all([x > 0 for x in s.run(xs)])


def test_vars_assignment():
    s = tf.Session()
    vs = Vars()

    # Generate some variables.
    vs.get(1., name='a')
    vs.pos(2., name='b')
    vs.init(s)

    yield eq, 1., s.run(vs['a'])
    yield eq, 2., s.run(vs['b'])

    # Assign new values.
    s.run(vs.assign('a', 3.))
    s.run(vs.assign('b', 4.))

    yield eq, 3., s.run(vs['a'])
    yield eq, 4., s.run(vs['b'])


def test_initialiser():
    s = tf.Session()
    vs = Vars()
    init = Initialiser()

    a = vs.get(1., name='a')
    b = vs.pos(2., name='b')
    vs.init(s)

    init.assign('a', [3., 4.])
    init.assign('b', [5., 6.])
    inits = init.generate(vs)

    for initialiser, values in zip(inits, product([3., 4.], [5., 6.])):
        s.run(initialiser)
        yield eq, s.run(a), values[0]
        yield eq, s.run(b), values[1]


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
