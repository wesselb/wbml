# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numpy.testing import assert_allclose, assert_array_almost_equal
from stheno.matrix import Dense, dense
from plum import Dispatcher

__all__ = ['allclose', 'approx']

_dispatch = Dispatcher()


@_dispatch(object)
def to_numpy(x):
    return x


@_dispatch(Dense)
def to_numpy(x):
    return dense(x)


def allclose(x, y):
    return assert_allclose(to_numpy(x), to_numpy(y))


def approx(x, y, digits=7):
    return assert_array_almost_equal(x, y, decimal=digits)
