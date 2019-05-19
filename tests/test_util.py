# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, allclose
from . import inv_perm


def test_inv_perm():
    perm = np.random.permutation(10)
    yield allclose, perm[inv_perm(perm)], np.arange(10)
