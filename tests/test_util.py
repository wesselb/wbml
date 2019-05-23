# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from stheno import Normal
import lab as B

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, allclose
from . import inv_perm, normal1d_logpdf, BatchVars


def test_inv_perm():
    perm = np.random.permutation(10)
    yield allclose, perm[inv_perm(perm)], B.range(10)


def test_normal1d_logpdf():
    means = B.randn(3, 3)
    covs = B.randn(3, 3) ** 2
    x = B.randn(3, 3)
    logpdfs = normal1d_logpdf(x, covs, means)
    for i in range(3):
        for j in range(3):
            dist = Normal(covs[i:i + 1, j:j + 1], means[i:i + 1, j:j + 1])
            yield allclose, logpdfs[i, j], dist.logpdf(x[i, j])


def test_batchvars():
    source = B.randn(5, 2 + 3 * 4)
    vs = BatchVars(source=source)
    yield allclose, vs.get(shape=(1, 2)), B.reshape(source[:, :2], 5, 1, 2)
    yield allclose, vs.get(shape=(3, 4)), B.reshape(source[:, 2:], 5, 3, 4)
