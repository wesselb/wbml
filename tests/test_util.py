import lab as B
import numpy as np
from stheno import Normal

from wbml.util import inv_perm, normal1d_logpdf, BatchVars
from .util import approx


def test_inv_perm():
    perm = np.random.permutation(10)
    approx(perm[inv_perm(perm)], B.range(10))


def test_normal1d_logpdf():
    means = B.randn(3, 3)
    covs = B.randn(3, 3) ** 2
    x = B.randn(3, 3)
    logpdfs = normal1d_logpdf(x, covs, means)
    for i in range(3):
        for j in range(3):
            dist = Normal(means[i : i + 1, j : j + 1], covs[i : i + 1, j : j + 1])
            approx(logpdfs[i, j], dist.logpdf(x[i, j]))


def test_batchvars():
    source = B.randn(5, 2 + 3 * 4)
    vs = BatchVars(source=source)
    approx(vs.get(shape=(1, 2)), B.reshape(source[:, :2], 5, 1, 2))
    approx(vs.get(shape=(3, 4)), B.reshape(source[:, 2:], 5, 3, 4))
