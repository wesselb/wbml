# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from wbml.vi import elbo
from stheno import Normal
import lab as B
from .util import allclose


def rand_normal(n=3):
    cov = B.randn(n, n)
    cov = B.mm(cov, cov, tr_b=True)
    return Normal(cov, B.randn(n, 1))


def test_elbo():
    lik = rand_normal()
    p = rand_normal()
    q = rand_normal()

    # Check that the two implementations are consistent.
    estimate1 = elbo.invoke(object, Normal, Normal)(lik.logpdf, p, q,
                                                    num_samples=50000)
    estimate2 = elbo.invoke(object, object, object)(lik.logpdf, p, q,
                                                    num_samples=50000)
    allclose(estimate1, estimate2, rtol=1e-2)
