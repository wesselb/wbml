# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab.tf import B
from plum import Dispatcher
from stheno import Normal

from .util import map_cols

dispatch = Dispatcher()

__all__ = ['elbo']


@dispatch(object, object, object, [object])
def elbo(lik, p, q, num_samples=1):
    samples = q.sample(num_samples)
    log_lik = B.mean(map_cols(lik, samples))
    log_prior = B.mean(p.log_pdf(samples))
    log_q = B.mean(q.log_pdf(samples))
    return log_lik - log_prior + log_q


@dispatch(object, Normal, Normal, [object])
def elbo(lik, p, q, num_samples=1):
    return B.mean(map_cols(lik, q.sample(num_samples))) - q.kl(p)
