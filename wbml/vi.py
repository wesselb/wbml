# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lab.tf import B
from plum import Dispatcher
from stheno import Normal

dispatch = Dispatcher()

__all__ = ['elbo']


@dispatch(object, object, object, [object])
def elbo(lik, p, q, num_samples=1):
    samples = q.sample(num_samples)
    log_lik = B.sum(tf.map_fn(lambda z: lik(z[:, None]),
                              B.transpose(samples)))
    log_prior = B.sum(p.log_pdf(samples))
    log_q = B.sum(q.log_pdf(samples))
    return (log_lik - log_prior + log_q) / num_samples


@dispatch(object, Normal, Normal, [object])
def elbo(lik, p, q, num_samples=1):
    samples = q.sample(num_samples)
    log_lik = B.sum(tf.map_fn(lambda z: lik(z[:, None]),
                              B.transpose(samples)))
    return log_lik / num_samples - p.kl(q)
