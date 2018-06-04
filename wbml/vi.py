# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lab.tf import B
from plum import Dispatcher
from stheno import Normal

dispatch = Dispatcher()

__all__ = ['elbo']


def map_fn_columns(f, xs):
    return tf.map_fn(lambda x: f(x[:, None]), B.transpose(xs))


@dispatch(object, object, object, [object])
def elbo(lik, p, q, num_samples=1):
    samples = q.sample(num_samples)
    log_lik = B.mean(map_fn_columns(lik, samples))
    log_prior = B.mean(p.log_pdf(samples))
    log_q = B.mean(q.log_pdf(samples))
    return log_lik - log_prior + log_q


@dispatch(object, Normal, Normal, [object])
def elbo(lik, p, q, num_samples=1):
    return B.mean(map_fn_columns(lik, q.sample(num_samples))) - p.kl(q)
