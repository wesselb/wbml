# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lab import B


def elbo(lik, p, q, num_samples=1):
    def lik_fun(z):
        return B.sum(lik.log_pdf(z[:, None]))

    samples = q.sample(num_samples)
    log_lik = B.sum(tf.map_fn(lik_fun, B.transpose(samples)))
    log_prior = B.sum(p.log_pdf(samples))
    log_q = B.sum(q.log_pdf(samples))
    return (log_lik - log_prior + log_q) / num_samples
