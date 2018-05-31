# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lab import B
from stheno import Normal, Diagonal
from wbml import create_var


class QNormal:
    @staticmethod
    def random_init_pars(d=2):
        return [create_var(x) for x in [(d, 1), (d, d)]]

    def __init__(self, mu, chol):
        self.mu = mu
        self.chol = chol
        self._d = Normal(B.matmul(chol, chol, tr_b=True), mu)

    def sample(self, num_samples=1):
        return self._d.sample(num_samples)

    def log_pdf(self, samples):
        return B.sum(self._d.log_pdf(samples))


class QNormalDiag:
    @staticmethod
    def random_init_pars(d=2):
        return create_var([d, 1]), create_var([d])

    def __init__(self, mu, log_s2):
        self.mu = mu
        self.log_s2 = log_s2
        self._d = Normal(Diagonal(B.exp(log_s2)), mu)

    def sample(self, num_samples=1, noise=None):
        return self._d.sample(num_samples, noise)

    def log_pdf(self, samples):
        return B.sum(self._d.log_pdf(samples))


def elbo(lik, p, q, num_samples=1):
    def lik_fun(z):
        return B.sum(lik.log_pdf(z[:, None]))

    samples = q.sample(num_samples)
    log_lik = B.sum(tf.map_fn(lik_fun, B.transpose(samples)))
    log_prior = B.sum(p.log_pdf(samples))
    log_q = B.sum(q.log_pdf(samples))
    return (log_lik + log_prior - log_q) / num_samples
