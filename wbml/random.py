# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B


class Bernoulli:
    def __init__(self, theta=.5):
        self._theta = theta

    def sample(self, shape=()):
        return B.cast(B.rand(shape) > self._theta)
