# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from wbml import vars32
from lab import B

__all__ = ['LogisticRegression']


class LogisticRegression(object):
    """Basic logistic regression.

    Args:
        data (:class:`.data.Data`): Training data.
        vars (:class:`.util.Vars`): Variable manager.
    """

    def __init__(self, data, vars=vars32):
        self.w = vars.get(shape=[data.m, 1])
        self.b = vars.get()
        self.data = data

    def predict(self, x):
        """Predict.

        Args:
            x (tensor): Points to predict at.

        Returns:
            tensor: Predictions.
        """
        return B.sigmoid(B.matmul(x, self.w) + self.b)

    def loss(self):
        """Construct the loss function.

        Returns:
            tensor: Loss function.
        """
        ps = self.predict(self.data.x)
        return -B.sum(self.data.y * B.log(ps) +
                      (1 - self.data.y) * B.log(1 - ps))
