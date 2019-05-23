# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul
import lab as B

__all__ = ['inv_perm', 'normal1d_logpdf', 'BatchVars']


def inv_perm(perm):
    """Invert a permutation.

    Args:
        perm (list): Permutation to invert.

    Returns:
        list: Inverse permutation.
    """
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return out


def normal1d_logpdf(x, var, mean=0):
    """Broadcast the one-dimensional normal logpdf.

    Args:
        x (tensor): Point to evaluate at.
        var (tensor): Variances.
        mean (tensor): Means.

    Returns:
        tensor: Logpdf.
    """
    return -(B.log_2_pi + B.log(var) + (x - mean) ** 2 / var) / 2


class BatchVars(object):
    """Extract variables from a source with a batch axis as the first axis.

    Args:
        source (tensor): Source.
    """

    def __init__(self, source):
        self.source = source
        self.index = 0

    def get(self, shape):
        """Get a batch of tensor of a particular shape.

        Args:
            shape (shape): Shape of tensor.

        Returns:
            tensor: Batch of tensors of shape `shape`.
        """
        length = reduce(mul, shape, 1)
        res = self.source[:, self.index:self.index + length]
        self.index += length
        return B.reshape(res, -1, *shape)
