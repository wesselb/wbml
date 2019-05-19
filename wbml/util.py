# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

__all__ = ['inv_perm']


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
