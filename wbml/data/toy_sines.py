# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from .data import Data

__all__ = ['load']


def load():
    x = np.linspace(0, 6 * np.pi, 100)
    y = np.stack((np.sin(x),
                  0.1 * x,
                  2 * np.sin(x),
                  0.1 * x + 0.2 * np.sin(x)),
                 axis=1)
    y = y + 0.2 * np.random.randn(*y.shape)

    # Construct data and reorder outputs.
    d_all = Data(x, y, ['sin(x)', 'x^2', 'sin(x)^2', 'x + sin(x)'])

    # Select data from the paper.
    d_test, d_train = d_all.interval_y(['sin(x)^2', 'x + sin(x)'],
                                       4 * np.pi, 6 * np.pi)
    return d_all, d_train, [d_test]
