# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B
from numpy.testing import assert_allclose
from stheno import EQ
from . import OLMM, LMMPP
from wbml.lmm import _to_tuples

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, allclose, isinstance


def test_to_tuples():
    x = np.linspace(0, 1, 3)[:, None]
    y = np.random.randn(3, 2)
    y[0, 0] = np.nan
    y[1, 1] = np.nan

    # Check correctness.
    (x1, i1, y1), (x2, i2, y2) = _to_tuples(x, y)
    yield allclose, x1, x[[1, 2]]
    yield eq, i1, 0
    yield allclose, y1, y[[1, 2], 0]
    yield allclose, x2, x[[0, 2]]
    yield eq, i2, 1
    yield allclose, y2, y[[0, 2], 1]

    # Test check that any data is extracted.
    y_nan = y.copy()
    y_nan[:] = np.nan
    yield raises, ValueError, lambda: _to_tuples(x, y_nan)


def test_lmm_missing_data():
    # Setup model.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = .1
    noises_latent = np.array([.1, .2])
    H = np.random.randn(3, 2)

    # Construct model.
    lmm = LMMPP(kernels, noise_obs, noises_latent, H)

    # Construct data.
    x = np.linspace(0, 3, 5)
    y = lmm.sample(x, latent=False)

    # Throw away random data points and check that the logpdf computes.
    y2 = y.copy()
    y2[0, 0] = np.nan
    y2[2, 2] = np.nan
    y2[4, 1] = np.nan
    yield ok, not np.isnan(lmm.logpdf(x, y2))

    # Throw away an entire time point and check correctness.
    y2 = y.copy()
    y2[1, :] = np.nan
    yield assert_allclose, \
          lmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), \
          lmm.logpdf(x, y2)

    # Check LML after conditioning.
    lmm.observe(x, y2)
    yield assert_allclose, \
          lmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), \
          lmm.logpdf(x, y2)


def test_compare_lmm_olmm():
    # Setup models.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = .1
    noises_latent = np.array([.1, .2])
    U, S, _ = B.svd(B.randn(3, 2))
    H = np.dot(U, np.diag(S) ** .5)

    # Construct models.
    lmm = LMMPP(kernels, noise_obs, noises_latent, H)
    olmm = OLMM(kernels, noise_obs, noises_latent, H)

    # Construct data.
    x = np.linspace(0, 3, 5)
    y = lmm.sample(x, latent=False)
    x2 = np.linspace(4, 7, 5)
    y2 = lmm.sample(x2, latent=False)

    # Check LML before conditioning.
    yield assert_allclose, lmm.logpdf(x, y), olmm.logpdf(x, y)
    yield assert_allclose, lmm.logpdf(x2, y2), olmm.logpdf(x2, y2)

    # Check LML after conditioning.
    lmm.observe(x, y)
    olmm.observe(x, y)
    # Note: `lmm_pp.lml(x, y)` will not equal `olmm.lml(x, y)` due to
    # assumptions in the OLMM, so the follow is not tested.
    # yield assert_allclose, lmm.logpdf(x, y), olmm.logpdf(x, y)
    yield assert_allclose, lmm.logpdf(x2, y2), olmm.logpdf(x2, y2)

    # Predict.
    preds_pp, means_pp, vars_pp = lmm.marginals(x2)
    preds, means, vars = olmm.marginals(x2)

    # Check predictions per time point.
    for i in range(5):
        yield assert_allclose, means_pp[i], means[i]
        yield assert_allclose, vars_pp[i], vars[i]

    # Check predictions per output.
    for i in range(3):
        yield assert_allclose, preds_pp[i][0], preds[i][0]
        yield assert_allclose, preds_pp[i][1], preds[i][1]
        yield assert_allclose, preds_pp[i][2], preds[i][2]


class TrackedIterator(object):
    """An iterator that keeps track of whether it has been used.

    Attributes:
        used (bool): Boolean indicating whether it has been usd.

    Args:
        wrap (object): Object to wrap.
    """
    instances = []

    def __init__(self, wrap):
        self.used = False
        self.wrap = wrap
        TrackedIterator.instances.append(self)

    def __iter__(self):
        self.used = True
        return iter(self.wrap)

    @staticmethod
    def reset():
        """Reset all."""
        for instance in TrackedIterator.instances:
            instance.used = False


def test_lmm_olmm_sample():
    # Setup models.
    kernels = [EQ()] * 2
    noise_obs = .1
    noises_latent = np.array([.1, .2])
    H = B.randn(3, 2)

    # Construct models.
    lmm = LMMPP(kernels, noise_obs, noises_latent, H)
    olmm = OLMM(kernels, noise_obs, noises_latent, H)

    # Wrap.
    lmm.fs = TrackedIterator(lmm.fs)
    lmm.ys = TrackedIterator(lmm.ys)
    olmm.xs = TrackedIterator(olmm.xs)
    olmm.xs_noisy = TrackedIterator(olmm.xs_noisy)

    # Test latent samples.

    x = np.linspace(0, 1, 3)

    yield isinstance, lmm.sample(x, latent=True), B.NPNumeric
    yield ok, lmm.fs.used, 'fs used'
    yield ok, not lmm.ys.used, 'ys not used'

    yield isinstance, olmm.sample(x, latent=True), B.NPNumeric
    yield ok, olmm.xs.used, 'xs used'
    yield ok, not olmm.xs_noisy.used, 'xs_noisy not used'

    # Test observed samples.

    TrackedIterator.reset()

    yield isinstance, lmm.sample(x, latent=False), B.NPNumeric
    yield ok, not lmm.fs.used, 'fs not used'
    yield ok, lmm.ys.used, 'ys used'

    yield isinstance, olmm.sample(x, latent=False), B.NPNumeric
    yield ok, not olmm.xs.used, 'xs not used'
    yield ok, olmm.xs_noisy.used, 'xs_noisy used'
