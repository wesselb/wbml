# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab.tf import B
from numpy.testing import assert_allclose
from stheno import EQ

from wbml import OLMM, LMMPPP
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, eprint


def test_olmm():
    B.backend_to_np()

    # Setup models.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = .1
    noises_latent = [.1, .2]
    U, S, _ = B.svd(np.random.randn(3, 2))
    H = U @ np.diag(S) ** .5

    # Construct models.
    lmm_ppp = LMMPPP(kernels, noise_obs, noises_latent, H)
    olmm = OLMM(kernels, noise_obs, noises_latent, H)

    # Construct data.
    x = np.linspace(0, 3, 5)
    y = lmm_ppp.sample(x)
    x2 = np.linspace(4, 7, 5)
    y2 = lmm_ppp.sample(x2)

    # Check LML.
    yield assert_allclose, lmm_ppp.lml(x, y), olmm.lml(x, y)

    # Check LML after conditioning.
    lmm_ppp.observe(x, y)
    olmm.observe(x, y)
    yield assert_allclose, lmm_ppp.lml(x2, y2), olmm.lml(x2, y2)

    # Need to recondition now: computing LML reverts to prior.
    lmm_ppp.observe(x, y)
    olmm.observe(x, y)

    # Predict.
    preds_ppp, means_ppp, vars_ppp = lmm_ppp.predict(x2)
    preds, means, vars = olmm.predict(x2)

    # Check predictions per time point.
    for i in range(5):
        yield assert_allclose, means_ppp[i], means[i]
        yield assert_allclose, vars_ppp[i], vars[i]

    # Check predictions per output.
    for i in range(3):
        yield assert_allclose, preds_ppp[i][0], preds[i][0]
        yield assert_allclose, preds_ppp[i][1], preds[i][1]
        yield assert_allclose, preds_ppp[i][2], preds[i][2]

    B.backend_to_tf()
