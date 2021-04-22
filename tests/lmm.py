import numpy as np
import pytest
from lab import B
from stheno import EQ

from wbml.lmm import _to_tuples, OLMM, LMMPP
from .util import approx


def test_to_tuples():
    x = np.linspace(0, 1, 3)[:, None]
    y = np.random.randn(3, 2)
    y[0, 0] = np.nan
    y[1, 1] = np.nan

    # Check correctness.
    (x1, i1, y1), (x2, i2, y2) = _to_tuples(x, y)
    approx(x1, x[[1, 2]])
    assert i1 == 0
    approx(y1, y[[1, 2], 0])
    approx(x2, x[[0, 2]])
    assert i2 == 1
    approx(y2, y[[0, 2], 1])

    # Test check that any data is extracted.
    y_nan = y.copy()
    y_nan[:] = np.nan
    with pytest.raises(ValueError):
        _to_tuples(x, y_nan)


def test_lmm_missing_data():
    # Setup model.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = 0.1
    noises_latent = np.array([0.1, 0.2])
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
    assert not np.isnan(lmm.logpdf(x, y2))

    # Throw away an entire time point and check correctness.
    y2 = y.copy()
    y2[1, :] = np.nan
    approx(lmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), lmm.logpdf(x, y2))

    # Check LML after conditioning.
    lmm = lmm.condition(x, y2)
    approx(lmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), lmm.logpdf(x, y2))


def test_compare_lmm_olmm():
    # Setup models.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = 0.1
    noises_latent = np.array([0.1, 0.2])
    U, S, _ = B.svd(B.randn(3, 2))
    H = np.dot(U, np.diag(S) ** 0.5)

    # Construct models.
    lmm = LMMPP(kernels, noise_obs, noises_latent, H)
    olmm = OLMM(kernels, noise_obs, noises_latent, H)

    # Construct data.
    x = np.linspace(0, 3, 5)
    y = lmm.sample(x, latent=False)
    x2 = np.linspace(4, 7, 5)
    y2 = lmm.sample(x2, latent=False)

    # Check LML before conditioning.
    approx(lmm.logpdf(x, y), olmm.logpdf(x, y))
    approx(lmm.logpdf(x2, y2), olmm.logpdf(x2, y2))

    # Check LML after conditioning.
    lmm = lmm.condition(x, y)
    olmm = olmm.condition(x, y)
    # Note: `lmm_pp.lml(x, y)` will not equal `olmm.lml(x, y)` due to
    # assumptions in the OLMM, so the follow is not tested.
    # allclose(lmm.logpdf(x, y), olmm.logpdf(x, y))
    approx(lmm.logpdf(x2, y2), olmm.logpdf(x2, y2))

    # Predict.
    preds_pp, means_pp, vars_pp = lmm.marginals(x2)
    preds, means, vars = olmm.marginals(x2)

    # Check predictions per time point.
    for i in range(5):
        approx(means_pp[i], means[i])
        approx(vars_pp[i], vars[i])

    # Check predictions per output.
    for i in range(3):
        approx(preds_pp[i][0], preds[i][0])
        approx(preds_pp[i][1], preds[i][1])
        approx(preds_pp[i][2], preds[i][2])


class TrackedIterator:
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
    noise_obs = 0.1
    noises_latent = np.array([0.1, 0.2])
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
    x = B.randn(10)
    assert isinstance(lmm.sample(x, latent=True), B.NPNumeric)
    assert lmm.fs.used, "lmm.fs was not used."
    assert not lmm.ys.used, "lmm.ys was used."

    assert isinstance(olmm.sample(x, latent=True), B.NPNumeric)
    assert olmm.xs.used, "olmm.xs was not used."
    assert not olmm.xs_noisy.used, "olmm.xs_noisy was used."

    # Test observed samples.
    TrackedIterator.reset()

    assert isinstance(lmm.sample(x, latent=False), B.NPNumeric)
    assert not lmm.fs.used, "lmm.fs was used."
    assert lmm.ys.used, "lmm.ys was not used"

    assert isinstance(olmm.sample(x, latent=False), B.NPNumeric)
    assert not olmm.xs.used, "olmm.xs was used"
    assert olmm.xs_noisy.used, "olmm.xs_noisy was not used."
