from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher
import lab as B

__all__ = ["allclose", "approx"]

_dispatch = Dispatcher()


def allclose(x, y, rtol=1e-7, atol=0):
    return assert_allclose(B.to_numpy(x), B.to_numpy(y), rtol=rtol, atol=atol)


def approx(x, y, digits=7):
    return assert_array_almost_equal(B.to_numpy(x), B.to_numpy(y), decimal=digits)
