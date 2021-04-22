import lab as B
from numpy.testing import assert_allclose
from plum import Dispatcher

__all__ = ["approx"]

_dispatch = Dispatcher()


def approx(x, y, rtol=1e-7, atol=0):
    return assert_allclose(B.to_numpy(x), B.to_numpy(y), rtol=rtol, atol=atol)
