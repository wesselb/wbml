from functools import wraps

import lab as B
import numpy as np
import pandas as pd
from plum import Dispatcher, Union

__all__ = ["mse", "smse", "rmse", "srmse", "mae", "mll", "smll", "r2"]

_dispatch = Dispatcher()

_Pandas = Union[pd.DataFrame, pd.Series]
_PandasOrScalar = Union[_Pandas, B.Number]


def _auto_convert(f):
    @wraps(f)
    def wrapped_f(*args):
        converted_args = ()
        for i, arg in enumerate(args):
            # Check if the argument is already a Pandas object.
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                converted_args += (arg,)
                continue

            # It is not already a Pandas object. Attempt to convert.
            if B.rank(arg) == 0:
                converted_args += (arg,)
            elif B.rank(arg) == 1:
                converted_args += (pd.Series(arg),)
            elif B.rank(arg) == 2:
                converted_args += (pd.DataFrame(arg),)
            else:
                raise ValueError(
                    f"Argument {i} has rank {B.rank(arg)}, which "
                    f"cannot be automatically converted to a "
                    f"Pandas object."
                )

        return f(*converted_args)

    return wrapped_f


@_auto_convert
@_dispatch
def mse(mean: _PandasOrScalar, data: _Pandas):
    """Mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean squared error.
    """
    return ((mean - data) ** 2).mean()


@_auto_convert
@_dispatch
def smse(mean: _PandasOrScalar, data: _Pandas):
    """Standardised mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Standardised mean squared error.
    """
    return mse(mean, data) / mse(data.mean(), data)


@_auto_convert
@_dispatch
def rmse(mean: _PandasOrScalar, data: _Pandas):
    """Root mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Root mean squared error.
    """
    return mse(mean, data) ** 0.5


@_auto_convert
@_dispatch
def srmse(mean: _PandasOrScalar, data: _Pandas):
    """Standardised root mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Standardised root mean squared error.
    """
    return rmse(mean, data) / rmse(data.mean(), data)


@_auto_convert
@_dispatch
def mae(mean: _PandasOrScalar, data: _Pandas):
    """Mean absolute error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean absolute error.
    """
    return np.abs(mean - data).mean()


@_auto_convert
@_dispatch
def mll(mean: _PandasOrScalar, variance: _PandasOrScalar, data: _Pandas):
    """Mean log loss.

    Args:
        mean (tensor): Mean of prediction.
        variance (tensor): Variance of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean log loss.
    """
    return (
        0.5 * np.log(2 * np.pi * variance) + 0.5 * (mean - data) ** 2 / variance
    ).mean()


@_auto_convert
@_dispatch
def smll(mean: _PandasOrScalar, variance: _PandasOrScalar, data: _Pandas):
    """Standardised mean log loss.

    Args:
        mean (tensor): Mean of prediction.
        variance (tensor): Variance of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Standardised mean log loss.
    """
    return mll(mean, variance, data) - mll(data.mean(), data.var(ddof=0), data)


@_auto_convert
@_dispatch
def r2(mean: _PandasOrScalar, data: _Pandas):
    """R-squared.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: R-squared.
    """
    return 1 - ((data - mean) ** 2).mean() / data.var(ddof=0)