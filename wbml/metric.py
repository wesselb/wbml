from functools import wraps

import lab as B
import numpy as np
import pandas as pd
from plum import Dispatcher, Union

__all__ = ['mse', 'smse',
           'rmse', 'srmse',
           'mae',
           'mll', 'smll',
           'r2']

_dispatch = Dispatcher()
_Pandas = Union(pd.DataFrame, pd.Series)
_PandasOrScalar = Union(_Pandas, B.Number)


def _auto_convert(f):
    @_dispatch([B.Numeric])
    @wraps(f)
    def _convert(*args):
        converted_args = ()
        for i, arg in enumerate(args):
            if B.rank(arg) == 0:
                converted_args += (arg,)
            elif B.rank(arg) == 1:
                converted_args += (pd.Series(arg),)
            elif B.rank(arg) == 2:
                converted_args += (pd.DataFrame(arg),)
            else:
                raise ValueError(f'Argument {i} has rank {B.rank(arg)}, which '
                                 f'cannot be automatically converted to a '
                                 f'Pandas object.')

        return f(*converted_args)

    return f


@_auto_convert
@_dispatch(_PandasOrScalar, _Pandas)
def mse(mean, data):
    """Mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean squared error.
    """
    return ((mean - data) ** 2).mean()


@_auto_convert
@_dispatch(_PandasOrScalar, _Pandas)
def smse(mean, data):
    """Standardised mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Standardised mean squared error.
    """
    return mse(mean, data) / mse(data.mean(), data)


@_auto_convert
@_dispatch(_PandasOrScalar, _Pandas)
def rmse(mean, data):
    """Root mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Root mean squared error.
    """
    return mse(mean, data) ** .5


@_auto_convert
@_dispatch(_PandasOrScalar, _Pandas)
def srmse(mean, data):
    """Standardised root mean squared error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Standardised root mean squared error.
    """
    return rmse(mean, data) / rmse(data.mean(), data)


@_auto_convert
@_dispatch(_PandasOrScalar, _Pandas)
def mae(mean, data):
    """Mean absolute error.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean absolute error.
    """
    return np.abs(mean - data).mean()


@_auto_convert
@_dispatch(_PandasOrScalar, _PandasOrScalar, _Pandas)
def mll(mean, variance, data):
    """Mean log loss.

    Args:
        mean (tensor): Mean of prediction.
        variance (tensor): Variance of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: Mean log loss.
    """
    return (0.5 * np.log(2 * np.pi) +
            0.5 * (mean - data) ** 2 / variance).mean()


@_auto_convert
@_dispatch(_PandasOrScalar, _PandasOrScalar, _Pandas)
def smll(mean, variance, data):
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
@_dispatch(_PandasOrScalar, _Pandas)
def r2(mean, data):
    """R-squared.

    Args:
        mean (tensor): Mean of prediction.
        data (tensor): Reference data.

    Returns:
        tensor: R-squared.
    """
    return 1 - ((data - mean) ** 2).mean() / data.var(ddof=0)
