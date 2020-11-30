import lab as B
import pandas as pd
import pytest

import wbml.metric

two_arg = ["mse", "smse", "rmse", "srmse", "mae", "r2"]
three_arg = ["mll", "smll"]

two_arg_standardised = [("smse", 1), ("srmse", 1)]
three_arg_standardised = [("smll", 0)]


@pytest.mark.parametrize("name", two_arg)
def test_two_arg(name):
    metric = getattr(wbml.metric, name)

    # Test scalar usage.
    assert isinstance(metric(1, B.randn(10)), B.Number)

    # Test series usage.
    assert isinstance(metric(B.randn(10), B.randn(10)), B.Number)

    # Test matrix usage.
    assert isinstance(metric(B.randn(10, 10), B.randn(10, 10)), pd.Series)

    # Check that higher-order tensors fail.
    with pytest.raises(ValueError):
        metric(B.randn(10, 10, 10), B.randn(10, 10, 10))


@pytest.mark.parametrize("name", three_arg)
def test_three_arg(name):
    metric = getattr(wbml.metric, name)

    # Test scalar usage.
    assert isinstance(metric(1, 1, B.randn(10)), B.Number)

    # Test series usage.
    assert isinstance(metric(B.randn(10), B.rand(10), B.randn(10)), B.Number)

    # Test matrix usage.
    assert isinstance(
        metric(B.randn(10, 10), B.rand(10, 10), B.randn(10, 10)), pd.Series
    )

    # Check that higher-order tensors fail.
    with pytest.raises(ValueError):
        metric(B.randn(10, 10, 10), B.rand(10, 10, 10), B.randn(10, 10, 10))


@pytest.mark.parametrize("name, neutral_value", two_arg_standardised)
def test_two_arg_standardised(name, neutral_value):
    metric = getattr(wbml.metric, name)
    x = B.randn(10)
    mean = x.mean() * B.ones(10)
    assert B.all(metric(mean, x) == neutral_value)


@pytest.mark.parametrize("name, neutral_value", three_arg_standardised)
def test_three_arg_standardised(name, neutral_value):
    metric = getattr(wbml.metric, name)
    x = B.randn(10)
    mean = x.mean() * B.ones(10)
    variance = x.var(ddof=0) * B.ones(10)
    assert B.all(metric(mean, variance, x) == neutral_value)
