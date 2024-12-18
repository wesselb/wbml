import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .data import data_path, resource

__all__ = ["load"]


def load(detrend_method="gp"):
    _fetch()

    df = pd.read_csv(
        data_path("mauna_loa", "co2_mm_mlo.txt"),
        delim_whitespace=True,
        header=72,
    )
    df = df.iloc[:, [2, 3]]
    df.columns = ["year", "ppm"]
    df = df.set_index("year")

    # Remove missing data.
    df[df < 0] = np.nan
    df.dropna(inplace=True)

    # Also compute trend and detrended version.
    if detrend_method == "gp":
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
            random_state=0,
        )
    elif detrend_method == "linear":
        model = LinearRegression()
    else:
        raise ValueError('`detrend_method` must be "gp" or "linear".')
    index = np.array(df.index)[:, None]
    model.fit(index, df["ppm"])
    trend = model.predict(index)
    df["ppm_trend"] = trend
    df["ppm_detrended"] = df["ppm"] - trend

    return df


def _fetch():
    resource(
        target=data_path("mauna_loa", "co2_mm_mlo.txt"),
        url="ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt",
    )
