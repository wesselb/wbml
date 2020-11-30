import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .data import data_path, resource

__all__ = ["load"]


def load():
    _fetch()

    df = pd.read_csv(
        data_path("mauna_loa", "co2_mm_mlo.txt"), delim_whitespace=True, header=72
    )
    df = df.iloc[:, [2, 3]]
    df.columns = ["year", "ppm"]
    df = df.set_index("year")

    # Remove missing data.
    df[df < 0] = np.nan
    df.dropna(inplace=True)

    # Also compute detrended version.
    lr = LinearRegression()
    lr.fit(df.index[:, None], df["ppm"])
    df["ppm_detrended"] = df["ppm"] - lr.predict(df.index[:, None])

    return df


def _fetch():
    resource(
        target=data_path("mauna_loa", "co2_mm_mlo.txt"),
        url="ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt",
    )
