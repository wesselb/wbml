import pandas as pd

from .data import data_path, date_to_decimal_year, asserted_dependency

__all__ = ["load"]


def load():
    _fetch()

    df = pd.read_csv(data_path("snp", "snp500_vol.csv"))
    df = df[["date", "vol"]]
    df.columns = ["date", "volume"]

    # Convert date to decimal years and make index.
    df["date"] = [date_to_decimal_year(date, "%Y-%m-%d") for date in df["date"]]
    df.set_index("date", inplace=True)

    return df


def _fetch():
    asserted_dependency(target=data_path("snp", "snp500_vol.csv"))
