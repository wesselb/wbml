import pandas as pd

from .data import data_path, resource, date_to_decimal_year

__all__ = ["load"]


def load():
    _fetch()

    data = pd.read_csv(data_path("crude_oil", "crude_oil.csv"))
    data.columns = [c.lower() for c in data.columns]
    data.rename(columns={"close/last": "close"})
    data.date = list(map(lambda x: date_to_decimal_year(x, "%m/%d/%Y"), data.date))

    return data


def _fetch():
    resource(
        target=data_path("crude_oil", "crude_oil.csv"),
        url="https://www.dropbox.com/s/6ah5l0b64w9n78l/crude_oil.csv?dl=1",
    )
