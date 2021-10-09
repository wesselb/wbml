import pandas as pd

from .data import data_path, resource, date_to_decimal_year

__all__ = ["load"]


def load():
    _fetch()

    df = pd.read_csv(data_path("crude_oil", "crude_oil.csv"))
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"close/last": "close"})
    df.date = list(map(lambda x: date_to_decimal_year(x, "%m/%d/%Y"), df.date))
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    return df


def _fetch():
    resource(
        target=data_path("crude_oil", "crude_oil.csv"),
        url="https://www.dropbox.com/s/6ah5l0b64w9n78l/crude_oil.csv?dl=1",
    )
