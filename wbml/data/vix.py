from datetime import datetime

import pandas as pd

from .data import data_path, resource

__all__ = ["load"]


def load():
    """VIX data.

    Source:
        The history starting at 2 Jan 1990 until the time of download is downloaded
        from the following link:
            https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv

    Returns:
        :class:`pd.DataFrame`: VIX data.
    """
    _fetch()

    df = pd.read_csv(data_path("vix", "vix.csv"))
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"close/last": "close"})
    df.date = list(map(lambda x: datetime.strptime(x, "%m/%d/%Y"), df.date))
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    return df


def _fetch():
    resource(
        target=data_path("vix", "vix.csv"),
        url="https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    )
