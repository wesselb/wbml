import datetime
import os

import numpy as np
import pandas as pd

from .data import data_path, split_df, resource

__all__ = ["load"]

cache = data_path("air_temp", "data.parquet")


def _to_float(x):
    try:
        return float(x)
    except ValueError as e:
        if x in {"", "n/a", " "}:
            return np.nan
        elif x.startswith(".-"):
            # Unclear what this means. Just treat it as missing.
            return np.nan
        else:
            raise e


def load(split=None):
    """Load air temperature data.

    Args:
        split (str, optional): Name of split. Must be `None` or `"requeima19"`.

    Returns:
        object: Loaded data. Depends on the value for `split`.
    """
    days = _fetch()

    # Cache the data. Loading the individual files takes a while.
    if not os.path.exists(cache):
        sites = ["Bra", "Cam", "Chi", "Sot"]
        features = [("DEPTH", "height"), ("WSPD", "speed"), ("ATMP", "temp")]

        # Read specified days.
        dfs = []
        for site in sites:
            for day in days:
                month_short = day.strftime("%b")
                try:
                    data = pd.read_csv(
                        data_path(
                            "air_temp",
                            f"{site:s}{day.day:02d}{month_short}{day.year}.csv",
                        )
                    )
                except:
                    # The site might have returned a 404, which is saved as the CSV. Just
                    # ignore these entires.
                    continue

                # Convert times and dates.
                index = []
                for i, row in data.iterrows():
                    day, month, year = map(int, row["Date"].split("/"))
                    hour, minute = map(int, row["Time"].split(":"))
                    dt = datetime.datetime(year, month, day, hour, minute)
                    index.append((site, dt))

                # Construct desired data frame.
                index = pd.MultiIndex.from_tuples(index, names=["site", "date"])
                data = pd.DataFrame(
                    {
                        name_to: list(map(_to_float, data[name_from]))
                        for name_from, name_to in features
                    },
                    index=index,
                )
                dfs.append(data)

        # Concat all and fix order.
        df = pd.concat(dfs)
        # Remove duplications; keep the first.
        df = df[~df.index.duplicated(keep="first")]
        df = df.unstack("site").reindex(
            [
                ("height", "Bra"),
                ("height", "Sot"),
                ("height", "Cam"),
                ("height", "Chi"),
                ("speed", "Bra"),
                ("speed", "Sot"),
                ("speed", "Cam"),
                ("speed", "Chi"),
                ("temp", "Bra"),
                ("temp", "Sot"),
                ("temp", "Cam"),
                ("temp", "Chi"),
            ],
            axis=1,
        )

        # Write cache.
        df.to_parquet(cache)
    else:
        # Cache already exists. Load cache.
        df = pd.read_parquet(cache)

    # Clear invalid sensor readings.
    df[df < -1000] = np.nan

    # Clear an obviously invalid period of readings.
    mask = (
        (datetime.datetime(2014, 1, 1) <= df.index)
        & (df.index < datetime.datetime(2014, 9, 1))
        & (df.temp.Sot == 0)
    )
    df.loc[mask, ("temp", "Sot")] = np.nan

    # TODO: Clear more invalid sensor readings.

    if split is None:
        return df
    elif split == "requeima19":

        def make_range(lower, upper):
            day = datetime.timedelta(days=1)
            date_start = datetime.datetime(year=2013, month=7, day=1)
            return date_start + (lower - 1) * day, date_start + (upper - 1) * day

        # Construct data sets of various sizes.
        ds = []
        for lower, upper in [
            make_range(10, 20),
            make_range(8, 23),
            (
                datetime.datetime(year=2013, month=7, day=1),
                datetime.datetime(year=2013, month=8, day=1),
            ),
        ]:
            data = df[(df.index >= lower) & (df.index < upper)]

            test1, train = split_df(data, make_range(12, 16), [("temp", "Cam")])
            test2, train = split_df(train, make_range(14, 18), [("temp", "Chi")])

            test1_ext, _ = split_df(data, make_range(11.5, 16.5), [("temp", "Cam")])
            test2_ext, _ = split_df(train, make_range(13.5, 18.5), [("temp", "Chi")])

            ds.append((data, train, [test1, test2, test1_ext, test2_ext]))
        return ds
    else:  # pragma: no cover
        raise ValueError(f'Bad split "{split}".')


def _fetch():
    site_urls = [
        ("Cam", "http://www.cambermet.co.uk"),
        ("Chi", "http://www.chimet.co.uk"),
        ("Bra", "http://www.bramblemet.co.uk"),
        ("Sot", "http://www.sotonmet.co.uk"),
    ]
    days = []
    day = datetime.datetime(year=2013, month=1, day=1)
    while day < datetime.datetime(year=2016, month=1, day=1):
        days.append(day)
        day += datetime.timedelta(days=1)
    for day in days:
        month_full = day.strftime("%B")
        month_short = day.strftime("%b")
        for site, url in site_urls:
            resource(
                target=data_path(
                    "air_temp",
                    f"{site}{day.day:02d}{month_short}{day.year}.csv",
                ),
                url=(
                    f"{url}/archive/{day.year}/{month_full}/CSV/"
                    f"{site}{day.day:02d}{month_short}{day.year}.csv"
                ),
            )

    return days
