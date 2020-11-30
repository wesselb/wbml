import datetime

import numpy as np
import pandas as pd

from .data import data_path, split_df, resource

__all__ = ["load"]


def _to_float(x):
    try:
        return float(x)
    except ValueError as e:
        if x in {"", "n/a", " "}:
            return np.nan
        else:
            raise e


def load():
    _fetch()

    sites = ["Bra", "Cam", "Chi", "Sot"]
    features = [("DEPTH", "height"), ("WSPD", "speed"), ("ATMP", "temp")]

    days = list(range(1, 1 + 31))
    fn = "{site:s}{day:02d}Jul2013.csv"

    dfs = []
    for site in sites:
        for day in days:
            df = pd.read_csv(data_path("air_temp", fn.format(site=site, day=day)))

            # Convert times and dates.
            index = []
            for i, row in df.iterrows():
                day, month, year = map(int, row["Date"].split("/"))
                hour, minute = map(int, row["Time"].split(":"))
                dt = datetime.datetime(year, month, day, hour, minute)
                index.append((site, dt))

            # Construct desired data frame.
            index = pd.MultiIndex.from_tuples(index, names=["site", "date"])
            df = pd.DataFrame(
                {
                    name_to: list(map(_to_float, df[name_from]))
                    for name_from, name_to in features
                },
                index=index,
            )
            dfs.append(df)

    # Concat and fix order to match the paper.
    df_all = (
        pd.concat(dfs)
        .unstack("site")
        .reindex(
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
    )

    # Clear invalid sensor readings.
    df_all[df_all < -1000] = np.nan

    def make_range(lower, upper):
        day = datetime.timedelta(days=1)
        date_start = df_all.index.min()
        return date_start + (lower - 1) * day, date_start + (upper - 1) * day

    # Construct data sets of various sizes.
    ds = []
    for lower, upper in [
        make_range(10, 20),
        make_range(8, 23),
        (df_all.index.min(), df_all.index.max()),
    ]:
        df = df_all[(df_all.index >= lower) & (df_all.index <= upper)]

        test1, train = split_df(df, make_range(12, 16), [("temp", "Cam")])
        test2, train = split_df(train, make_range(14, 18), [("temp", "Chi")])

        test1_ext, _ = split_df(df, make_range(11.5, 16.5), [("temp", "Cam")])
        test2_ext, _ = split_df(train, make_range(13.5, 18.5), [("temp", "Chi")])

        ds.append((df, train, [test1, test2, test1_ext, test2_ext]))

    return ds


def _fetch():
    site_urls = [
        ("Cam", "http://www.cambermet.co.uk/archive/2013/July/CSV/"),
        ("Chi", "http://www.chimet.co.uk/archive/2013/July/CSV/"),
        ("Bra", "http://www.bramblemet.co.uk/archive/2013/July/CSV/"),
        ("Sot", "http://www.sotonmet.co.uk/archive/2013/July/CSV/"),
    ]
    for day in range(1, 32):
        for site, url in site_urls:
            resource(
                target=data_path("air_temp", f"{site}{day:02d}Jul2013.csv"),
                url=f"{url}{site}{day:02d}Jul2013.csv",
            )
