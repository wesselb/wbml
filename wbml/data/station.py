import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas

from .data import data_path, asserted_dependency
from .. import out
from ..plot import tweak

__all__ = ["load", "stats_and_vis"]


def load():
    _fetch()

    data = pandas.read_csv(data_path("station", "global_station_data.csv"))
    data.rename(columns={"value": "temp"}, inplace=True)

    # Extract locations per year and month.
    per_month = []
    data.groupby(["year", "month"]).apply(lambda x: per_month.append(x.copy()))

    # Extract dates.
    per_month_dates = [
        datetime.date(year=x["year"].iloc[0], month=x["month"].iloc[0], day=1)
        for x in per_month
    ]

    return data, per_month_dates, per_month


def _fetch():
    asserted_dependency(target=data_path("station", "global_station_data.csv"))


def stats_and_vis():
    data, per_month_dates, per_month = load()

    # Print some stats.
    num_measurements = np.array(list(map(len, per_month)))
    out.kv("Total number of dates", len(per_month))
    out.kv(
        "Number of dates with 20 measurements or more", np.sum(num_measurements >= 20)
    )
    out.kv("Average number of measurements", int(np.mean(num_measurements)))

    # Show number of measurements.
    plt.figure()
    plt.title("Number Of Measurements")
    plt.scatter(per_month_dates, num_measurements, s=2)
    plt.xlabel("Date")
    plt.ylabel("Number of measurements")
    tweak(legend=False)

    # Show 16 randomly chosen snapshots.
    plt.figure(figsize=(12, 8))
    vmin = data["temp"].min()
    vmax = data["temp"].max()
    lon_range = data["lon"].min(), data["lon"].max()
    lat_range = data["lat"].min(), data["lat"].max()
    inds = sorted(np.random.permutation(len(per_month))[:16])
    for i, ind in enumerate(inds):
        plt.subplot(4, 4, i + 1).set_title(per_month_dates[ind])
        plt.scatter(
            per_month[ind]["lon"],
            per_month[ind]["lat"],
            s=4,
            c=per_month[ind]["temp"],
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()
        plt.xlim(*lon_range)
        plt.ylim(*lat_range)
        tweak(legend=False)

    plt.show()
