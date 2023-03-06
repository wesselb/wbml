import datetime

import pandas as pd

from .data import data_path, dependency, resource

__all__ = ["load"]

start = datetime.datetime(2019, 1, 1)
day = datetime.timedelta(days=1)
hour = datetime.timedelta(hours=1)


def load():
    _fetch()

    dfs = []

    for i in range(365):
        date = start + i * day

        # Load CSV.
        date_str = date.strftime("%Y%m%d")
        df = pd.read_csv(data_path("miso", f"{date_str}_rt_lmp_final.csv"), header=3)
        # Select LMPs.
        df = df[df["Value"] == "LMP"].set_index("Node")

        # Fix column names and transpose.
        df = df[[f"HE {j + 1}" for j in range(24)]]
        df.columns = [date + j * hour for j in range(24)]
        df = df.T

        dfs.append(df)

    return pd.concat(dfs, sort=True)


def _fetch():
    current = start

    for i in range(1, 12 + 1):
        # Get all the files for that month.
        month = datetime.datetime(start.year, i, 1)
        date_str_month = month.strftime("%Y%m")
        resource(
            target=data_path("miso", f"{date_str_month}_rt_lmp_final_csv.zip"),
            url=f"https://docs.misoenergy.org/marketreports/"
            f"{date_str_month}_rt_lmp_final_csv.zip",
        )

        # Unzip the downloaded files until we hit the next month.
        while current.month == i:
            date_str_current = current.strftime("%Y%m%d")
            dependency(
                target=data_path("miso", f"{date_str_current}_rt_lmp_final.csv"),
                source=data_path("miso", f"{date_str_month}_rt_lmp_final_csv.zip"),
                commands=[f"unzip -n {date_str_month}_rt_lmp_final_csv.zip"],
            )
            current += day
