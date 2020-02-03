import datetime

import pandas as pd

from .data import data_path

__all__ = ['load']


def load():
    start = datetime.datetime(2019, 1, 1)
    day = datetime.timedelta(days=1)
    hour = datetime.timedelta(hours=1)

    dfs = []

    for i in range(365):
        date = start + i * day

        # Load CSV.
        date_str = date.strftime('%Y%m%d')
        df = pd.read_csv(data_path('miso', f'{date_str}_rt_lmp_final.csv'),
                         header=3)
        # Select LMPs.
        df = df[df['Value'] == 'LMP'].set_index('Node')

        # Fix column names and transpose.
        df = df[[f'HE {j + 1}' for j in range(24)]]
        df.columns = [date + j * hour for j in range(24)]
        df = df.T

        dfs.append(df)

    return pd.concat(dfs)
