import pandas as pd

from .data import data_path, split_df, date_to_decimal_year

__all__ = ['load']


def load():
    df = pd.read_csv(data_path('exchange', 'exchange.csv'))

    # Extract and convert to rates.
    rates = 1 / df.iloc[:, 3:]

    # Set index to years.
    years = [date_to_decimal_year(x, '%Y/%m/%d') for x in df['YYYY/MM/DD']]
    rates['year'] = years
    rates = rates.set_index('year')

    # Reorder according to paper.
    last_names = ['XPT/USD', 'CAD/USD', 'JPY/USD', 'AUD/USD']
    names = sorted(set(rates.columns) - set(last_names)) + last_names
    rates = rates[names]

    # Split into train and test according to paper.
    test1, train = split_df(rates, (49, 100), ['CAD/USD'], iloc=True)
    test2, train = split_df(train, (49, 150), ['JPY/USD'], iloc=True)
    test3, train = split_df(train, (49, 200), ['AUD/USD'], iloc=True)
    test = pd.concat([test1, test2, test3], axis=1)

    return rates, train, test