import os
import datetime
import numpy as np

import pandas as pd

__all__ = ['split_df',
           'data_path',
           'date_to_decimal_year']


def data_path(*xs):
    """Get the path of a data file.

    Args:
        *xs (str): Parts of the path.

    Returns:
        str: Absolute path.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir,
                                        os.pardir,
                                        'data',
                                        *xs))


def split_df(df, index_range, columns, iloc=False):
    """Split a data frame by selecting from columns a particular range.

    Args:
        df (:class:`pd.DataFrame`): Data frame to split.
        index_range (tuple): Tuple containing lower and upper limit of the
            range to split the index by. If `index_range = (a, b)`, then
            `[a, b)` is taken.
        columns (list[object]): Columns to select.
        iloc (bool, optional): The index range is the integer location instead
            of the index value. Defaults to `False`.

    Returns:
        tuple[:class:`pd.DataFrame`]: Selected rows from selected columns
            and the remainder.
    """
    if iloc:
        inds = np.arange(df.shape[0])
        rows = (inds >= index_range[0]) & (inds < index_range[1])
    else:
        rows = (df.index >= index_range[0]) & (df.index < index_range[1])
    selected = pd.DataFrame([df[name][rows] for name in columns]).T
    remainder = pd.DataFrame([df[name][~rows] for name in columns] +
                             [df[name] for name in
                              set(df.columns) - set(columns)]).T

    # Fix order of columns.
    selected_cols = [c for c in df.columns if c in columns]
    selected = selected.reindex(selected_cols, axis=1)
    remainder = remainder.reindex(df.columns, axis=1)

    return selected, remainder


def date_to_decimal_year(date, format):
    """Convert a date to decimal year.

    Args:
        date (str): Date as a string.
        format (str): Format of the dat.

    Returns:
        float: Decimal year corresponding to the date.
    """
    date = datetime.datetime.strptime(date, format)
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length
