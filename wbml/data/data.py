import os

import pandas as pd

__all__ = ['split_df', 'data_path']


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


def split_df(df, index_range, columns):
    """Split a data frame by selecting from columns a particular range.

    Args:
        df (:class:`pd.DataFrame`): Data frame to split.
        index_range (tuple): Tuple containing lower and upper limit of the
            range to split the index by. If `index_range = (a, b)`, then
            `[a, b)` is taken.
        columns (list[object]): Columns to select.

    Returns:
        tuple[:class:`pd.DataFrame`]: Selected rows from selected columns
            and the remainder.
    """
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
