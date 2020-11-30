import numpy as np
import pandas as pd

from .data import split_df

__all__ = ["load"]


def load():
    x = np.linspace(0, 6 * np.pi, 100)
    y = np.stack((np.sin(x), 0.1 * x, 2 * np.sin(x), 0.1 * x + 0.2 * np.sin(x)), axis=1)
    y = y + 0.2 * np.random.randn(*y.shape)

    # Construct data and reorder outputs.
    df = pd.DataFrame(
        y,
        index=pd.Index(x, name="time"),
        columns=["sin(x)", "x^2", "sin(x)^2", "x + sin(x)"],
    )

    # Select data from the paper.
    df_train, df_test = split_df(
        df, (4 * np.pi, 6 * np.pi + 1), ["sin(x)^2", "x + sin(x)"]
    )

    return df, df_train, df_test
