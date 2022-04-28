import pandas as pd

from .data import data_path, resource

__all__ = ["load"]


def load():
    """Predator–prey data.

    Source:
        The data is a digitised version of the historical data by D. R. Hundley:
            http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt

    Returns:
        :class:`pd.DataFrame`: Predator–prey data.
    """
    _fetch()

    df = pd.read_csv(
        data_path("pred_prey", "LynxHare.txt"),
        header=None,
        delim_whitespace=True,
        index_col=0,
    )
    df.index.name = "year"
    df.columns = ["hare", "lynx"]

    return df


def _fetch():
    resource(
        target=data_path("pred_prey", "LynxHare.txt"),
        url="http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt",
    )
