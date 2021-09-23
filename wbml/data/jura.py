import pandas as pd

from .data import data_path, resource

__all__ = ["load"]


def load():
    _fetch()

    name_corr = {"Xloc": "x", "Yloc": "y", "Landuse": "land", "Rock": "rock"}

    # Load training data.
    columns = pd.read_csv(
        data_path("jura", "prediction.dat"), skiprows=2, header=None, nrows=11
    )
    columns = [name_corr[c] if c in name_corr else c for c in columns[0]]
    train = pd.read_csv(
        data_path("jura", "prediction.dat"), skiprows=13, header=None, sep=r"\s+"
    )
    train.columns = columns
    train.set_index(["x", "y"], inplace=True)

    # Load test data.
    test = pd.read_csv(data_path("jura", "validation.dat"), sep="\t")
    test.columns = [name_corr[c] if c in name_corr else c for c in test.columns]
    test.set_index(["x", "y"], inplace=True)

    # Setup according to experiment.
    train = pd.concat([train[["Ni", "Zn", "Cd"]], test[["Ni", "Zn"]]])
    test = test[["Cd"]]

    return train, test


def _fetch():
    resource(
        target=data_path("jura", "prediction.dat"),
        url="https://www.dropbox.com/s/n7jamwpsi9p4wvs/prediction.dat?dl=1"
    )
    resource(
        target=data_path("jura", "validation.dat"),
        url="https://www.dropbox.com/s/slvclnw8qr6gjv8/validation.dat?dl=1"
    )
