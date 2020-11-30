import pandas as pd

from .data import data_path, split_df, date_to_decimal_year, resource

__all__ = ["load"]


def load(extended=False, nguyen=False):
    """Load the data.

    Args:
        extended (bool, optional): Also return years 2000 to 2010. Defaults
            to `False`.
        nguyen (bool, optional): Return the split from Nguyen & Bonilla (2014).

    Returns:
        tuple[:class:`pd.DataFrame`]: Three-tuple or four-tuple containing
            all the data, training data, test data, and potentially years
            2000 to 2010.
    """
    _fetch()

    rates = _load_csv(data_path("exchange", "exchange2007.csv"))

    if nguyen:
        # Return the split from Nguyen & Bonilla (2014).
        inds_cad = (49, 100)
        inds_jpy = (99, 150)
        inds_aud = (149, 200)
    else:
        # Return the split from Requeima et al. (2019).
        inds_cad = (49, 100)
        inds_jpy = (49, 150)
        inds_aud = (49, 200)

    # Split into train and test according to paper.
    test1, train = split_df(rates, inds_cad, ["USD/CAD"], iloc=True)
    test2, train = split_df(train, inds_jpy, ["USD/JPY"], iloc=True)
    test3, train = split_df(train, inds_aud, ["USD/AUD"], iloc=True)
    test = pd.concat([test1, test2, test3], axis=1)

    if extended:
        # Also return the rates from years 2000 to 2010.
        return (
            rates,
            train,
            test,
            pd.concat(
                [
                    _load_csv(data_path("exchange", f"exchange{year}.csv"))
                    for year in range(1990, 2015 + 1)
                ],
                axis=0,
            ),
        )

    return rates, train, test


def _load_csv(path):
    df = pd.read_csv(path, header=1)

    # Remove tail.
    df = df.iloc[:-1, :]

    # Extract and invert rates.
    rates = 1 / df.iloc[:, 3:]
    rates.columns = [
        "{}/{}".format(*reversed(name.split("/"))) for name in rates.columns
    ]

    # Set index to years.
    years = [date_to_decimal_year(x, "%Y/%m/%d") for x in df["YYYY/MM/DD"]]
    rates["year"] = years
    rates = rates.set_index("year")

    # Reorder according to paper.
    last_names = ["USD/XPT", "USD/CAD", "USD/JPY", "USD/AUD"]
    names = sorted(set(rates.columns) - set(last_names)) + last_names
    rates = rates[names]

    return rates


def _fetch():
    for year in range(1990, 2015 + 1):
        resource(
            target=data_path("exchange", f"exchange{year}.csv"),
            url="http://fx.sauder.ubc.ca/cgi/fxdata",
            post=True,
            data={
                "b": "USD",
                "c": ["X12", "XC3"],
                "rd": "",
                "fd": "1",
                "fm": "1",
                "fy": str(year),
                "ld": "31",
                "lm": "12",
                "ly": str(year),
                "y": "daily",
                "q": "volume",
                "f": "csv",
                "o": "",
            },
        )
