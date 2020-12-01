import importlib
import os.path
import shutil

import matplotlib.pyplot as plt
import pytest

from wbml.data import data_path, DependencyError


class _SetDataAside:
    def __init__(self, name):
        self.path = data_path(name)
        self.moved = False

    def __enter__(self):
        if os.path.exists(self.path):
            # Set data aside.
            os.rename(self.path, self.path + "_set_aside")
            self.moved = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.moved:
            # Remove any data created.
            shutil.rmtree(self.path, ignore_errors=True)
            # Put original data back into place.
            os.rename(self.path + "_set_aside", self.path)


def _import_and_execute(name, monkeypatch):
    # Kill showing any plots.
    monkeypatch.setattr(plt, "show", lambda: None)

    module = importlib.import_module(f"wbml.data.{name}")
    module.load()

    # Test any other possible methods.
    for method in ["stats_and_vis"]:
        if hasattr(module, method):
            getattr(module, method)()


@pytest.mark.parametrize(
    "name", ["air_temp", "eeg", "exchange", "jura", "mauna_loa", "miso", "toy_sines"]
)
def test_import(name, monkeypatch):
    with _SetDataAside(name):
        _import_and_execute(name, monkeypatch)


@pytest.mark.parametrize(
    "name", ["cmip5", "stratis", "station", "snp"]  # Only test `stratis` here.
)
@pytest.mark.xfail()
def test_import_unable_to_fetch(name, monkeypatch):
    _import_and_execute(name, monkeypatch)


@pytest.mark.parametrize("name", ["cmip5", "station", "snp"])
def test_import_fail_unable_to_fetch(name, monkeypatch):
    with _SetDataAside(name):
        with pytest.raises(DependencyError):
            _import_and_execute(name, monkeypatch)
