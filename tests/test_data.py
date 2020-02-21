import importlib

import pytest


@pytest.mark.parametrize('name',
                         ['air_temp',
                          'eeg',
                          'exchange',
                          'mauna_loa',
                          'miso',
                          'toy_sines'])
def test_import(name):
    module = importlib.import_module(f'wbml.data.{name}')
    module.load()


@pytest.mark.parametrize('name',
                         ['cmip5',
                          'stratis',
                          'station',
                          'snp'])
@pytest.mark.xfail()
def test_import_unable_to_fetch(name):
    module = importlib.import_module(f'wbml.data.{name}')
    module.load()
