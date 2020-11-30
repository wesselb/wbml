import warnings

import pytest
from wbml.warning import warn_upmodule


def test_warn_upmodule(monkeypatch):
    orig_warn = warnings.warn

    def mock_warn(*args, **kw_args):
        assert kw_args["stacklevel"] > 2
        orig_warn(*args, **kw_args)

    monkeypatch.setattr(warnings, "warn", mock_warn)

    with pytest.warns(UserWarning, match="Test warning"):
        warn_upmodule("Test warning", category=UserWarning)


def test_warning_formatting():
    pass
