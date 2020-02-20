# noinspection PyUnresolvedReferences
from .cmip5 import load as load_cmip5
import warnings

__all__ = ['load']


def load():
    warnings.warn('Importing "wbml.data.stratis" is deprecated. Please import '
                  '"wbml.data.cmip5" instead.',
                  category=DeprecationWarning,
                  stacklevel=2)
    return load_cmip5()
