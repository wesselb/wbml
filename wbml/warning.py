import inspect
import warnings

__all__ = ["warn_upmodule"]

# By default show deprecation warnings.
warnings.filterwarnings("always", category=DeprecationWarning)


def _get_module_name(frame_info):
    return frame_info.frame.f_globals["__name__"].split(".")[0]


def warn_upmodule(*args, **kw_args):
    """Call :func:`warnings.warn`, but set the keyword argument `stacklevel`
    corresponding to the first other module above the current module.
    """
    level = 2
    stack = inspect.stack()[1:]
    this_module = _get_module_name(stack[0])

    # Up the level until the first other module is reached.
    for frame_info in stack:
        if _get_module_name(frame_info) != this_module:
            break
        else:
            level += 1

    warnings.warn(*args, stacklevel=level, **kw_args)
