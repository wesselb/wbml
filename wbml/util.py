import inspect
import warnings
from functools import reduce
from operator import mul

import lab as B

__all__ = ["warn_upmodule", "inv_perm", "normal1d_logpdf", "BatchVars"]


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


def inv_perm(perm):
    """Invert a permutation.

    Args:
        perm (list): Permutation to invert.

    Returns:
        list: Inverse permutation.
    """
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return out


def normal1d_logpdf(x, var, mean=0):
    """Broadcast the one-dimensional normal logpdf.

    Args:
        x (tensor): Point to evaluate at.
        var (tensor): Variances.
        mean (tensor): Means.

    Returns:
        tensor: Logpdf.
    """
    return -(B.log_2_pi + B.log(var) + (x - mean) ** 2 / var) / 2


class BatchVars:
    """Extract variables from a source with a batch axis as the first axis.

    Args:
        source (tensor): Source.
    """

    def __init__(self, source):
        self.source = source
        self.index = 0

    def get(self, shape):
        """Get a batch of tensor of a particular shape.

        Args:
            shape (shape): Shape of tensor.

        Returns:
            tensor: Batch of tensors of shape `shape`.
        """
        length = reduce(mul, shape, 1)
        res = self.source[:, self.index : self.index + length]
        self.index += length
        return B.reshape(res, -1, *shape)
