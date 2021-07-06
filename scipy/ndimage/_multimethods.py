import functools
import numpy as np
from scipy._lib.uarray import Dispatchable, all_of_type, create_multimethod
from scipy.ndimage import _api


mark_non_coercible = lambda x: Dispatchable(x, np.ndarray, coercible=False)
create_ndimage = functools.partial(create_multimethod,
                                   domain="numpy.scipy.ndimage")


def _get_docs(func):
    """
    Decorator to take the docstring from original
    function and assign to the multimethod.
    """
    func.__doc__ = getattr(_api, func.__name__).__doc__

    @functools.wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)
    return inner


def _input_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional output kwarg.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "output" in kw_out:
            kw_out["output"] = dispatchables[1]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)