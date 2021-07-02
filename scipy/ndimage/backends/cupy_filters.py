import cupy as cp
import numpy as np
import scipy.ndimage as _scipy_ndimage
from cupyx.scipy.ndimage import filters as _cupy_filters
from collections import Sequence
from numbers import Number

# Backend support for scipy.ndimage

__ua_domain__ = 'numpy.scipy.ndimage'
_implemented = {}


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                cp.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    # if not all(d.type is not np.ndarray or isinstance(r, cp.ndarray)
    #            for r, d in zip(replaced, dispatchables)):
    #     return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented

    # may need warnings or errors related to API changes here
    #if 'multichannel' in kwargs and not _skimage_1_0:
    #    warnings.warn('The \'multichannel\' argument is not supported for scikit-image >= 1.0')
    return fn(*args, **kwargs)


def _implements(scipy_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[scipy_func] = func
        return func

    return inner


def _asarray(x):
    """Convert scalars to a sequence, otherwise pass through ``x`` unchanged"""
    if isinstance(x, Number) or isinstance(x, Sequence):
        return cp.array(x)
    return x


@_implements(_scipy_ndimage.correlate1d)
def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    weights = _asarray(weights)
    return _cupy_filters.correlate1d(
        input, weights, axis=axis, output=output, mode=mode, cval=cval, origin=origin)
correlate1d.__doc__ = _cupy_filters.correlate1d.__doc__
