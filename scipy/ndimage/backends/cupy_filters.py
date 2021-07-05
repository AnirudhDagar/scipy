import cupy as cp
import numpy as np
import scipy.ndimage as _scipy_ndimage
import cupyx.scipy.ndimage as _cupy_ndimage
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
    elif dispatchables == None:
        replaced = []
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


# def _asarray(x):
#     """Convert scalars to a sequence, otherwise pass through ``x`` unchanged"""
#     if isinstance(x, Number) or isinstance(x, Sequence):
#         return cp.array(x)
#     return x


@_implements(_scipy_ndimage.correlate1d)
def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    return _cupy_ndimage.correlate1d(
        input, weights, axis=axis, output=output, mode=mode, cval=cval, origin=origin)
correlate1d.__doc__ = _cupy_ndimage.correlate1d.__doc__


@_implements(_scipy_ndimage.convolve1d)
def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    return _cupy_ndimage.convolve1d(
        input, weights, axis=axis, output=output, mode=mode, cval=cval, origin=origin)
convolve1d.__doc__ = _cupy_ndimage.convolve1d.__doc__


@_implements(_scipy_ndimage.gaussian_filter1d)
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0):
    return _cupy_ndimage.gaussian_filter1d(
        input, sigma, axis=axis, order=order, output=output, mode=mode, cval=cval, truncate=truncate)
gaussian_filter1d.__doc__ = _cupy_ndimage.gaussian_filter1d.__doc__


@_implements(_scipy_ndimage.gaussian_filter)
def gaussian_filter(input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0):
    return _cupy_ndimage.gaussian_filter(
        input, sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate)
gaussian_filter.__doc__ = _cupy_ndimage.gaussian_filter.__doc__


@_implements(_scipy_ndimage.prewitt)
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return _cupy_ndimage.prewitt(input, axis=axis, output=output, mode=mode, cval=cval)
prewitt.__doc__ = _cupy_ndimage.prewitt.__doc__


@_implements(_scipy_ndimage.sobel)
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return _cupy_ndimage.sobel(input, axis=axis, output=output, mode=mode, cval=cval)
sobel.__doc__ = _cupy_ndimage.sobel.__doc__


# @_implements(_scipy_ndimage.sobel)
# def generic_laplace(input, derivative2, output=None, mode="reflect", cval=0.0,
#                     extra_arguments=(), extra_keywords=None):
#     return (input, derivative2, mark_non_coercible(output))

# @_implements(_scipy_ndimage.laplace)
# def laplace(input, output=None, mode="reflect", cval=0.0):
#     return (input, mark_non_coercible(output))
# sobel.__doc__ = _cupy_ndimage.sobel.__doc__

# @_implements(_scipy_ndimage.gaussian_laplace)
# def gaussian_laplace(input, sigma, output=None, mode="reflect",
#                      cval=0.0, **kwargs):
#     return _cupy_ndimage.gaussian_laplace(input, sigma, output=output, mode=mode, cval=cval)
# sobel.__doc__ = _cupy_ndimage.sobel.__doc__


@_implements(_scipy_ndimage.correlate)
def correlate(input, weights, output=None, mode='reflect', cval=0.0,
              origin=0):
    return _cupy_ndimage.correlate(
        input, weights, output=output, mode=mode, cval=cval, origin=origin)
correlate.__doc__ = _cupy_ndimage.correlate.__doc__


@_implements(_scipy_ndimage.convolve)
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    return _cupy_ndimage.convolve(
        input, weights, output=output, mode=mode, cval=cval, origin=origin)
convolve.__doc__ = _cupy_ndimage.convolve.__doc__


@_implements(_scipy_ndimage.uniform_filter1d)
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return _cupy_ndimage.uniform_filter1d(input, size,
        axis=axis, output=output, mode=mode, cval=cval, origin=origin)
uniform_filter1d.__doc__ = _cupy_ndimage.uniform_filter1d.__doc__


@_implements(_scipy_ndimage.uniform_filter)
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0):
    return _cupy_ndimage.uniform_filter(
        input, size=size, output=output, mode=mode, cval=cval, origin=origin)
uniform_filter.__doc__ = _cupy_ndimage.uniform_filter.__doc__


@_implements(_scipy_ndimage.minimum_filter1d)
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return _cupy_ndimage.minimum_filter1d(input, size,
        axis=axis, output=output, mode=mode, cval=cval, origin=origin)
minimum_filter1d.__doc__ = _cupy_ndimage.minimum_filter1d.__doc__


@_implements(_scipy_ndimage.maximum_filter1d)
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return _cupy_ndimage.maximum_filter1d(input, size,
        axis=axis, output=output, mode=mode, cval=cval, origin=origin)
maximum_filter1d.__doc__ = _cupy_ndimage.maximum_filter1d.__doc__

@_implements(_scipy_ndimage.percentile_filter)
def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
    return _cupy_ndimage.percentile_filter(input, percentile, size=size,
        footprint=footprint, output=output, mode=mode, cval=cval, origin=origin)
percentile_filter.__doc__ = _cupy_ndimage.percentile_filter.__doc__


@_implements(_scipy_ndimage.laplace)
def laplace(input, output=None, mode="reflect", cval=0.0):
    return _cupy_ndimage.laplace(input, output=output, mode=mode, cval=cval)
laplace.__doc__ = _cupy_ndimage.laplace.__doc__


@_implements(_scipy_ndimage.generate_binary_structure)
def generate_binary_structure(rank, connectivity):
    return _cupy_ndimage.generate_binary_structure(rank, connectivity)
generate_binary_structure.__doc__ = _cupy_ndimage.generate_binary_structure.__doc__


@_implements(_scipy_ndimage.minimum_filter)
def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    return _cupy_ndimage.minimum_filter(input, size=size, footprint=footprint,
        output=output, mode=mode, cval=cval, origin=origin)
minimum_filter.__doc__ = _cupy_ndimage.minimum_filter.__doc__


@_implements(_scipy_ndimage.fourier_gaussian)
def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    return _cupy_ndimage.generate_binary_structure(input, sigma, n=n, axis=axis, output=output)
fourier_gaussian.__doc__ = _cupy_ndimage.fourier_gaussian.__doc__
