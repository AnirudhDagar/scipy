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


def _double_input_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace two required input arrays
    and optional output kwarg.
    """
    def self_method(input1, input2, *args, **kwargs):
        kw_out = kwargs.copy()
        if "output" in kw_out:
            kw_out["output"] = dispatchables[2]
        return dispatchables[:2], kw_out
    return self_method(*args, **kwargs)


def _input_footprint_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional footprint and output kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "footprint" in kw_out:
            kw_out["footprint"] = dispatchables[1]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[2]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)



############################################
""" filters multimethods """
############################################

@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def correlate1d(input, weights, axis=-1, output=None,
                mode="reflect", cval=0.0, origin=0):
    return input, weights, mark_non_coercible(output)


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def convolve1d(input, weights, axis=-1, output=None,
               mode="reflect", cval=0.0, origin=0):
    return input, weights, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0, extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def laplace(input, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def gaussian_gradient_magnitude(input, sigma, output=None,
                                mode="reflect", cval=0.0, **kwargs):
    return input, mark_non_coercible(output)


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def correlate(input, weights, output=None, mode='reflect', cval=0.0,
              origin=0):
    return input, weights, mark_non_coercible(output)


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    return input, weights, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def generic_filter1d(input, function, filter_size, axis=-1,
                     output=None, mode="reflect", cval=0.0, origin=0,
                     extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
@_get_docs
def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0,
                   extra_arguments=(), extra_keywords=None):
    return input, footprint, mark_non_coercible(output)

