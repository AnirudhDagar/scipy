import functools
import numpy as np
from scipy._lib.uarray import Dispatchable, all_of_type, create_multimethod
from scipy.ndimage import _api


mark_non_coercible = lambda x: Dispatchable(x, np.ndarray, coercible=False)
create_ndimage = functools.partial(create_multimethod, domain="numpy.scipy.ndimage")

# def _get_docs(func):
#     """Decorator to take the docstring from original function and assign to the multimethod"""
#     def inner(*args, **kwargs):
#         func.__doc__ = getattr(_api, func.__name__).__doc__
#         return func(*args, **kwargs)
#     return inner


def _identity_arg_replacer(args, kwargs, arrays):
    def self_method(*args, **kwargs):
        return args, kwargs
    return self_method(*args, **kwargs)


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


def _input_structure_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional structure and output kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "structure" in kw_out:
            kw_out["structure"] = dispatchables[1]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[2]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


def _input_structure_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional structure, mask and output kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "structure" in kw_out:
            kw_out["structure"] = dispatchables[1]
        if "mask" in kw_out:
            kw_out["mask"] = dispatchables[2]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[3]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


def _input_structure1_structure2_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional structure1, structure2 and output kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "structure1" in kw_out:
            kw_out["structure1"] = dispatchables[1]
        if "structure2" in kw_out:
            kw_out["structure2"] = dispatchables[2]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[3]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


def _input_footprint_structure_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array
    and optional footprint, structure and output kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "footprint" in kw_out:
            kw_out["footprint"] = dispatchables[1]
        if "structure" in kw_out:
            kw_out["structure"] = dispatchables[2]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[3]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


def _input_markers_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input and markers array.
    Optional structure and output kwargs are also handled.
    """
    def self_method(input, markers, *args, **kwargs):
        kw_out = kwargs.copy()
        if "structure" in kw_out:
            kw_out["structure"] = dispatchables[2]
        if "output" in kw_out:
            kw_out["output"] = dispatchables[3]
        return (dispatchables[:2],) + args, kw_out
    return self_method(*args, **kwargs)


def _input_labels_index_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input array and
    optional labels and index kwargs.
    """
    def self_method(input, *args, **kwargs):
        kw_out = kwargs.copy()
        if "labels" in kw_out:
            kw_out["labels"] = dispatchables[1]
        if "index" in kw_out:
            kw_out["index"] = dispatchables[2]
        return (dispatchables[0],) + args, kw_out
    return self_method(*args, **kwargs)


############################################
""" filters multimethods """
############################################


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    return input, weights, mark_non_coercible(output)
correlate1d.__doc__ = _api.correlate1d.__doc__


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    return input, weights, mark_non_coercible(output)
convolve1d.__doc__ = _api.convolve1d.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    return input, mark_non_coercible(output)
gaussian_filter1d.__doc__ = _api.gaussian_filter1d.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    return input, mark_non_coercible(output)
gaussian_filter.__doc__ = _api.gaussian_filter.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)
prewitt.__doc__ = _api.prewitt.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)
sobel.__doc__ = _api.sobel.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def generic_laplace(input, derivative2, output=None, mode="reflect", cval=0.0,
                    extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)
generic_laplace.__doc__ = _api.generic_laplace.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def laplace(input, output=None, mode="reflect", cval=0.0):
    return input, mark_non_coercible(output)
laplace.__doc__ = _api.laplace.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
    return input, mark_non_coercible(output)
gaussian_laplace.__doc__ = _api.gaussian_laplace.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)
generic_gradient_magnitude.__doc__ = _api.generic_gradient_magnitude.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def gaussian_gradient_magnitude(input, sigma, output=None,
                                mode="reflect", cval=0.0, **kwargs):
    return input, mark_non_coercible(output)
gaussian_gradient_magnitude.__doc__ = _api.gaussian_gradient_magnitude.__doc__


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
def correlate(input, weights, output=None, mode='reflect', cval=0.0,
              origin=0):
    return input, weights, mark_non_coercible(output)
correlate.__doc__ = _api.correlate.__doc__


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    return input, weights, mark_non_coercible(output)
convolve.__doc__ = _api.convolve.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)
uniform_filter1d.__doc__ = _api.uniform_filter1d.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0):
    return input, mark_non_coercible(output)
uniform_filter.__doc__ = _api.uniform_filter.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)
minimum_filter1d.__doc__ = _api.minimum_filter1d.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    return input, mark_non_coercible(output)
maximum_filter1d.__doc__ = _api.maximum_filter1d.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)
minimum_filter.__doc__ = _api.minimum_filter.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)
maximum_filter.__doc__ = _api.maximum_filter.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)
rank_filter.__doc__ = _api.rank_filter.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)
median_filter.__doc__ = _api.median_filter.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, mark_non_coercible(output)
percentile_filter.__doc__ = _api.percentile_filter.__doc__


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def generic_filter1d(input, function, filter_size, axis=-1,
                     output=None, mode="reflect", cval=0.0, origin=0,
                     extra_arguments=(), extra_keywords=None):
    return input, mark_non_coercible(output)
generic_filter1d.__doc__ = _api.generic_filter1d.__doc__


@create_ndimage(_input_footprint_arg_replacer)
@all_of_type(np.ndarray)
def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0,
                   extra_arguments=(), extra_keywords=None):
    return input, footprint, mark_non_coercible(output)
generic_filter.__doc__ = _api.generic_filter.__doc__


############################################
""" fourier multimethods """
############################################


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    return input, mark_non_coercible(output)


############################################
""" interpolation multimethods """
############################################


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def spline_filter1d(input, order=3, axis=-1, output=np.float64,
                    mode='mirror'):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def spline_filter(input, order=3, output=np.float64, mode='mirror'):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def geometric_transform(input, mapping, output_shape=None,
                        output=None, order=3,
                        mode='constant', cval=0.0, prefilter=True,
                        extra_arguments=(), extra_keywords={}):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def map_coordinates(input, coordinates, output=None, order=3,
                    mode='constant', cval=0.0, prefilter=True):
    return input, mark_non_coercible(output)


@create_ndimage(_double_input_arg_replacer)
@all_of_type(np.ndarray)
def affine_transform(input, matrix, offset=0.0, output_shape=None,
                     output=None, order=3, mode='constant',
                     cval=0.0, prefilter=True):
    return (input, matrix, mark_non_coercible(output))


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def shift(input, shift, output=None, order=3, mode='constant', cval=0.0,
          prefilter=True):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True, *, grid_mode=False):
    return input, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True):
    return input, mark_non_coercible(output)


############################################
""" measurement multimethods """
############################################


@create_ndimage(_input_structure_arg_replacer)
@all_of_type(np.ndarray)
def label(input, structure=None, output=None):
    return input, structure, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def find_objects(input, max_label=0):
    return (input,)


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def labeled_comprehension(input, labels, index, func, out_dtype,
                          default, pass_positions=False):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def sum_labels(input, labels=None, index=None):
    return input, labels, index


# alias for sum_labels; kept for backward compatibility
def sum(input, labels=None, index=None):
    return sum_labels(input, labels=labels, index=index)


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def mean(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def variance(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def standard_deviation(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def minimum(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def maximum(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def median(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def minimum_position(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def maximum_position(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def extrema(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def center_of_mass(input, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_labels_index_arg_replacer)
@all_of_type(np.ndarray)
def histogram(input, min, max, bins, labels=None, index=None):
    return input, labels, index


@create_ndimage(_input_markers_arg_replacer)
@all_of_type(np.ndarray)
def watershed_ift(input, markers, structure=None, output=None):
    return input, markers, structure, mark_non_coercible(output)


############################################
""" morphology multimethods """
############################################


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def iterate_structure(structure, iterations, origin=None):
    return (structure,)


@create_ndimage(_identity_arg_replacer)
def generate_binary_structure(rank, connectivity):
    return


@create_ndimage(_input_structure_mask_arg_replacer)
@all_of_type(np.ndarray)
def binary_erosion(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False):
    return input, structure, mask, mark_non_coercible(output)


@create_ndimage(_input_structure_mask_arg_replacer)
@all_of_type(np.ndarray)
def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    return input, structure, mask, mark_non_coercible(output)


@create_ndimage(_input_structure_mask_arg_replacer)
@all_of_type(np.ndarray)
def binary_opening(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    return input, structure, mask, mark_non_coercible(output)


@create_ndimage(_input_structure_mask_arg_replacer)
@all_of_type(np.ndarray)
def binary_closing(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    return input, structure, mask, mark_non_coercible(output)


@create_ndimage(_input_structure1_structure2_arg_replacer)
@all_of_type(np.ndarray)
def binary_hit_or_miss(input, structure1=None, structure2=None,
                       output=None, origin1=0, origin2=None):
    return input, structure1, structure2, mark_non_coercible(output)


@create_ndimage(_input_structure_mask_arg_replacer)
@all_of_type(np.ndarray)
def binary_propagation(input, structure=None, mask=None,
                       output=None, border_value=0, origin=0):
    return input, structure, mask, mark_non_coercible(output)


@create_ndimage(_input_structure_arg_replacer)
@all_of_type(np.ndarray)
def binary_fill_holes(input, structure=None, output=None, origin=0):
    return input, structure, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def grey_erosion(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def grey_dilation(input, size=None, footprint=None, structure=None,
                  output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def grey_opening(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def grey_closing(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def morphological_gradient(input, size=None, footprint=None, structure=None,
                           output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def morphological_laplace(input, size=None, footprint=None,
                          structure=None, output=None,
                          mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def white_tophat(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_footprint_structure_arg_replacer)
@all_of_type(np.ndarray)
def black_tophat(input, size=None, footprint=None,
                 structure=None, output=None, mode="reflect",
                 cval=0.0, origin=0):
    return input, footprint, structure, mark_non_coercible(output)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def distance_transform_bf(input, metric="euclidean", sampling=None,
                          return_distances=True, return_indices=False,
                          distances=None, indices=None):
    return (input,)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def distance_transform_cdt(input, metric='chessboard', return_distances=True,
                           return_indices=False, distances=None, indices=None):
    return (input,)


@create_ndimage(_input_arg_replacer)
@all_of_type(np.ndarray)
def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    return (input,)
