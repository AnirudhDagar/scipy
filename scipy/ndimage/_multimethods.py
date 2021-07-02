import functools
import numpy as np
from scipy._lib.uarray import Dispatchable, all_of_type, create_multimethod, generate_multimethod
# from unumpy import dtype, ndarray, mark_dtype
from scipy.ndimage import _api

# mark_dtype = mark_as(dtype)
mark_non_coercible = lambda x: Dispatchable(x, np.ndarray, coercible=False)

create_ndimage = functools.partial(create_multimethod, domain="numpy.scipy.ndimage")

# def _dtype_argreplacer(args, kwargs, dispatchables):
#     def replacer(*a, dtype=None, **kw):
#         out_kw = kw.copy()
#         out_kw["dtype"] = dispatchables[0]
#         if "out" in out_kw:
#             out_kw["out"] = dispatchables[1]`

#         return a, out_kw

#     return replacer(*args, **kwargs)


# def _self_argreplacer(args, kwargs, dispatchables):
#     def self_method(a, *args, **kwargs):
#         kw_out = kwargs.copy()
#         if "out" in kw_out:
#             kw_out["out"] = dispatchables[1]

#         return (dispatchables[0],) + args, kw_out

#     return self_method(*args, **kwargs)


# def _identity_arg_replacer(args, kwargs, arrays):
#     return args, kwargs


# def _image_arg_replacer(args, kwargs, dispatchables):
#     """
#     uarray argument replacer to replace the input image (``image``) and
#     """
#     def self_method(image, *args, **kwargs):
#         kw_out = kwargs.copy()
#         # if "mask" in kw:
#         #     kw_out["mask"] = dispatchables[1]
#         return (dispatchables[0],) + args, kw_out
#     return self_method(*args, **kwargs)

# def _image_triple_arg_replacer(args, kwargs, dispatchables):
#     """
#     uarray argument replacer to replace the input image (``image``) and
#     """
#     def self_method(image1, image2, image3, **kwargs):
#         return dispatchables[:2], kwargs
#     return self_method(*args, **kwargs)


# def _image_and_mask_arg_replacer(args, kwargs, dispatchables):
#     """
#     uarray argument replacer to replace the input image (``image``) and
#     """
#     def self_method(image, *args, **kwargs):
#         kw_out = kwargs.copy()
#         if "mask" in kw:
#             kw_out["mask"] = dispatchables[1]
#         return (dispatchables[0],) + args, kw_out
#     return self_method(*args, **kwargs)


# def _image_and_hist_arg_replacer(args, kwargs, dispatchables):
#     """
#     uarray argument replacer to replace the input image (``image``) and
#     """
#     def self_method(image, *args, **kwargs):
#         kw_out = kwargs.copy()
#         if "hist" in kw:
#             kw_out["hist"] = dispatchables[1]
#         return (dispatchables[0],) + args, kw_out
#     return self_method(*args, **kwargs)


# def _dispatch_identity(func):
#     """
#     Function annotation that creates a uarray multimethod from the function
#     """
#     return generate_multimethod(func, _identity_arg_replacer, domain="numpy.skimage.filters")



# def _x_replacer(args, kwargs, dispatchables):
#     """
#     uarray argument replacer to replace the transform input array (``x``)
#     """
#     if len(args) > 0:
#         return (dispatchables[0],) + args[1:], kwargs
#     kw = kwargs.copy()
#     kw['x'] = dispatchables[0]
#     return args, kw


def _image_weights_arg_replacer(args, kwargs, dispatchables):
    import pdb; pdb.set_trace();
    kw_out = kwargs.copy()
    if len(args)>0:
        return dispatchables[:2] + args[2:], kw_out
    if "output" in kw_out:
        kw_out["output"] = dispatchables[-1]
    return (dispatchables[:2],) + args, kw_out


def _dispatch_image(func):
    return generate_multimethod(func, _image_weights_arg_replacer, domain="numpy.scipy.ndimage")

# @_dispatch_image
# def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
#              multichannel=None, preserve_range=False, truncate=4.0):
#     return (Dispatchable(image, np.ndarray),)




""" filters.py multimethods """

# TODO: Add real docstring
# @_ni_docstrings.docfiller
@_dispatch_image
@all_of_type(np.ndarray)
def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    return input, weights, mark_non_coercible(output)


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
#     return (input, weights, mark_non_coercible(output))
    

# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
#                       mode="reflect", cval=0.0, truncate=4.0):
#     return (input, sigma, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def gaussian_filter(input, sigma, order=0, output=None,
#                     mode="reflect", cval=0.0, truncate=4.0):
#     return (input, sigma, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def generic_laplace(input, derivative2, output=None, mode="reflect",
#                     cval=0.0,
#                     extra_arguments=(),
#                     extra_keywords=None):
#     return (input, derivative2, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def laplace(input, output=None, mode="reflect", cval=0.0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def gaussian_laplace(input, sigma, output=None, mode="reflect",
#                      cval=0.0, **kwargs):
#     return (input, sigma, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def generic_gradient_magnitude(input, derivative, output=None,
#                                mode="reflect", cval=0.0,
#                                extra_arguments=(), extra_keywords=None):
#     return (input, derivative, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def gaussian_gradient_magnitude(input, sigma, output=None,
#                                 mode="reflect", cval=0.0, **kwargs):
#     return (input, weights, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_weights_arg_replacer)
# @all_of_type(np.ndarray)
# def correlate(input, weights, output=None, mode='reflect', cval=0.0,
#               origin=0):
#     return (input, weights, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def convolve(input, weights, output=None, mode='reflect', cval=0.0,
#              origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def uniform_filter1d(input, size, axis=-1, output=None,
#                      mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def uniform_filter(input, size=3, output=None, mode="reflect",
#                    cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))

# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def minimum_filter1d(input, size, axis=-1, output=None,
#                      mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))

# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def maximum_filter1d(input, size, axis=-1, output=None,
#                      mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))

# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def minimum_filter(input, size=None, footprint=None, output=None,
#                    mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def maximum_filter1d(input, size, axis=-1, output=None,
#                      mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def rank_filter(input, rank, size=None, footprint=None, output=None,
#                 mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def median_filter(input, size=None, footprint=None, output=None,
#                   mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def percentile_filter(input, percentile, size=None, footprint=None,
#                       output=None, mode="reflect", cval=0.0, origin=0):
#     return (input, mark_non_coercible(output))


# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def generic_filter1d(input, function, filter_size, axis=-1,
#                      output=None, mode="reflect", cval=0.0, origin=0,
#                      extra_arguments=(), extra_keywords=None):
#     return (input, mark_non_coercible(output))



# # TODO: Add real docstring
# # @_ni_docstrings.docfiller
# @create_ndimage(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def generic_filter(input, function, size=None, footprint=None,
#                    output=None, mode="reflect", cval=0.0, origin=0,
#                    extra_arguments=(), extra_keywords=None):
#     return (input, mark_non_coercible(output))


# """ _gaussian.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def _guess_spatial_dimensions(image):
#     return (image,)


# """ _gabor.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None,
#           sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0):
#     return (image, )
# gabor.__doc__ = _api.gabor.__doc__


# @create_skimage_filters(_identity_arg_replacer)
# def gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
#                  n_stds=3, offset=0):
#     return
# gabor_kernel.__doc__ = _api.gabor_kernel.__doc__


# """ _median.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def median(image, selem=None, out=None, mode='nearest', cval=0.0,
#            behavior='ndimage'):
#     return (image,)
# median.__doc__ = _api.median.__doc__


# """ _rank_order.py multimethods """

# @create_skimage_filters(_identity_arg_replacer)
# def rank_order(image):
#     return
# rank_order.__doc__ = _api.rank_order.__doc__


# """ _sparse.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def correlate_sparse(image, kernel, mode='reflect'):
#     return (image,)
# correlate_sparse.__doc__ = _api.correlate_sparse.__doc__

# from .._shared import utils


# """ _unsharp_mask.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False,
#                  preserve_range=False, *, channel_axis=None):
#     return (image,)
# unsharp_mask.__doc__ = _api.unsharp_mask.__doc__


# """ _window.py multimethods """

# @create_skimage_filters(_identity_arg_replacer)
# def window(window_type, shape, warp_kwargs=None):
#     return
# window.__doc__ = _api.window.__doc__


# """ edges.py multimethods """

# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
#     return (image, mask)
# sobel.__doc__ = _api.sobel.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def sobel_h(image, mask=None):
#     return (image, mask)
# sobel_h.__doc__ = _api.sobel_h.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def sobel_v(image, mask=None):
#     return (image, mask)
# sobel_v.__doc__ = _api.sobel_v.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
#     return (image, mask)
# scharr.__doc__ = _api.scharr.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def scharr_h(image, mask=None):
#     return (image, mask)
# scharr_h.__doc__ = _api.scharr_h.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def scharr_v(image, mask=None):
#     return (image, mask)
# scharr_v.__doc__ = _api.scharr_v.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
#     return (image, mask)
# prewitt.__doc__ = _api.prewitt.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def prewitt_h(image, mask=None):
#     return (image, mask)
# prewitt_h.__doc__ = _api.prewitt_h.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def prewitt_v(image, mask=None):
#     return (image, mask)
# prewitt_v.__doc__ = _api.prewitt_v.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def roberts(image, mask=None):
#     return (image, mask)
# roberts.__doc__ = _api.roberts.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def roberts_pos_diag(image, mask=None):
#     return (image, mask)
# roberts_pos_diag.__doc__ = _api.roberts_pos_diag.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def roberts_neg_diag(image, mask=None):
#     return (image, mask)
# roberts_neg_diag.__doc__ = _api.roberts_neg_diag.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def laplace(image, ksize=3, mask=None):
#     return (image, mask)
# laplace.__doc__ = _api.laplace.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def farid(image, *, mask=None):
#     return (image, mask)
# farid.__doc__ = _api.farid.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def farid_h(image, *, mask=None):
#     return (image, mask)
# farid_h.__doc__ = _api.farid_h.__doc__


# @create_skimage_filters(_image_and_mask_arg_replacer)
# @all_of_type(np.ndarray)
# def farid_v(image, *, mask=None):
#     return (image, mask)
# farid_v.__doc__ = _api.farid_v.__doc__


# """ lpi_filter.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def forward(data, impulse_response=None, filter_params={},
#             predefined_filter=None):
#     return (data,)
# forward.__doc__ = _api.forward.__doc__

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def inverse(data, impulse_response=None, filter_params={}, max_gain=2,
#             predefined_filter=None):
#     return (data,)
# inverse.__doc__ = _api.inverse.__doc__

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def wiener(data, impulse_response=None, filter_params={}, K=0.25,
#            predefined_filter=None):
#     return (data,)
# wiener.__doc__ = _api.wiener.__doc__


# """ ridges.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def compute_hessian_eigenvalues(image, sigma, sorting='none', mode='constant',
#                                 cval=0):
#     return (image,)
# compute_hessian_eigenvalues.__doc__ = _api.compute_hessian_eigenvalues.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def meijering(image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True,
#               mode='reflect', cval=0):
#     return (image,)
# meijering.__doc__ = _api.meijering.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode=None, cval=0):
#     return (image,)
# sato.__doc__ = _api.sato.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def frangi(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
#            alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect',
#            cval=0):
#     return (image,)
# frangi.__doc__ = _api.frangi.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def hessian(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
#             alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode=None,
#             cval=0):
#     return (image,)
# hessian.__doc__ = _api.hessian.__doc__


# """ thresholding.py multimethods """

# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def try_all_threshold(image, figsize=(8, 5), verbose=True):
#     return (image,)
# try_all_threshold.__doc__ = _api.try_all_threshold.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_local(image, block_size, method='gaussian', offset=0,
#                     mode='reflect', param=None, cval=0):
#     return (image,)
# threshold_local.__doc__ = _api.threshold_local.__doc__


# @create_skimage_filters(_image_and_hist_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_otsu(image=None, nbins=256, *, hist=None):
#     return (image, hist)
# threshold_otsu.__doc__ = _api.threshold_otsu.__doc__


# @create_skimage_filters(_image_and_hist_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_yen(image=None, nbins=256, *, hist=None):
#     return (image, hist)
# threshold_yen.__doc__ = _api.threshold_yen.__doc__


# @create_skimage_filters(_image_and_hist_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
#     return (image, hist)
# threshold_isodata.__doc__ = _api.threshold_isodata.__doc__


# @create_skimage_filters(_image_and_hist_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_li(image, *, tolerance=None, initial_guess=None,
#                  iter_callback=None):
#     return (image, hist)
# threshold_li.__doc__ = _api.threshold_li.__doc__


# @create_skimage_filters(_image_and_hist_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_minimum(image=None, nbins=256, max_iter=10000, *, hist=None):
#     return (image, hist)
# threshold_minimum.__doc__ = _api.threshold_minimum.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_mean(image):
#     return (image,)
# threshold_mean.__doc__ = _api.threshold_mean.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_triangle(image, nbins=256):
#     return (image,)
# threshold_triangle.__doc__ = _api.threshold_triangle.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_niblack(image, window_size=15, k=0.2):
#     return (image,)
# threshold_niblack.__doc__ = _api.threshold_niblack.__doc__


# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_sauvola(image, window_size=15, k=0.2, r=None):
#     return (image,)
# threshold_sauvola.__doc__ = _api.threshold_sauvola.__doc__


# # TODO: low, high can be float or np.ndarray
# @create_skimage_filters(_image_triple_arg_replacer)
# @all_of_type(np.ndarray)
# def apply_hysteresis_threshold(image, low, high):
#     return (image, low, high)
# apply_hysteresis_threshold.__doc__ = _api.apply_hysteresis_threshold.__doc__


# # TODO: low, high can be float or np.ndarray
# @create_skimage_filters(_image_arg_replacer)
# @all_of_type(np.ndarray)
# def threshold_multiotsu(image, classes=3, nbins=256):
#     return (image, low, high)
# threshold_multiotsu.__doc__ = _api.threshold_multiotsu.__doc__
