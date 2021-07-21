from . import _multimethods
from ._multimethods import (correlate1d, convolve1d, gaussian_filter1d,
                            gaussian_filter, prewitt, sobel, generic_laplace,
                            laplace, gaussian_laplace, generic_filter,
                            generic_gradient_magnitude, generic_filter1d,
                            gaussian_gradient_magnitude, correlate, convolve,
                            uniform_filter1d, uniform_filter, minimum_filter1d,
                            maximum_filter1d, minimum_filter, maximum_filter,
                            rank_filter, median_filter, percentile_filter)

__all__ = [s for s in dir() if not s.startswith("_")]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.filters has no attribute {name}.")

    return getattr(_multimethods, name)
