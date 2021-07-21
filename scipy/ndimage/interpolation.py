# This file is not meant for public use and will be removed in the future
# versions of SciPy. Use the `scipy.ndimage` namespace for importing the
# functions included below.

import warnings
from . import _multimethods

__all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform',
           'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.interpolation has no attribute {name}.")

    warnings.warn(f"Please use `{name}` from the `scipy.ndimage` namespace, "
                  "the `scipy.ndimage.interpolation` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_multimethods, name)
