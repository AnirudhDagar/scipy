from . import _multimethods
from ._multimethods import (spline_filter1d, spline_filter,
                            geometric_transform, map_coordinates,
                            affine_transform, shift, zoom, rotate)

__all__ = [s for s in dir() if not s.startswith("_")]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.interpolation has no attribute {name}.")

    return getattr(_multimethods, name)
