from . import _multimethods
from ._multimethods import (iterate_structure, generate_binary_structure,
                            binary_erosion, binary_dilation, binary_opening,
                            binary_closing, binary_hit_or_miss,
                            binary_propagation, binary_fill_holes,
                            grey_erosion, grey_dilation, grey_opening,
                            grey_closing, white_tophat, black_tophat,
                            morphological_laplace, morphological_gradient,
                            distance_transform_bf, distance_transform_cdt,
                            distance_transform_edt)

__all__ = [s for s in dir() if not s.startswith("_")]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.morphology has no attribute {name}.")

    return getattr(_multimethods, name)
