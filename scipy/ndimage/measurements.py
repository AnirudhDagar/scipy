from . import _multimethods
from ._multimethods import (label, find_objects, labeled_comprehension,
                            sum, mean, variance, standard_deviation, minimum,
                            maximum, median, minimum_position,
                            maximum_position, extrema, center_of_mass,
                            histogram, watershed_ift, sum_labels)

__all__ = [s for s in dir() if not s.startswith("_")]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.measurements has no attribute {name}.")

    return getattr(_multimethods, name)
