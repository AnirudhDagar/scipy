from . import _multimethods

__all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean',
           'variance', 'standard_deviation', 'minimum', 'maximum', 'median',
           'minimum_position', 'maximum_position', 'extrema', 'center_of_mass',
           'histogram', 'watershed_ift', 'sum_labels']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.ndimage.measurements has no attribute {name}.")

    return getattr(_multimethods, name)
