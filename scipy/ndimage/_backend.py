import scipy._lib.uarray as ua
from scipy.ndimage import _api


class _ScipyImageBackend:
    """The default backend for ndimage calculations

    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because in the
    future, ``uarray`` will treat the domain as a hierarchy. This means the user
    can install a single backend for ``numpy`` and have it implement
    ``numpy.scipy.ndimage`` as well.
    """
    __ua_domain__ = "numpy.scipy.ndimage"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        import pdb; pdb.set_trace()
        fn = getattr(_api, method.__name__, None)

        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)


_named_backends = {
    'scipy_image': _ScipyImageBackend,
}


def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend"""

    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError('Unknown backend {}'.format(backend)) from e

    if backend.__ua_domain__ != 'numpy.scipy.ndimage':
        raise ValueError('Backend does not implement "numpy.scipy.ndimage"')

    return backend


def set_global_backend(backend, coerce=False, only=False, try_last=False):
    """Sets the global ndimage backend

    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.ndimage``.

    Notes
    -----
    This will overwrite the previously set global backend, which, by default, is
    the SciPy implementation.

    Examples
    --------
    We can set the global ndimage backend:

    >>> from scipy.ndimage import correlate, set_global_backend
    >>> set_global_backend("scipy")  # Sets global backend. "scipy" is the default backend.
    >>> correlate(np.arange(10), [1, 2.5])  # Calls the global backend
    array([ 0,  2,  6,  9, 13, 16, 20, 23, 27, 30])
    """
    backend = _backend_from_arg(backend)
    ua.set_global_backend(backend, coerce=coerce, only=only, try_last=try_last)


def register_backend(backend):
    """
    Register a backend for permanent use.

    Registered backends have the lowest priority and will be tried after the
    global backend.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.ndimage``.

    Examples
    --------
    We can register a new ndimage backend:
    >>> from scipy.ndimage import correlate, register_backend, set_global_backend
    >>> class NoopBackend:  # Define an invalid Backend
    ...     __ua_domain__ = "numpy.scipy.ndimage"
    ...     def __ua_function__(self, func, args, kwargs):
    ...          return NotImplemented
    >>> set_global_backend(NoopBackend())  # Set the invalid backend as global
    >>> register_backend("scipy")  # Register a new backend
    >>> correlate(np.arange(10), [1, 2.5])  # The registered backend is called because the global backend returns `NotImplemented`
    array([ 0,  2,  6,  9, 13, 16, 20, 23, 27, 30])
    >>> set_global_backend("scipy")  # Restore global backend to default

    """
    backend = _backend_from_arg(backend)
    ua.register_backend(backend)


def set_backend(backend, coerce=False, only=False):
    """Context manager to set the backend within a fixed scope.

    Upon entering the ``with`` statement, the given backend will be added to
    the list of available backends with the highest priority. Upon exit, the
    backend is reset to the state before entering the scope.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool, optional
        Whether to allow expensive conversions for the ``x`` parameter. e.g.,
        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
    only : bool, optional
        If only is ``True`` and this backend returns ``NotImplemented``, then a
        BackendNotImplemented error will be raised immediately. Ignoring any
        lower priority backends.

    Examples
    -------- 
    >>> import scipy.ndimage as ndimage
    >>> with ndimage.set_backend('scipy', only=True):
    ...     ndimage.correlate(np.arange(10), [1, 2.5])  # Always calls the scipy implementation
    array([ 0,  2,  6,  9, 13, 16, 20, 23, 27, 30])
    """
    backend = _backend_from_arg(backend)
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    """Context manager to skip a backend within a fixed scope.

    Within the context of a ``with`` statement, the given backend will not be
    called. This covers backends registered both locally and globally. Upon
    exit, the backend will again be considered.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to skip.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Examples
    --------
    >>> import scipy.ndimage as ndimage
    >>> ndimage.correlate(np.arange(10), [1, 2.5])  # Calls default SciPy backend
    array([ 0,  2,  6,  9, 13, 16, 20, 23, 27, 30])
    >>> with ndimage.skip_backend('scipy'):  # We explicitly skip the SciPy backend
    ...     ndimage.correlate(np.arange(10), [1, 2.5])  # leaving no implementation available
    Traceback (most recent call last):
        ...
    BackendNotImplementedError: No selected backends had an implementation ...
    """
    backend = _backend_from_arg(backend)
    return ua.skip_backend(backend)


set_global_backend('scipy_image', try_last=True)
