################################
# This file needs some cleanup.
# This is utility for Array-API
# functionality in SciPy.
################################

# Methods required for the Array-API Demo
import numpy as np
import torch  # This will be removed in actual array-api util; no external dependency
from typing import Optional


def __array_namespace__(Array, /, *, api_version: Optional[str] = None) -> object:
    if api_version is not None:
        raise ValueError("Unrecognized array API version")
    import torch
    return torch


# Monkey-Patch the protocol to torch Tensor class
# This is actually not meant to be here.
# This part goes in the notebook demo and is kept here temporarily

# setattr(torch.Tensor, '__array_namespace__', __array_namespace__)


# TODO: Make sure numpy array passed also is monkey patched similarly.
# For now the workaround is to set xp to numpy whenever an input is not
# array api compliant. See the modification below where the code to raise
# ValueError is commented out.


def get_namespace(*xs):
    # `xs` contains one or more arrays, or possibly Python scalars (accepting
    # those is a matter of taste, but doesn't seem unreasonable).
    namespaces = {
        x.__array_namespace__() if hasattr(x, '__array_namespace__')
        else None for x in xs if not isinstance(x, (bool, int, float, complex))
    }
    if not namespaces:
        raise ValueError("Unrecognized array input")
    if len(namespaces) != 1:
        raise ValueError(f"Multiple namespaces for array inputs: {namespaces}")
    xp, = namespaces
    if xp is None:
        # raise ValueError("The input is not a supported array type")
        # NumPy can't be directly monkeypatched hence I'm assuming if the input is not array api based
        # it is numpy's ndarray
        xp = np

    # Note these are specific for my PyTorch and numpy array api compliant
    # implementation only. These are added in anticipation of original array-api
    # release which is still in development.
    # Make some functions compatible in PyTorch using monkey patching
    if xp == torch:
        xp.asarray = xp.as_tensor
        xp.concat = xp.cat
    if xp == np:
        xp.concat = xp.concatenate

    return xp