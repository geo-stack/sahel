# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel
# =============================================================================

# ---- Standard imports
import functools
import inspect

# ---- Third party imports
from scipy.ndimage import generic_filter
from skimage.feature import local_binary_pattern as lbp
import numpy as np
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler
from scipy import stats


def filterfunc(func):
    """
    A decorator function to wrap data horizontally and pad data vertically
    before applying the filter function and unwrap and unpadd the data
    afterward.

    We do this because it is not possible with the generic_filter function
    to define different method to extend the data beyond its boundaries
    (horizontally and vertically).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # The EditCommand object only accept keyword arguments, so we
        # need to map the positional arguments to their keyword.
        kwargs = dict(
            inspect.signature(func)
            .bind(*args, **kwargs)
            .arguments
            )
        data = kwargs.pop(next(iter(kwargs)))
        n, m = data.shape

        if 'window' in kwargs:
            npad, mpad = kwargs['window']
        elif 'radius' in kwargs:
            npad = kwargs['radius']
            mpad = kwargs['radius']
        else:
            npad, mpad = data.shape

        # Wrap data horizontally.
        data = np.hstack([data[:, -mpad:], data, data[:, :mpad]])

        # Pad data vertically.
        pad_top = np.tile(data[0, :], npad).reshape(-1, data.shape[1])
        pad_bottom = np.tile(data[-1, :], npad).reshape(-1, data.shape[1])
        data = np.vstack([pad_top, data, pad_bottom])

        # Return the unwraped and unpaded filtered data.
        return func(data, **kwargs)[npad:-npad, mpad:-mpad]
    return wrapper


def nanfunc(name, arr):
    # We define our own nan function to avoid RuntimeWarning when arr
    # is composed entirely of nan values.
    if np.isnan(arr).all():
        return np.nan
    else:
        return getattr(np, name)(arr)


# ---- Filter functions
@filterfunc
def local_max(grid, window=(3, 3)):
    return generic_filter(
        grid, function=lambda arr: nanfunc('nanmax', arr), size=window)


@filterfunc
def local_min(grid, window=(3, 3)):
    return generic_filter(
        grid, function=lambda arr: nanfunc('nanmin', arr), size=window)


@filterfunc
def local_med(grid, window=(3, 3)):
    return generic_filter(
        grid, function=lambda arr: nanfunc('nanmedian', arr), size=window)


@filterfunc
def local_mean(grid, window=(3, 3)):
    return generic_filter(
        grid, function=lambda arr: nanfunc('nanmean', arr), size=window)


@filterfunc
def local_var(grid, window=(3, 3)):
    return generic_filter(
        grid, function=lambda arr: nanfunc('nanvar', arr), size=window)


@filterfunc
def local_kurtosis(grid, window=(3, 3)):
    function = stats.kurtosis
    function.axis = None
    function.fisher = False
    function.nan_policy = 'omit'
    return generic_filter(grid, function=function, size=window)


@filterfunc
def local_skewness(grid, window=(3, 3)):
    function = stats.skew
    function.axis = None
    function.nan_policy = 'omit'
    return generic_filter(grid, function=function, size=window)


@filterfunc
def local_binary_pattern(grid, radius=1, method='default'):
    # methods = ['default', 'uniform', 'var', 'ror', 'nri_uniform']
    return lbp(grid, P=2*radius+1, R=radius, method=method)


@filterfunc
def make_hilbert(array):
    ss = StandardScaler()
    norm_grid = ss.fit_transform(array.reshape(-1, 1))
    return np.imag(
        np.apply_along_axis(hilbert, 1, norm_grid.reshape(array.shape))
        )
