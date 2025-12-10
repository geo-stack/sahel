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

# ---- Third party imports
import numpy as np
from numba import njit, prange


# ---- Local imports
from hdml.math import bresenham_line

# IMPORTANT NOTE: When using @njit with 'fastmath=True', 'np.isnan'
# does not work, so we expect nan values to be '-32768'.
NODATA = -32768


# ---- Filter functions
@njit(parallel=True, fastmath=True)
def downslope_stats_numba(grid, stream_rows, stream_cols, fisher=False):

    nr, nc = grid.shape

    results = np.empty((6, nr, nc), dtype=np.float32)
    for i in prange(nr):
        for j in range(nc):
            if grid[i, j] != NODATA:

                line_pts = bresenham_line(
                    i, j, stream_rows[i, j], stream_cols[i, j]
                    )

                npts = len(line_pts)
                line_arr = np.empty((1, npts), dtype=np.float32)
                for k in range(npts):
                    line_arr[0, k] = grid[
                        line_pts[k, 0], line_pts[k, 1]
                        ]

                min_v, max_v, mean_v, var_v, skew_v, kurt_v = (
                    window_stats_numba(line_arr, fisher=fisher)
                    )

                results[0, i, j] = min_v
                results[1, i, j] = max_v
                results[2, i, j] = mean_v
                results[3, i, j] = var_v
                results[4, i, j] = skew_v
                results[5, i, j] = kurt_v
            else:
                results[:, i, j] = NODATA

    return results


@njit(parallel=True, fastmath=True)
def local_stats_numba(grid, window, fisher=False):
    """
    Calculate local statistics using a square window for each pixel.

    Edge pixels use truncated windows. Designed to be used within a
    higher-level function that handles padding and clipping.

    Parameters
    ----------
    grid : ndarray
        2D input array.
    window : int
        Size of the square window (must be odd).
    fisher : bool, optional
        If True, return Fisher's kurtosis (excess kurtosis).
        If False, return Pearson's kurtosis.  Default is False.

    Returns
    -------
    ndarray
        3D array of shape (6, nr, nc) where the first dimension contains:
        [0] = min
        [1] = max
        [2] = mean
        [3] = variance
        [4] = skewness
        [5] = kurtosis
    """
    nr, nc = grid.shape
    half_win = window // 2

    results = np.full((6, nr, nc), NODATA, dtype=np.float32)

    for i in prange(nr):
        i_min = max(0, i - half_win)
        i_max = min(nr, i + half_win + 1)
        for j in range(nc):
            if grid[i, j] != NODATA:
                j_min = max(0, j - half_win)
                j_max = min(nc, j + half_win + 1)

                window_data = grid[i_min:i_max, j_min:j_max]
                min_v, max_v, mean_v, var_v, skew_v, kurt_v = (
                    window_stats_numba(window_data, fisher=fisher)
                    )

                results[0, i, j] = min_v
                results[1, i, j] = max_v
                results[2, i, j] = mean_v
                results[3, i, j] = var_v
                results[4, i, j] = skew_v
                results[5, i, j] = kurt_v
            else:
                results[:, i, j] = NODATA

    return results


@njit(fastmath=True)
def window_stats_numba(arr, fisher=False):
    """
    Calculate comprehensive statistics for a 2D array window.

    Computes minimum, maximum, mean, variance, skewness, and kurtosis
    in two efficient passes through the data, with proper NaN handling.

    Parameters
    ----------
    arr : ndarray
        2D array (window data).
    fisher : bool, optional
        If True, return Fisher's kurtosis (excess kurtosis, normal = 0).
        If False, return Pearson's kurtosis (normal = 3).  Default is False.

    Returns
    -------
    tuple of float
        (min, max, mean, variance, skewness, kurtosis)
        Returns (nan, nan, nan, nan, nan, nan) if fewer than 4 valid values.
    """
    nr, nc = arr.shape
    if nr * nc < 4:
        return NODATA, NODATA, NODATA, NODATA, NODATA, NODATA

    # First pass: calculate mean, min, max.
    mean_val = 0.0
    min_val = np.inf
    max_val = -np.inf
    count = 0
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] != NODATA:
                val = arr[i, j]
                mean_val += val
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                count += 1

    if count < 4:
        return NODATA, NODATA, NODATA, NODATA, NODATA, NODATA

    mean_val /= count

    # Second pass: calculate variance, third moment, and fourth moment.
    var = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(nr):
        for j in range(nc):
            if not np.isnan(arr[i, j]):
                diff = arr[i, j] - mean_val
                diff2 = diff * diff
                var += diff2
                m3 += diff2 * diff
                m4 += diff2 * diff2

    var /= count
    m3 /= count
    m4 /= count

    if var == 0.0:
        skew = NODATA
        kurt = NODATA
    else:
        # Calculate skewness
        skew = m3 / (var ** 1.5)

        # Calculate kurtosis
        kurt = m4 / (var * var)

        # Apply kurtosis-skewness bound: Kurt >= Skew² + 1
        min_kurt = skew * skew + 1.0
        if kurt < min_kurt:
            kurt = min_kurt

        if fisher:
            kurt -= 3.0

    return min_val, max_val, mean_val, var, skew, kurt


if __name__ == '__main__':
    import rasterio
    dist_stream = "D:/Projets/sahel/data/training/tiles (overlapped)/dist_stream/dist_stream_tile_017_012.tif"
    with rasterio.open(dist_stream) as src:
        stream_rows = src.read(2).astype(int)
        stream_cols = src.read(3).astype(int)

    results = stream_downslope_stats(grid, rows_stream, cols_stream)
