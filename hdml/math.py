# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel_water_table_ml
# =============================================================================


from typing import Tuple

# ---- Third party imports
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def bresenham_line(row0: int, col0: int, row1: int, col1: int) -> np.ndarray:
    """
    Generate coordinates of pixels along a line using Bresenham's algorithm.

    Parameters
    ----------
    row0 : int
        Starting row coordinate.
    col0 : int
        Starting column coordinate.
    row1 : int
        Ending row coordinate.
    col1 : int
        Ending column coordinate.
    thickness : int, optional
        Line thickness in pixels. Default is 1 (single-pixel line).
        For thickness > 1, returns all pixels within (thickness-1)/2 distance
        from the center line.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with integer [row, col] coordinates representing
        pixels along the line, ordered from start to end. Includes both
        endpoints.

    Notes
    -----
    - Uses only integer arithmetic (no floating-point operations).
    - Works in all octants (all directions and slopes).
    - For thickness=1, returns the standard Bresenham line.
    - For thickness>1, approximates line thickness using perpendicular offsets.

    References
    ----------
    Bresenham, J.E. (1965). "Algorithm for computer control of a digital
    plotter". IBM Systems Journal, 4(1), 25-30.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    row0, row1 = int(row0), int(row1)
    col0, col1 = int(col0), int(col1)

    # Calculate differences
    drow = abs(row1 - row0)
    dcol = abs(col1 - col0)

    # Determine direction of line
    row_step = 1 if row0 < row1 else -1
    col_step = 1 if col0 < col1 else -1

    # Current position
    row = row0
    col = col0

    # Bresenham's algorithm with integer-only arithmetic.
    points = np.empty((max(drow, dcol) + 1, 2), dtype=np.int32)
    write_idx = 0
    if drow > dcol:
        # Line is more vertical than horizontal.
        error = drow
        while row != row1:
            points[write_idx, 0] = row
            points[write_idx, 1] = col
            write_idx += 1

            error -= 2 * dcol
            if error < 0:
                col += col_step
                error += 2 * drow
            row += row_step
    else:
        # Line is more horizontal than vertical.
        error = dcol
        while col != col1:
            points[write_idx, 0] = row
            points[write_idx, 1] = col
            write_idx += 1

            error -= 2 * drow
            if error < 0:
                row += row_step
                error += 2 * dcol
            col += col_step

    # Add final point
    points[write_idx, 0] = row
    points[write_idx, 1] = col
    write_idx += 1

    return points


@njit(cache=True, fastmath=True)
def precompute_spiral_offsets(size: int) -> np.ndarray:
    """
    Precompute offsets for a size × size grid.

    Parameters
    ----------
    size : int
        Grid dimension (e.g., 1000 for a 1000×1000 grid).

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with dtype int32, where N = (2*size - 1)².
        Columns are:
        - 0: dr (row offset from center)
        - 1: dc (column offset from center)
        - 2: dist_squared (dr² + dc², squared Euclidean distance)

        Rows are sorted by dist_squared in ascending order (closest first).
    """
    # Maximum offset: from (0,0) to (size-1, size-1) = size-1
    max_offset = size - 1

    # Offsets range from -max_offset to +max_offset in each dimension
    # Number of values per dimension: 2 * max_offset + 1
    n = 2 * max_offset + 1

    offsets_array = np.empty((n * n, 3), dtype=np.int32)
    write_idx = 0
    for dr in range(-max_offset, max_offset + 1):
        for dc in range(-max_offset, max_offset + 1):
            dist_sq = dr * dr + dc * dc
            offsets_array[write_idx, 0] = dr
            offsets_array[write_idx, 1] = dc
            offsets_array[write_idx, 2] = dist_sq
            write_idx += 1

    # Sort by distance (closest first).
    sorted_indices = np. argsort(offsets_array[:, 2])
    offsets_sorted = offsets_array[sorted_indices]

    return offsets_sorted


if __name__ == '__main__':
    offsets = precompute_spiral_offsets(10)
