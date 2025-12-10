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
from typing import Any, Callable

# ---- Standard imports
from pathlib import Path

# ---- Third party imports
from numba import njit, prange
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects

# ---- Local imports
from hdml.math import bresenham_line, precompute_spiral_offsets
from hdml.localfilters import local_stats_numba, downslope_stats_numba, NODATA


def extract_ridges(geomorphons: Path, output: Path, ridge_size: int = 30,
                   flow_acc: Path = None, max_flow_acc: int = 2):
    """
    Extract ridges with a pure geomorphon or a flow accumulation filtering.

    Parameters
    ----------
    geomorphons : Path
        Path to geomorphons classification GeoTIFF. Expected to contain integer
        class values from WhiteboxTools geomorphons analysis (1-10), where:
        - 1 = Flat
        - 2 = Peak (summit)
        - 3 = Ridge
        Values outside 1-3 and nodata/NaN are treated as non-ridge pixels.
    output : Path
        Path for output binary ridge raster. Output is uint8 with values:
        - 1 = ridge pixel
        - 0 = non-ridge pixel
    ridge_size : int, optional
        Minimum ridge region size in pixels. Connected ridge regions smaller
        than this threshold are removed before skeletonization. Default is 30.
    flow_acc : Path, optional
        Path to flow accumulation GeoTIFF. If provided, uses flow accumulation
        filtering instead of skeletonization to thin ridges. If None, applies
        morphological skeletonization to create 1-pixel-wide ridges.
    max_flow_acc : int
        Maximum flow accumulation value for valid ridge pixels. Only used when
        flow_acc is provided. Typically 1-2 for true ridges. Default is 2.

    References
    ----------
    WhiteboxTools Geomorphons documentation:
    https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#Geomorphons

    """

    with rasterio.open(geomorphons) as src:
        geomorph_data = src.read(1)
        profile = src.profile.copy()

    # Handle nan and nodata
    geomorph_data = np.maximum(geomorph_data, 0)
    geomorph_data[np.isnan(geomorph_data)] = 0

    # Create ridge mask (classes 1, 2, 3).
    mask_ridges = (geomorph_data > 0) & (geomorph_data < 4)

    # Remove small ridge regions.
    filtered_ridges = remove_small_objects(mask_ridges, min_size=ridge_size)

    # Filter by flow accumulation (optional).
    if flow_acc is not None:
        # Method 1: Flow accumulation filtering (hydrological ridges)
        with rasterio.open(flow_acc) as src:
            acc_data = src.read(1)
            acc_data = np.maximum(acc_data, 0)

        # Keep only ridge pixels with low flow accumulation.
        low_flow_mask = acc_data < max_flow_acc
        ridges = (filtered_ridges & low_flow_mask).astype(np.uint8)
    else:
        # Method 2: Morphological skeletonization (topographic ridges)

        # Skeletonize to get 1-pixel-wide ridge lines.
        thin_ridges = skeletonize(filtered_ridges)

        # Remove tiny skeleton fragments.
        ridges = remove_small_objects(thin_ridges, min_size=2).astype(np.uint8)

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='deflate',
        nodata=0
        )
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(ridges.astype('int'), 1)


def dist_to_ridges(dem: Path, ridges: Path, streams: Path, output: Path,
                   topological: bool = False):
    """
    Calculate distance to the nearest ridge for each pixel in a DEM.

    For each pixel in the input DEM, computes the Euclidean distance to the
    nearest ridge pixel. When `topological=True`, only ridges that can be
    reached without crossing a stream pixel are considered valid.
    When `topological=False`, a simple Euclidean distance to the nearest
    ridge is computed (ignoring streams).

    Outputs both the distance (in map units/meters) and the coordinates of
    the nearest ridge pixel for every non-nodata pixel. Nodata areas in the
    DEM are preserved in all output bands.

    When topological mode is enabled, drainage boundaries are respected:
    for each pixel, only ridges that do not require crossing a stream are
    considered valid when determining the distance.

    For example:

        R₁
        |
    ----S----S----
             |
             X
             |
        R₂

    where, R = ridge, S = stream, X = current point

    Even if X is geometrically closer to R₁ than R₂, the function will
    select R₂ as the nearest ridge since the path to R₁ would cross a stream.

    Parameters
    ----------
    dem : Path
        Path to the Digital Elevation Model (DEM) GeoTIFF. Used for spatial
        reference and nodata masking. DEM, ridges, and streams rasters must
        be aligned (same CRS, resolution, and extent).
    ridges : Path
        Path to the ridges raster GeoTIFF. Non-zero values are treated as
        ridge pixels. Can be produced by the 'extract_ridges' function.
        Must be aligned with the DEM.
    streams : Path
        Path to the streams raster GeoTIFF. Non-zero values are treated as
        stream pixels. Typically generated using tools like WhiteboxTools
        'extract_streams'. Must be aligned with the DEM.
    output : Path
        Path for the output 3-band GeoTIFF file.
    topological : bool, default False
        If True, computes distance to the nearest ridge that can be reached
        without crossing a stream (topologically constrained).
        If False, computes simple Euclidean distance to the nearest ridge,
        ignoring streams entirely.

    Returns
    -------
    None
        Writes a 3-band GeoTIFF to disk with the following bands:
        - Band 1: Distance to nearest ridge (float32, in map units/meters)
        - Band 2: Row index of nearest ridge pixel (float32)
        - Band 3: Column index of nearest ridge pixel (float32)
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile.copy()
        dem_data = src.read(1)
        dem_nodata = src.nodata

        dem_transform = src.transform

        assert abs(dem_transform.e) == abs(dem_transform.a)
        pixel_size = abs(dem_transform.e)

    with rasterio.open(streams) as src:
        streams_data = src.read(1)
        streams_transform = src.transform

    with rasterio.open(ridges) as src:
        ridges_data = src.read(1)
        ridges_transform = src.transform

    if not np.any(ridges_data > 0):
        raise ValueError("No ridge pixels found!")

    assert dem_transform == ridges_transform == streams_transform

    results = _dist_to_ridges_numba(
        dem_data, streams_data, ridges_data,
        nodata=dem_nodata, pixel_size=pixel_size,
        topological=topological
        )

    # Write output raster with 3 bands (all float32):
    # Band 1: Distance to nearest ridge pixel in meters
    # Band 2: Row index of nearest ridge pixel
    # Band 3: Column index of nearest ridge pixel
    out_profile = dem_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=3,
        compress='deflate'
        )

    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(results[0], 1)
        dst.write(results[1], 2)
        dst.write(results[2], 3)

        # Add band descriptions
        dst.set_band_description(1, 'distance_to_ridge_meters')
        dst.set_band_description(2, 'nearest_ridge_row')
        dst.set_band_description(3, 'nearest_ridge_col')


@njit(parallel=True, fastmath=True)
def _dist_to_ridges_numba(
        dem_data: np.ndarray,
        streams_data: np.ndarray,
        ridges_data: np.ndarray,
        nodata: Any,
        pixel_size: float,
        topological: bool = False
        ) -> np.ndarray:
    """
    Compute distance to nearest ridge for each pixel in a DEM.

    Parameters
    ----------
    dem_data : np.ndarray
        2D array of DEM elevation values.
    streams_data : np.ndarray
        2D array where non-zero values indicate stream pixels.
    ridges_data : np.ndarray
        2D array where non-zero values indicate ridge pixels.
    nodata : Any
        Nodata value used in the DEM.
    pixel_size : float
        Size of each pixel in map units (meters).
    topological : bool, default False
        If True, only ridges reachable without crossing a stream are
        considered. If False, simple Euclidean distance to the nearest
        ridge is computed (ignoring streams).

    Returns
    -------
    np.ndarray
        3D array of shape (3, height, width) containing:
        - [0]: Distance to nearest ridge in meters
        - [1]: Row index of nearest ridge pixel
        - [2]: Column index of nearest ridge pixel
    """

    dem_height, dem_width = dem_data.shape
    spiral_offsets = precompute_spiral_offsets(max(dem_width, dem_height))
    ridge_dist = np.full((3, *dem_data.shape), nodata, dtype=np.float32)

    n_rows, n_cols = dem_data.shape
    for row in prange(n_rows):
        for col in range(n_cols):

            # Check if pixel is nodata.
            if dem_data[row, col] == nodata:
                continue

            # Check if pixel is a ridge.
            if ridges_data[row, col] == 1:
                ridge_dist[0, row, col] = 0
                ridge_dist[1, row, col] = row
                ridge_dist[2, row, col] = col
                continue

            # Find the topologically nearest ridge.
            for dr, dc, dist_sqrt in spiral_offsets:

                if dr == 0 and dc == 0:
                    # We already did checks for the point itself.
                    continue

                # Check that the offset point is in the grid.
                if not (0 <= row + dr < dem_height):
                    continue
                if not (0 <= col + dc < dem_width):
                    continue

                if ridges_data[row + dr, col + dc] == 0:
                    # Nearest point is not a ridge.
                    continue

                # If doing Euclidean search, or if current point is a stream,
                # simply keep the closest ridge.
                if not topological or streams_data[row, col] == 1:
                    ridge_dist[0, row, col] = (dr**2 + dc**2)**0.5 * pixel_size
                    ridge_dist[1, row, col] = row + dr
                    ridge_dist[2, row, col] = col + dc
                    break

                # Check if line from the point to the ridge crosses a stream.
                line_pts = bresenham_line(row, col, row + dr, col + dc)
                crosses_stream = False
                for lr, lc in line_pts:
                    if streams_data[lr, lc] == 1:
                        crosses_stream = True
                        break

                if crosses_stream:
                    # Go to next ridge candidate.
                    continue
                else:
                    # This candidate is valid, exit spiral search.
                    ridge_dist[0, row, col] = (dr**2 + dc**2)**0.5 * pixel_size
                    ridge_dist[1, row, col] = row + dr
                    ridge_dist[2, row, col] = col + dc
                    break

    return ridge_dist


def dist_to_streams(dem: Path, streams: Path, output: Path):
    """
    Calculate distance to nearest stream and its coordinates for
    each DEM pixel.

    For every pixel in the input DEM, computes the Euclidean distance to the
    nearest stream pixel and records that stream pixel's row and column
    indices. DEM nodata areas are preserved in all output bands.

    Parameters
    ----------
    dem : Path
        Path to the Digital Elevation Model (DEM) GeoTIFF. Used for spatial
        reference and nodata masking. DEM and streams rasters must be aligned
        (same CRS, resolution, and extent).
    streams : Path
        Path to the streams raster GeoTIFF. Non-zero values are treated as
        stream pixels. Typically generated by tools like WhiteboxTools
        'extract_streams'. Must be aligned with the DEM.
    output : Path
        Path for the output 3-band GeoTIFF file.

    Returns
    -------
    None
        Writes a 3-band GeoTIFF to disk with the following bands:
        - Band 1: Distance to nearest stream (float32, in map units/meters)
        - Band 2: Row index of nearest stream pixel (float32)
        - Band 3: Column index of nearest stream pixel (float32)
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile.copy()
        dem_data = src.read(1)

        dem_nodata = src.nodata
        nodata_mask = (dem_data == dem_nodata)

    with rasterio.open(streams) as src:
        streams_data = src.read(1)
        stream_mask = streams_data > 0
        transform = src.transform

    if not stream_mask.any():
        raise ValueError("No stream pixels found!")

    # Get pixel spacing in map units for accurate distance calculation.
    pixel_height = abs(transform.e)
    pixel_width = abs(transform.a)

    # Compute distances and indices of nearest stream pixel.
    distances, indices = distance_transform_edt(
        ~stream_mask,
        sampling=(pixel_height, pixel_width),
        return_distances=True,
        return_indices=True
        )

    # indices shape: (2, rows, cols)
    # indices[0] = row index of nearest stream pixel
    # indices[1] = col index of nearest stream pixel
    nearest_rows = indices[0]
    nearest_cols = indices[1]

    # Apply nodata mask to all output bands
    distances[nodata_mask] = dem_nodata
    nearest_rows[nodata_mask] = dem_nodata
    nearest_cols[nodata_mask] = dem_nodata

    # Write output raster with 3 bands (all float32):
    # Band 1: Distance to nearest stream pixel in meters
    # Band 2: Row index of nearest stream pixel
    # Band 3: Column index of nearest stream pixel
    out_profile = dem_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=3,
        compress='deflate'
        )

    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(distances.astype(np.float32), 1)
        dst.write(nearest_rows.astype(np.float32), 2)
        dst.write(nearest_cols.astype(np.float32), 3)

        # Add band descriptions
        dst.set_band_description(1, 'distance_to_stream_meters')
        dst.set_band_description(2, 'nearest_stream_row')
        dst.set_band_description(3, 'nearest_stream_col')


def ratio_dist(dem: Path, dist_stream: Path, dist_ridge: Path, output: Path):
    """
    Calculate the ratio of distances between streams and ridges for
    each pixel in a DEM and save the results.

    For all valid pixels in a digital elevation model (DEM), computes the
    ratio between the distance to the nearest stream (`dist_stream`) and
    the distance to the topologically nearest ridge (`dist_ridge`).

    Parameters
    ----------
    dem : Path
        Path to the input DEM file (Digital Elevation Model) as a GeoTIFF or
        raster dataset.
    dist_stream : Path
        Path to the raster file containing distances to the nearest stream for
        each pixel.
    dist_ridge : Path
        Path to the raster file containing distances to the topologically
        nearest ridge for each pixel.
    output : Path
        Path where the output raster file containing the computed ratio of
        distances will be saved.

    Output
    ------
    - A raster file saved to the `output` path containing the computed ratio
      of distances (`dist_stream` / `dist_ridge`) as a float32 raster.

    Notes
    -----
    - The DEM, `dist_stream`, and `dist_ridge` rasters must have the same
      spatial resolution, extent, and coordinate reference system (CRS).
    - Distance values in the `dist_ridge` raster are adjusted to the pixel
      size of the DEM's pixels to avoid division by zero during ratio
      calculations.
    - Pixels with NoData values in the DEM are excluded from the computation.
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile
        dem_data = src.read(1)
        dem_nodata = src.nodata

        dem_width = src.width
        dem_height = src.height

        dem_transform = src.transform
        assert abs(dem_transform.e) == abs(dem_transform.a)
        pixel_size = abs(dem_transform.e)

    with rasterio.open(dist_stream) as src:
        assert dem_transform == src.transform
        dist_stream_data = src.read(1)
        dist_stream_nodata = src.nodata

    with rasterio.open(dist_ridge) as src:
        assert dem_transform == src.transform
        dist_ridge_data = src.read(1)
        dist_ridge_nodata = src.nodata

    valid_pixels = (
        (dem_data != dem_nodata) &
        (dist_stream_data != dist_stream_nodata) &
        (dist_ridge_data != dist_ridge_nodata)
        )

    # Calculate ratio with minimum threshold to avoid division by zero.
    # Ridge pixels (distance = 0) and very small distances are clamped to
    # pixel_size to prevent numerical instability.
    ratio_dist = np.full(
        (dem_height, dem_width), dem_nodata, dtype=np.float32
        )
    ratio_dist[valid_pixels] = (
        dist_stream_data[valid_pixels] /
        np.maximum(dist_ridge_data[valid_pixels], pixel_size)
        )

    # Write output.
    out_profile = dem_profile.copy()
    out_profile.update(dtype=rasterio.float32, compress='deflate')
    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(ratio_dist, 1)


def height_above_nearest_drainage(dem: Path, dist_stream: Path, output: Path):
    """
    Calculate the height above the nearest drainage (HAND) stream for each
    pixel in a DEM and save the results.

    For each valid pixel in a digital elevation model (DEM), computes the
    height above the nearest stream . The height above streams is
    calculated as the elevation difference between the DEM pixel and the
    elevation of the nearest stream. The result is saved as a raster file.

    Parameters
    ----------
    dem : Path
        Path to the input DEM file (Digital Elevation Model) as a GeoTIFF or
        raster dataset.
    dist_stream : Path
        Path to the raster file containing distances to the nearest stream,
        along with the stream pixel coordinates.
    output : Path
        Path where the output raster file containing the height differences
        will be saved.

    Output
    ------
    - A raster file saved to the `output` path containing the computed height
      differences above the nearest stream as a float32 raster.

    Notes
    -----
    - The DEM and `dist_stream` rasters must have identical spatial resolution,
      extent, and coordinate reference system (CRS).
    - The `dist_stream` raster must contain two additional bands with the row
      and column indices of the nearest stream pixels.
    - Pixels with NoData values in the DEM are excluded from the computation.
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile
        dem_data = src.read(1)
        dem_nodata = src.nodata

        dem_width = src.width
        dem_height = src.height

        dem_transform = src.transform

    with rasterio.open(dist_stream) as src:
        assert dem_transform == src.transform
        stream_rows = src.read(2).astype(int)
        stream_cols = src.read(3).astype(int)
        dist_stream_nodata = int(src.nodata)

    # Valid where both DEM and projected stream are not nodata
    valid_pixels = (
        (dem_data != dem_nodata) &
        (stream_rows != dist_stream_nodata) &
        (stream_cols != dist_stream_nodata)
        )

    # Compute where valid.
    hand = np.full(
        (dem_height, dem_width), dem_nodata, dtype=np.float32
        )
    hand[valid_pixels] = (
        dem_data[valid_pixels] -
        dem_data[stream_rows[valid_pixels], stream_cols[valid_pixels]]
        )

    # Write output.
    out_profile = dem_profile.copy()
    out_profile.update(dtype=rasterio.float32, compress='deflate', count=1)
    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(hand, 1)


def height_below_nearest_ridge(dem: Path, dist_ridge: Path, output: Path):
    """
    Calculate the height below the nearest ridge (HBNR) for each pixel in
    a DEM and save the results.

    For each valid pixel in a digital elevation model (DEM), computes the
    height below the nearest ridge by subtracting the DEM value at each pixel
    from the DEM value at its nearest ridge pixel (as given by dist_ridge).

    Pixels are set to nodata if either the input DEM or nearest ridge indices
    are nodata.

    Parameters
    ----------
    dem : Path
        Path to the input DEM file (GeoTIFF).
    dist_ridge : Path
        Path to a raster with 3 bands:
        - Band 1: Distance to nearest ridge
        - Band 2: Row index of nearest ridge pixel
        - Band 3: Column index of nearest ridge pixel
    output : Path
        Output path for the float32 GeoTIFF containing height below
        nearest ridge.

    Notes
    -----
    - The DEM and `dist_ridge` rasters must have identical spatial resolution,
      extent, and coordinate reference system (CRS).
    - The `dist_ridge` raster must contain two additional bands with the row
      and column indices of the nearest ridge pixels.
    - Pixels with NoData values in the DEM are excluded from the computation.
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile
        dem_data = src.read(1)
        dem_nodata = src.nodata
        dem_transform = src.transform

        dem_width = src.width
        dem_height = src.height

    with rasterio.open(dist_ridge) as src:
        assert dem_transform == src.transform
        ridge_rows = src.read(2).astype(int)
        ridge_cols = src.read(3).astype(int)
        ridge_nodata = int(src.nodata)

    # Valid where both DEM and projected ridge are not nodata
    valid_pixels = (
        (dem_data != dem_nodata) &
        (ridge_rows != ridge_nodata) &
        (ridge_cols != ridge_nodata)
        )

    # Compute where valid.
    hbnr = np.full(
        (dem_height, dem_width), dem_nodata, dtype=np.float32
        )
    hbnr[valid_pixels] = (
        dem_data[ridge_rows[valid_pixels], ridge_cols[valid_pixels]] -
        dem_data[valid_pixels]
        )

    # Write output.
    out_profile = dem_profile.copy()
    out_profile.update(dtype=rasterio.float32, compress='deflate', count=1)
    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(hbnr, 1)


def ratio_stream(dem: Path, hand: Path, dist_stream: Path, output: Path):
    """
    Calculate the ratio of height above the nearest drainage/stream (HAND)
    to the distance from the nearest stream for each pixel in a DEM and save
    the results.

    This function computes the ratio of height above the nearest stream (HAND)
    to the distance to the nearest stream. The height above the nearest stream
    is provided as an input raster (`hand`), and the distance to the nearest
    stream is adjusted to avoid division by zero. The result is saved as an
    output raster.

    Parameters
    ----------
    dem : Path
        Path to the input DEM file (Digital Elevation Model) as a GeoTIFF or
        raster dataset.
    hand : Path
        Path to the raster file containing the height above the nearest stream
        for each pixel.
    dist_stream : Path
        Path to the raster file containing distances to the nearest stream for
        each pixel.
    output : Path
        Path where the output raster file containing the computed ratio will
        be saved.

    Output
    ------
    - A raster file saved to the `output` path containing the computed ratio of
      height above streams to the distance to streams as a float32 raster.

     Notes
     -----
     - The DEM, `hand`, and `dist_stream` rasters must have identical spatial
       resolution, extent, and coordinate reference system (CRS).
     - Distance values in the `dist_stream` raster are adjusted to the pixel
       size of the DEM to avoid division by zero during ratio calculations.
     - Pixels with NoData values in the DEM are excluded from the computation.
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile
        dem_data = src.read(1)
        dem_nodata = src.nodata

        dem_width = src.width
        dem_height = src.height

        dem_transform = src.transform
        assert abs(dem_transform.e) == abs(dem_transform.a)
        pixel_size = abs(dem_transform.e)

    with rasterio.open(hand) as src:
        assert dem_transform == src.transform
        hand_data = src.read(1)
        hand_nodata = src.nodata

    with rasterio.open(dist_stream) as src:
        assert dem_transform == src.transform
        dist_stream_data = src.read(1)
        dist_stream_nodata = src.nodata

    valid_pixels = (
        (dem_data != dem_nodata) &
        (hand_data != hand_nodata) &
        (dist_stream_data != dist_stream_nodata)
        )

    # Calculate ratio with minimum threshold to avoid division by zero.
    # Stream pixels (distance = 0) and very small distances are clamped to
    # pixel_size to prevent numerical instability.
    ratio_stream = np.full(
        (dem_height, dem_width), dem_nodata, dtype=np.float32
        )
    ratio_stream[valid_pixels] = (
        hand_data[valid_pixels] /
        np.maximum(dist_stream_data[valid_pixels], pixel_size)
        )

    # Write output.
    out_profile = dem_profile.copy()
    out_profile.update(dtype=rasterio.float32, compress='deflate', count=1)
    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(ratio_stream, 1)


def local_stats(raster: Path, window: int, output: Path,
                fisher: bool = False):
    """
    Calculate local neighborhood statistics for each pixel in a raster.

    Computes min, max, mean, variance, skewness, and kurtosis within a
    square moving window and saves results as a multi-band raster.

    Parameters
    ----------
    raster : Path
        Path to input raster file.
    window : int
        Size of the square window.  Will be adjusted to nearest odd number.
    output : Path
        Path where the output 6-band raster will be written, where
        1=min, 2=max, 3=mean, 4=variance, 5=skewness, 6=kurtosis.
    fisher : bool, optional
        If True, compute Fisher's (excess) kurtosis. Default is False.
    """

    with rasterio.open(raster) as src:
        profile = src.profile
        data = np.asarray(src.read(1), dtype='float32')

        nodata = float(src.nodata)
        nodata_mask = (data == nodata)

        width = src.width
        height = src.height

    # Make sure we use the nodata values expected in numba functions.
    data[nodata_mask] = NODATA

    # Make sure window is an uneven number.
    window = window + (1 - window % 2)

    results = local_stats_numba(data, window=window, fisher=fisher)
    assert results.shape[0] == 6
    assert results.shape[1] == height
    assert results.shape[2] == width

    # Preserve the 'nodata' mask in the input raster.
    results[:, nodata_mask] = NODATA

    # Replace any remaining 'nan' value by the 'nodata' value.
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        compress='deflate',
        count=6,
        nodata=NODATA
        )
    with rasterio.open(output, 'w', **out_profile) as dst:
        for i in range(6):
            dst.write(results[i, :, :], i + 1)

        # Add band descriptions
        dst.set_band_description(1, 'min')
        dst.set_band_description(2, 'max')
        dst.set_band_description(3, 'mean')
        dst.set_band_description(4, 'var')
        dst.set_band_description(5, 'skew')
        dst.set_band_description(6, 'kurt')


def stream_stats(
        raster: Path,
        dist_stream: Path,
        output: Path,
        fisher: bool = False
        ):
    """
    Calculate statistical summaries along flow paths from each pixel to its
    nearest stream.

    For each valid pixel, computes descriptive statistics (min, max, mean,
    variance, skewness, kurtosis) along the Bresenham line connecting it to
    its nearest stream pixel. Results are saved as a 6-band raster.

    Statistics are computed along straight-line paths (Bresenham algorithm),
    not flow-routed paths.

    raster : Path
        Path to input raster (e.g., DEM, slope, or other terrain attribute).
    dist_stream : Path
        Path to 3-band raster from `dist_to_streams()` containing distance
        to nearest stream (band 1), stream row index (band 2), and stream
        column index (band 3).
    output : Path
        Output path for 6-band GeoTIFF with statistics:
        Band 1: minimum, Band 2: maximum, Band 3: mean,
        Band 4: variance, Band 5: skewness, Band 6: kurtosis.
    fisher : bool, optional
        If True, compute Fisher's excess kurtosis (kurtosis - 3).
        If False, compute Pearson's kurtosis.  Default is False.
    """

    with rasterio.open(raster) as src:
        profile = src.profile
        data = np.asarray(src.read(1), dtype='float32')
        nodata = src.nodata

        width = src.width
        height = src.height

    with rasterio.open(dist_stream) as src:
        stream_rows = src.read(2).astype(int)
        stream_cols = src.read(3).astype(int)
        dist_stream_nodata = int(src.nodata)

    # Identify valid pixels:  must have valid raster data AND valid
    # stream indices
    valid_pixels = (
        (data != nodata) &
        (stream_rows != dist_stream_nodata) &
        (stream_cols != dist_stream_nodata)
        )

    # Make sure we use the nodata values expected in numba functions
    data[~valid_pixels] = NODATA

    results = downslope_stats_numba(data, stream_rows, stream_cols, fisher)
    assert results.shape[0] == 6
    assert results.shape[1] == height
    assert results.shape[2] == width

    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        compress='deflate',
        count=6,
        nodata=NODATA
        )

    with rasterio.open(output, 'w', **out_profile) as dst:
        for i in range(6):
            dst.write(results[i, :, :], i + 1)

        # Add band descriptions
        dst.set_band_description(1, 'min')
        dst.set_band_description(2, 'max')
        dst.set_band_description(3, 'mean')
        dst.set_band_description(4, 'var')
        dst.set_band_description(5, 'skew')
        dst.set_band_description(6, 'kurt')


if __name__ == '__main__':
    from sahel import __datadir__ as datadir
    tile_dir = datadir / "training" / "tiles (overlapped)"

    output = tile_dir / "dist_ridge" / "dist_ridge_tile_017_012.tif"
    output.parent.mkdir(parents=True, exist_ok=True)

    from time import perf_counter
    t0 = perf_counter()
    dist_to_ridges(
        dem=tile_dir / "dem" / "dem_tile_017_012.tif",
        ridges=tile_dir / "ridges" / "ridges_tile_017_012.tif",
        streams=tile_dir / "streams" / "streams_tile_017_012.tif",
        output=output
        )
    t1 = perf_counter()
