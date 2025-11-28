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
from typing import Any

# ---- Standard imports
from pathlib import Path

# ---- Third party imports
from numba import njit
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects

# ---- Local imports
from sahel.math import bresenham_line, precompute_spiral_offsets


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


def dist_to_ridges(dem: Path, ridges: Path, streams: Path, output: Path):
    """
    Calculate distance to the topologically nearest ridge for each pixel
    in a DEM.

    For each pixel in the input DEM, computes the Euclidean distance to the
    nearest *topologically reachable* ridge pixel—i.e., the closest ridge that
    can be reached without crossing a stream pixel. Outputs both the distance
    (in map units/meters) and the coordinates of the nearest ridge pixel for
    every non-nodata pixel. Nodata areas in the DEM are preserved in all
    output bands.

    The drainage boundaries are respected: for each pixel, only ridges that do
    not require crossing a stream are considered valid when determining
    the topological distance.

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

    results = dist_to_ridges_numba(
        dem_data, streams_data, ridges_data,
        nodata=dem_nodata, pixel_size=pixel_size
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


@njit
def dist_to_ridges_numba(
        dem_data: np.ndarray,
        streams_data: np.ndarray,
        ridges_data: np.ndarray,
        nodata: Any,
        pixel_size: float
        ) -> np.ndarray:

    dem_height, dem_width = dem_data.shape
    spiral_offsets = precompute_spiral_offsets(max(dem_width, dem_height))
    ridge_dist = np.full((3, *dem_data.shape), nodata, dtype=np.float32)

    n_rows, n_cols = dem_data.shape
    for row in range(n_rows):
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

                # If point is a stream, we simply keep the closest ridge.
                if streams_data[row, col] == 1:
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
        nodata_mask = dem_data == dem_nodata

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

    # Set nodata value where dem is null.
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


def height_above_stream(output: Path, dem: Path, streams: Path):
    # Read DEM data and metadata.
    with rasterio.open(dem) as src:
        dem_data = src.read(1).astype('float32')
        meta = src.meta.copy()
        nodata = src.nodata

    # Replace nodata by nan.
    dem_data[dem_data == nodata] = np.nan

    # Read streams (0 = no stream, >0 = stream).
    with rasterio.open(streams) as src:
        streams_data = src.read(1)

    # Create binary stream mask (1 = stream, 0 = non-stream).
    stream_mask = streams_data > 0

    # Get elevation of nearest stream for each pixel
    indices = distance_transform_edt(
        ~stream_mask,
        return_distances=False,
        return_indices=True)

    rows = indices[0]
    cols = indices[1]

    nearest_stream_elevation = dem_data[rows, cols]

    # Calculate elevation above nearest stream
    elevation_above_stream = dem_data - nearest_stream_elevation

    # Fill nan value with nodata.
    elevation_above_stream[np.isnan(elevation_above_stream)] = nodata

    # Save elevation above stream
    meta.update(dtype='float32')
    with rasterio.open(output, 'w', **meta) as dst:
        dst.write(elevation_above_stream.astype('float32'), 1)
