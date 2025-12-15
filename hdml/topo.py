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
import ast
from pathlib import Path
from time import perf_counter
from math import sqrt

# ---- Third party imports
import pandas as pd
from numba import njit, prange
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
import whitebox

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.math import bresenham_line, precompute_spiral_offsets
from hdml.localfilters import local_stats_numba, downslope_stats_numba, NODATA
from hdml.tiling import extract_tile, crop_tile


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


def nearest_ridge_coords(dem: Path, ridges: Path, output: Path,
                         streams: Path = None):
    """
    Find the (x, y, z) coordinates and (row, col) indices of the nearest
    ridge pixel for each cell.

    Parameters
    ----------
    dem : Path
        Path to the Digital Elevation Model (DEM) GeoTIFF. Used for spatial
        reference, nodata masking, and elevation values. DEM, ridges, and
        streams rasters must be aligned (same CRS, resolution, and extent).
    ridges : Path
        Path to the ridges raster GeoTIFF. Ridge pixels are identified where
        pixel values are positive (> 0). Can be produced by the
        'extract_ridges' function. Must be aligned with the DEM.
    streams : Path, optional
        Path to the streams raster GeoTIFF. Stream pixels are identified where
        pixel values are positive (> 0). Must be aligned with the DEM.
        When provided, computes distance to the nearest ridge that can be
        reached without crossing a stream (topologically constrained).
        When None, computes simple Euclidean distance to the nearest ridge.
    output : Path
        Path for the output 5-band GeoTIFF file.

    Returns
    -------
    None
        Writes a 5-band GeoTIFF:
          1. Row index of nearest ridge (float32)
          2. Col index of nearest ridge (float32)
          3. x coordinate of nearest ridge (float32)
          4. y coordinate of nearest ridge (float32)
          5. z (elevation) of nearest ridge (float32)

    Notes
    -----
    When topological mode is enabled (by providing `streams`), drainage
    boundaries are respected. For each pixel, only ridges that do not
    require crossing a stream are considered valid.

    Example topology::

        R₁
        |
    ----S----S----
             |
             X
             |
        R₂

    where R = ridge, S = stream, X = current point.

    Even if X is geometrically closer to R₁, the function will select R₂
    as the nearest ridge since the path to R₁ would cross a stream.
    """
    with rasterio.open(dem) as src:
        dem_profile = src.profile.copy()
        dem_data = src.read(1)
        dem_nodata = src.nodata
        nodata_mask = (dem_data == dem_nodata)
        dem_transform = src.transform

        assert abs(dem_transform.e) == abs(dem_transform.a)
        pixel_size = abs(dem_transform.e)

    with rasterio.open(ridges) as src:
        ridges_data = src.read(1)
        ridges_transform = src.transform

    if not np.any(ridges_data > 0):
        raise ValueError("No ridge pixels found!")
    assert dem_transform == ridges_transform

    if streams is None:
        # Computes the Euclidean distance to the nearest ridge,
        # ignoring streams entirely.
        results = _dist_to_ridge_euclidean(
            dem_data, ridges_data, dem_nodata, pixel_size
            )
    else:
        # Computes distance to the nearest ridge that can be reached
        # without crossing a stream (topologically constrained)
        with rasterio.open(streams) as src:
            streams_data = src.read(1)
            streams_transform = src.transform

        assert dem_transform == streams_transform

        results = _dist_to_ridges_topological(
            dem_data, streams_data, ridges_data,
            nodata=dem_nodata, pixel_size=pixel_size,
            topological=True
            )

    nearest_rows = results[1]
    nearest_cols = results[2]

    # x and y coordinates of nearest ridge pixel.
    xs, ys = rasterio.transform.xy(
        ridges_transform,
        nearest_rows.ravel(), nearest_cols.ravel(),
        offset='center'
        )
    xs = np.array(xs, dtype=np.float32).reshape(nearest_rows.shape)
    ys = np.array(ys, dtype=np.float32).reshape(nearest_rows.shape)

    # z (elevation) value at the location of the nearest ridge pixel.
    z_ridge = dem_data[nearest_rows, nearest_cols].astype(np.float32)

    # Apply nodata mask to all output bands
    nearest_rows[nodata_mask] = dem_nodata
    nearest_cols[nodata_mask] = dem_nodata
    xs[nodata_mask] = dem_nodata
    ys[nodata_mask] = dem_nodata
    z_ridge[nodata_mask] = dem_nodata

    # Write output raster with 5 bands (all float32):
    out_profile = dem_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=5,
        compress='deflate'
        )

    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(nearest_rows.astype(np.float32), 1)
        dst.write(nearest_cols.astype(np.float32), 2)
        dst.write(xs, 3)
        dst.write(ys, 4)
        dst.write(z_ridge, 5)

        # Add band descriptions
        dst.set_band_description(1, 'nearest_ridge_row')
        dst.set_band_description(2, 'nearest_ridge_col')
        dst.set_band_description(3, 'nearest_ridge_x')
        dst.set_band_description(4, 'nearest_ridge_y')
        dst.set_band_description(5, 'nearest_ridge_z')


def _dist_to_ridge_euclidean(
        dem_data: np.array,
        ridges_data: np.ndarray,
        nodata: Any,
        pixel_size: float,
        ):

    ridges_mask = ridges_data > 0

    if not ridges_mask.any():
        raise ValueError("No ridge pixels found!")

    # Compute distances and indices of nearest ridge pixel.
    distances, indices = distance_transform_edt(
        ~ridges_mask,
        sampling=(pixel_size, pixel_size),
        return_distances=True,
        return_indices=True
        )

    ridge_dist = np.empty((3, *dem_data.shape), dtype=np.float32)
    ridge_dist[0] = distances
    ridge_dist[1] = indices[0]  # row index of nearest ridge pixel
    ridge_dist[2] = indices[1]  # col index of nearest ridge pixel

    # Enforce nodata values in the DEM.
    nodata_mask = (dem_data == nodata)
    ridge_dist[:, nodata_mask] = nodata

    return ridge_dist


@njit(parallel=True, fastmath=True)
def _dist_to_ridges_topological(
        dem_data: np.ndarray,
        streams_data: np.ndarray,
        ridges_data: np.ndarray,
        nodata: Any,
        pixel_size: float,
        topological: bool = False
        ) -> np.ndarray:

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

                if ridges_data[row + dr, col + dc] != 1:
                    # Nearest point is not a ridge.
                    continue

                # If doing Euclidean search, or if current point is a stream,
                # simply keep the closest ridge.
                if not topological or streams_data[row, col] == 1:
                    ridge_dist[0, row, col] = sqrt(dist_sqrt) * pixel_size
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


def nearest_stream_coords(dem: Path, streams: Path, output: Path):
    """
    Find the (x, y, z) coordinates and (row, col) indices of the nearest
    stream pixel for each cell of the DEM.

    Parameters
    ----------
    dem : Path
        Path to the Digital Elevation Model (DEM) GeoTIFF. Used for spatial
        reference, nodata masking, and elevation values. DEM and streams
        rasters must be aligned (same CRS, resolution, and extent).
    streams : Path
        Path to the streams raster GeoTIFF. Stream pixels are identified where
        pixel values are positive (> 0). Must be aligned with the DEM.
    output : Path
        Path for the output 5-band GeoTIFF file.

    Returns
    -------
    None
        Writes a 5-band GeoTIFF:
          1. Row index of nearest stream (float32)
          2. Col index of nearest stream (float32)
          3. x coordinate of nearest stream (float32)
          4. y coordinate of nearest stream (float32)
          5. z (elevation) of nearest stream (float32)
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
    indices = distance_transform_edt(
        ~stream_mask,
        sampling=(pixel_height, pixel_width),
        return_distances=False,
        return_indices=True
        )

    # indices shape: (2, rows, cols)
    # indices[0] = row index of nearest stream pixel
    # indices[1] = col index of nearest stream pixel
    nearest_rows = indices[0]
    nearest_cols = indices[1]

    # x and y coordinates of nearest stream pixel.
    xs, ys = rasterio.transform.xy(
        transform, nearest_rows.ravel(), nearest_cols.ravel(), offset='center'
        )
    xs = np.array(xs, dtype=np.float32).reshape(nearest_rows.shape)
    ys = np.array(ys, dtype=np.float32).reshape(nearest_rows.shape)

    # z (elevation) value at the location of the nearest stream pixel
    z_stream = dem_data[nearest_rows, nearest_cols].astype(np.float32)

    # Apply nodata mask to all output bands
    nearest_rows[nodata_mask] = dem_nodata
    nearest_cols[nodata_mask] = dem_nodata
    xs[nodata_mask] = dem_nodata
    ys[nodata_mask] = dem_nodata
    z_stream[nodata_mask] = dem_nodata

    # Write output raster with 5 bands (all float32):
    out_profile = dem_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=5,
        compress='deflate'
        )

    with rasterio.open(output, 'w', **out_profile) as dst:
        dst.write(nearest_rows.astype(np.float32), 1)
        dst.write(nearest_cols.astype(np.float32), 2)
        dst.write(xs, 3)
        dst.write(ys, 4)
        dst.write(z_stream, 5)

        # Add band descriptions
        dst.set_band_description(1, 'nearest_stream_row')
        dst.set_band_description(2, 'nearest_stream_col')
        dst.set_band_description(3, 'nearest_stream_x')
        dst.set_band_description(4, 'nearest_stream_y')
        dst.set_band_description(5, 'nearest_stream_z')


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
        Path to 5-band raster from `nearest_stream_coords()` where the
        row index and col index of nearest stream is stored in Band 1 & 2..
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
        stream_rows = src.read(1).astype(int)
        stream_cols = src.read(2).astype(int)
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


def generate_topo_features_for_tile(
        tile_bbox_data: pd.Series,
        dem_path: str | Path,
        crop_tile_dir: str | Path,
        ovlp_tile_dir: str | Path,
        print_affix: str = None,
        extract_streams_treshold: int = 1500,
        gaussian_filter_sigma: int = 1,
        ridge_size: int = 30,
        long_stats_window: int = 41,
        short_stats_window: int = 7,
        overwrite: bool = False,
        ):
    """
    Generate all topo-derived features for the ML model for the
    specified tile.
    """
    if print_affix != '':
        print_affix += ' '

    crop_tile_dir = Path(crop_tile_dir)
    if not crop_tile_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {crop_tile_dir}")
    if not crop_tile_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {crop_tile_dir}")

    ovlp_tile_dir = Path(ovlp_tile_dir)
    if not ovlp_tile_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {ovlp_tile_dir}")
    if not ovlp_tile_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {ovlp_tile_dir}")

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(
            "The path you provided for 'dem_path' is not valid. "
            "Check that you provided the right path or "
            "run 'process_dem_data.py' to produce a valid dem mosaic."
            )

    tile_index = tile_bbox_data.tile_index
    ty, tx = ast.literal_eval(tile_index)

    FEATURES = [
        'dem', 'dem_smooth', 'dem_cond', 'slope', 'curvature',
        'flow_accum', 'streams', 'nearest_stream_coords',
        'geomorphons', 'ridges', 'dist_top',
        'long_hessian_stats', 'long_grad_stats',
        'short_grad_stats',
        'stream_grad_stats', 'stream_hessian_stats'
        ]

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False

    core_bbox_pixels = ast.literal_eval(tile_bbox_data.core_bbox_pixels)
    ovlp_bbox_pixels = ast.literal_eval(tile_bbox_data.ovlp_bbox_pixels)

    crop_kwargs = {
        'crop_x_offset': tile_bbox_data.crop_x_offset,
        'crop_y_offset': tile_bbox_data.crop_y_offset,
        'width': core_bbox_pixels[2],
        'height': core_bbox_pixels[3],
        'overwrite': False
        }

    tile_name_template = '{name}_tile_{ty:03d}_{tx:03d}.tif'

    # Helper to process a feature.
    def process_feature(name, func, **kwargs):
        tile_name = tile_name_template.format(name=name, ty=ty, tx=tx)

        overlap_tile_path = ovlp_tile_dir / name / tile_name
        overlap_tile_path.parent.mkdir(parents=True, exist_ok=True)

        if not overlap_tile_path.exists() or overwrite:
            func(output=str(overlap_tile_path), **kwargs)

            cropped_tile_path = crop_tile_dir / name / tile_name
            cropped_tile_path.parent.mkdir(parents=True, exist_ok=True)

            crop_tile(overlap_tile_path, cropped_tile_path, **crop_kwargs)

        return overlap_tile_path

    all_processed = True
    tile_paths = {}
    for name in FEATURES:
        tile_name = tile_name_template.format(name=name, ty=ty, tx=tx)
        tile_paths[name] = ovlp_tile_dir / name / tile_name

        if not(crop_tile_dir / name / tile_name).exists():
            all_processed = False

    if all_processed is True:
        print(f"{print_affix}Features already calculated "
              f"for tile {tile_index}.")
        return

    func_kwargs = {
        'dem': {
            'func': lambda output, **kwargs: extract_tile(
                output_tile=output, **kwargs),
            'kwargs': {'input_raster': dem_path,
                       'bbox': ovlp_bbox_pixels,
                       'overwrite': overwrite,
                       'output_dtype': 'Float32'}
            },
        'dem_smooth': {
            'func': wbt.gaussian_filter,
            'kwargs': {'i': tile_paths['dem'],
                       'sigma': 1.0}
            },
        'dem_cond': {
            'func': wbt.fill_depressions,
            'kwargs': {'dem': tile_paths['dem_smooth']}
            },
        'slope': {
            'func': wbt.slope,
            'kwargs': {'dem': tile_paths['dem_cond']}
            },
        'curvature': {
            'func': wbt.profile_curvature,
            'kwargs': {'dem': tile_paths['dem_cond']}
            },
        'flow_accum': {
            'func': wbt.d8_flow_accumulation,
            'kwargs': {'i': tile_paths['dem_cond'],
                       'out_type': 'cells'}
            },
        'streams': {
            'func': wbt.extract_streams,
            'kwargs': {'flow_accum': tile_paths['flow_accum'],
                       'threshold': extract_streams_treshold}
            },
        'nearest_stream_coords': {
            'func': nearest_stream_coords,
            'kwargs': {'dem': tile_paths['dem_cond'],
                       'streams': tile_paths['streams']}
            },
        'geomorphons': {
            'func': wbt.geomorphons,
            'kwargs': {'dem': tile_paths['dem_cond'],
                       'search': 100,
                       'threshold': 1.0,
                       'fdist': 0,
                       'skip': 0,
                       'forms': True,
                       'residuals': True
                       }
            },
        'ridges': {
            'func': extract_ridges,
            'kwargs': {'geomorphons': tile_paths['geomorphons'],
                       'ridge_size': ridge_size,
                       # 'flow_acc': tile_paths['flow_accum'],
                       # 'max_flow_acc': 2
                       }

            },
        'nearest_ridge_coords': {
            'func': nearest_ridge_coords,
            'kwargs': {'dem': tile_paths['dem_cond'],
                       'ridges': tile_paths['ridges']}
            },
        'long_hessian_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['curvature'],
                       'window': long_stats_window}
            },
        'long_grad_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'window': long_stats_window}
            },
        'short_grad_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'window': short_stats_window}
            },
        'stream_grad_stats': {
            'func': stream_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'dist_stream': tile_paths['nearest_stream_coords'],
                       'fisher': False}
            },
        'stream_hessian_stats': {
            'func': stream_stats,
            'kwargs': {'raster': tile_paths['curvature'],
                       'dist_stream': tile_paths['nearest_stream_coords'],
                       'fisher': False}
            },
        }

    # max_short_distance = 7 pixels == 210 m -> halfwidth de 105 m
    # max_long_distance = 41 = 1230 m -> halfwidth = 615 m

    ttot0 = perf_counter()
    for name in FEATURES:
        t0 = perf_counter()
        print(f"{print_affix}Computing {name} for tile {tile_index}...",
              end='')
        func = func_kwargs[name]['func']
        kwargs = func_kwargs[name]['kwargs']
        process_feature(name, func, **kwargs)
        t1 = perf_counter()
        print(f' done in {round(t1 - t0):0.0f} sec')
    ttot1 = perf_counter()
    print(f"{print_affix} All topo feature for tile {tile_index} "
          f"computed in {round(ttot1 - ttot0):0.0f} sec")
