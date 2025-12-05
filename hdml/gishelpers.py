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

"""GIS data manipulation utilities."""

# ---- Standard imports
from pathlib import Path
import os
import os.path as osp
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Third party imports
import shapely
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from osgeo import gdal

gdal.UseExceptions()


def get_dem_filepaths(dirname: str) -> list:
    """
    Return a list of filepaths to all DEM (.tif) files for the
    specified directory.

    Parameters
    ----------
    directory : str
        The directory where to retrieve DEM filepaths.

    Returns
    -------
    list of str
        List of filepaths to the DEM raster files contained in the provided
        directory.
    """
    if not osp.exists(dirname):
        return []
    else:
        return [
            osp.join(dirname, f) for f in
            os.listdir(dirname) if f.endswith('.tif')
            ]


def convert_hgt_to_geotiff(
        zip_path: str, tif_path: str, compress: str = 'LZW',
        include_swb: bool = False):
    """
    Extracts the .hgt file from a zip archive and converts it to a GeoTIFF.

    Parameters
    ----------
    zip_path : str
        Path to the input ZIP archive containing the .hgt DEM tile.
    tif_path : str
        Path where to save the GeoTiff file.
    compress : str or None, optional
        Compression method to use for the output GeoTIFF (e.g., 'zstd',
        'DEFLATE', 'LZW'). If None, no compression is applied.
        Default is 'zstd'.
    """
    # Find the .hgt file name inside the zip archive.
    with zipfile.ZipFile(zip_path, 'r') as zf:
        hgt_filename = None
        swb_filename = None
        for fname in zf.namelist():
            if fname.lower().endswith('.hgt'):
                hgt_filename = fname
            elif fname.lower().endswith('.swb'):
                swb_filename = fname

        if hgt_filename is None:
            raise FileNotFoundError(f"No .hgt file found in '{zip_path}'.")
        if swb_filename is None:
            raise FileNotFoundError(f"No .swt file found in '{zip_path}'.")

    # Open the .hgt and .swb file using rasterio (via GDAL VFS).
    vsi_path = f'/vsizip/{zip_path}/{hgt_filename}'
    with rasterio.open(vsi_path) as src:
        profile = src.profile
        hgt_data = src.read(1)

    if include_swb:
        vsi_path = f'/vsizip/{zip_path}/{swb_filename}'
        with rasterio.open(vsi_path) as src:
            swt_data = src.read(1).astype(hgt_data.dtype)

    # Write to GeoTIFF.
    profile.update(
        driver='GTiff',
        count=2 if include_swb else 1,
        dtype=hgt_data.dtype
        )
    if compress is not None:
        profile.update(compress=compress)

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(hgt_data, 1)
        if include_swb:
            dst.write(swt_data, 2)


def multi_convert_hgt_to_geotiff(zip_paths: list, tif_paths: list):
    """
    Converts multiple NASADEM .hgt ZIP archives to GeoTIFF files in parallel.

    For each ZIP archive in `zip_paths`, this function extracts the .hgt
    DEM tile and its corresponding surface water mask (.swb), then writes
    both as bands 1 and 2 of a new GeoTIFF file at the specified path
    in `tif_paths`.

    Only tiles for which the input ZIP exists and the output TIFF does
    not exist will be processed.

    Parameters
    ----------
    zip_paths : list of str
        List of paths to input ZIP archives containing
        NASADEM .hgt and .swb files.
    tif_paths : list of str
        List of output paths for the converted GeoTIFF files. Each entry in
        `tif_paths` should correspond to the respective entry in `zip_paths`.
    """

    def process_tile(zip_path: str, tif_path: str):
        convert_hgt_to_geotiff(zip_path, tif_path)
        return zip_path

    count = 0
    progress = 0
    with ThreadPoolExecutor() as executor:

        futures = []
        for zip_path, tif_path in zip(zip_paths, tif_paths):
            if not osp.exists(zip_path) or osp.exists(tif_path):
                continue

            futures.append(executor.submit(
                process_tile, zip_path, tif_path
                ))

            count += 1

        if len(futures) == 0:
            print(f"All {len(zip_paths)} tiles were alreaty converted to tif.")
        else:
            print(f"Converting {count} tiles (out of {len(zip_paths)}) "
                  f"to GeoTIFF...")

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    progress += 1
                    print(f'Converted {progress} of {count}.')
            except Exception as exc:
                print(f'Error: {exc}')


def create_pyramid_overview(
        geotif_path: str, overview_levels: list[int] = None,
        overwrite: bool = False):
    """
    Create pyramid overviews for a GeoTIFF at specified levels using GDAL.

    Pyramid overviews are reduced-resolution versions of the raster that
    allow GIS software to display and zoom large rasters much more quickly.
    They are essential for efficient visualization and navigation of
    large datasets.

    Parameters
    ----------
    geotif_path : str
        Path to the GeoTIFF file for which to build overviews.
    overview_levels : list of int, optional
        List of overview (downsampling) factors to generate overviews at.
        Each level represents the reduction factor relative to the original
        resolution. If None, defaults to [2, 4, 8, 16].
    """
    geotif_path = Path(geotif_path)
    if geotif_path.with_suffix('.tif.ovr').exists() and not overwrite:
        return

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16]

    # Open in update mode so that overviews can be written
    ds = gdal.Open(geotif_path, gdal.GA_ReadOnly)

    ds.BuildOverviews("average", overview_levels)
    ds = None


def resample_raster(input_path, output_path, target_res=500,
                    resample_method='average'):
    """
    Resample a raster to a target resolution using GDAL.

    Parameters
    ----------
    input_path : str
        Path to input raster (e.g., DEM).
    output_path : str
        Path to output resampled raster.
    target_res : float
        Target resolution in map units (meters for projected CRS).
    resample_method : str
        GDAL resampling method: 'average', 'bilinear', 'nearest', 'cubic', etc.
        For aggregation, 'average' is recommended.
    """
    # Build gdalwarp command
    gdal.Warp(
        output_path,
        input_path,
        xRes=target_res,
        yRes=target_res,
        resampleAlg=resample_method,
        format='GTiff'
        )


def rasterize_streams(
        vector_path: Path,
        template_raster: Path,
        output_raster: Path,
        burn_value: int = 1,
        background_value: int = 0,
        attribute: str = None,
        all_touched: bool = False,
        overwrite: bool = False
        ) -> Path:
    """
    Rasterize a vector stream network to match a DEM grid.

    Parameters
    ----------
    vector_path : Path
        Path to vector file (shapefile, geojson, etc.)
    template_raster : Path
        Path to DEM or other raster to match (grid, extent, CRS)
    output_raster : Path
        Path for output rasterized stream network
    burn_value : int, optional
        Value to burn for streams. Default is 1.
    background_value : int, optional
        Value for non-stream pixels. Default is 0.
    attribute : str, optional
        Vector attribute to use for burn values instead of fixed burn_value.
        Example: 'stream_order' to burn Strahler order values.
    all_touched : bool, optional
        If True, all pixels touched by lines are burned.
        If False, only pixels whose center is covered. Default is False.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.

    Returns
    -------
    Path
        Path to output raster
    """
    if output_raster.exists() and not overwrite:
        return output_raster

    # Get grid parameters and nodata mask from template
    with rasterio.open(template_raster) as src:
        meta = src.meta.copy()
        template_data = src.read(1)
        template_nodata = src.nodata
        template_crs = src.crs

    # Create nodata mask from template
    if template_nodata is not None:
        nodata_mask = (template_data == template_nodata)
    else:
        nodata_mask = None

    # Update metadata for output.
    meta.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw',
        'tiled': True,
        'nodata': 255
        })

    # Read streams vector.
    gdf = gpd.read_file(vector_path)
    if gdf.crs != template_crs:
        gdf = gdf.to_crs(template_crs)

    invalid_count = (~gdf.geometry.is_valid).sum()
    empty_count = gdf.geometry.is_empty.sum()

    if invalid_count > 0 or empty_count > 0:
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    # Prepare shapes
    if attribute:
        shapes = ((geom, value) for geom, value in
                  zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = ((geom, burn_value) for geom in gdf.geometry)

    # Rasterize
    burned = rasterize(
        shapes=shapes,
        out_shape=(meta['height'], meta['width']),
        transform=meta['transform'],
        fill=background_value,
        all_touched=all_touched,
        dtype='uint8'
        )

    # Apply nodata mask from template
    if nodata_mask is not None:
        burned[nodata_mask] = 255

    # Write output
    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(burned, 1)

    return output_raster


def extract_zonal_means(
        raster_path: Path,
        geometries: list[shapely.Geometry],
        ) -> np.ndarray:
    """
    Extract mean raster values for a list of geometries.

    Computes the spatial mean of raster values (e.g., NDVI, precipitation)
    within each provided geometry (e.g., watershed polygons, administrative
    boundaries). Nodata values are excluded from the mean calculation.
    Geometries that do not intersect the raster or contain only nodata will
    return NaN.

    This implementation keeps the raster file open for the entire loop,
    which is highly efficient for VRT files and large numbers of geometries.

    Parameters
    ----------
    raster_path : Path
        Path to the raster file (GeoTIFF, VRT, etc.).
    geometries : list of shapely.Geometry
        List of geometries (polygons, multipolygons) for which to extract
        raster values.  Must be in the same CRS as the raster.

    Returns
    -------
    np.ndarray
        Array of mean values, one per geometry.
    """
    n_geoms = len(geometries)
    mean_values = np.empty(n_geoms, dtype=np.float32)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata

        for i, geom in enumerate(geometries):
            try:
                # Mask raster with the geometry.
                data, transform = mask(
                    src, [geom], crop=True, filled=True, nodata=nodata
                )
                array = data[0]  # Get first band

                # Compute mean, excluding nodata
                if nodata is not None:
                    valid_pixels = array[array != nodata]
                else:
                    valid_pixels = array

                if valid_pixels.size > 0:
                    mean_values[i] = np.mean(valid_pixels)
                else:
                    mean_values[i] = np.nan

            except ValueError:
                # Geometry doesn't intersect raster.
                mean_values[i] = np.nan

    return mean_values
