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
import itertools
from pathlib import Path
import math
import os
import os.path as osp
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Third party imports
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
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
        zip_path: str, tif_path: str, compress: str = 'LZW'):
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

    vsi_path = f'/vsizip/{zip_path}/{swb_filename}'
    with rasterio.open(vsi_path) as src:
        swt_data = src.read(1).astype(hgt_data.dtype)

    # Write to GeoTIFF.
    profile.update(driver='GTiff', count=2, dtype=hgt_data.dtype)
    if compress is not None:
        profile.update(compress=compress)

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(hgt_data, 1)
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


def generate_tiles_bbox(
        input_raster: Path,
        tile_size: int = 5000,
        overlap: int = 100,
        ) -> dict:
    """
    Generate bounding box information for tiling a large raster with overlap.

    Pre-computes the pixel coordinates for a grid of tiles covering the input
    raster, including both the core (non-overlapping) tile extents and the
    overlapped extents needed for accurate edge processing.

    Parameters
    ----------
    input_raster : Path
        Path to the input raster file to be tiled.
    tile_size : int, optional
        Size of each tile in pixels (width and height of the core tile area,
        excluding overlap). Default is 5000.
    overlap : int, optional
        Number of pixels to overlap between adjacent tiles. This overlap
        ensures that edge effects are minimized when processing tiles
        independently. Default is 100.

    Returns
    -------
    dict
        Dictionary mapping tile indices (ty, tx) to tile geometry information.
        Each entry contains:
        - 'core': [x_start, y_start, width, height] - Core tile extent without
          overlap, in pixel coordinates relative to the input raster. Used for
          the final mosaic.
        - 'overlap': [x_start, y_start, width, height] - Extended tile extent
          with overlap, in pixel coordinates. Used for processing to ensure
          accurate calculations at tile edges.
        - 'crop_x_offset': int - Number of pixels to skip from the left edge
          of the overlapped tile to extract the core tile.
        - 'crop_y_offset': int - Number of pixels to skip from the top edge
          of the overlapped tile to extract the core tile.

        Note: Tiles smaller than 10x10 pixels are excluded from the output.
    """

    # Open DEM to get dimensions.
    ds = gdal.Open(str(input_raster))
    if ds is None:
        raise ValueError(f"Cannot open input raster: {input_raster}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    # Calculate number of tiles
    n_tiles_x = math.ceil(width / tile_size)
    n_tiles_y = math.ceil(height / tile_size)

    tiles_bbox_data = {}
    for ty, tx in itertools.product(range(n_tiles_y), range(n_tiles_x)):
        # Calculate the CORE tile extent WITHOUT overlap (for final mosaic).
        x_start = tx * tile_size
        y_start = ty * tile_size
        x_end = min(width, (tx + 1) * tile_size)
        y_end = min(height, (ty + 1) * tile_size)

        w = x_end - x_start
        h = y_end - y_start

        # Calculate tile extent WITH overlap for processing
        x_start_ovlp = max(0, tx * tile_size - overlap)
        y_start_ovlp = max(0, ty * tile_size - overlap)
        x_end_ovlp = min(width, (tx + 1) * tile_size + overlap)
        y_end_ovlp = min(height, (ty + 1) * tile_size + overlap)

        w_ovlp = x_end_ovlp - x_start_ovlp
        h_ovlp = y_end_ovlp - y_start_ovlp

        # Skip tiny edge tiles. This is ok since the boundary that we used to
        # clip the DEM had a 100km buffer (~1110 pixels).
        if w_ovlp < 10 or h_ovlp < 10:
            continue

        tiles_bbox_data[(ty, tx)] = {
            'core': [x_start, y_start, w, h],
            'overlap': [x_start_ovlp, y_start_ovlp, w_ovlp, h_ovlp],
            'crop_x_offset': x_start - x_start_ovlp,
            'crop_y_offset': y_start - y_start_ovlp
            }

    return tiles_bbox_data


def extract_tile(
        input_raster: Path,
        output_tile: Path,
        bbox: list,
        overwrite: bool = False
        ) -> Path:
    """
    Extract a tile from a raster using pixel coordinates.

    Parameters
    ----------
    input_raster : Path
        Path to input raster file.
    output_tile : Path
        Path where the extracted tile will be saved.
    bbox : list
        Bounding box as [x_start, y_start, width, height] in pixel coordinates.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.

    Returns
    -------
    Path
        Path to the extracted tile.
    """
    if not output_tile.exists() or overwrite:
        gdal.Translate(
            str(output_tile),
            str(input_raster),
            srcWin=bbox,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
            )

    return output_tile


def crop_tile(
        input_tile: Path,
        output_tile: Path,
        crop_x_offset: int,
        crop_y_offset: int,
        width: int,
        height: int,
        overwrite: bool = False
        ) -> Path:
    """
    Crop overlap margins from a processed tile.

    Parameters
    ----------
    input_tile : Path
        Path to the tile with overlap.
    output_tile : Path
        Path where the cropped tile will be saved.
    crop_x_offset : int
        Number of pixels to skip from left edge.
    crop_y_offset : int
        Number of pixels to skip from top edge.
    width : int
        Width of the cropped tile in pixels.
    height : int
        Height of the cropped tile in pixels.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.

    Returns
    -------
    Path
        Path to the cropped tile.
    """
    if not output_tile.exists() or overwrite:
        gdal.Translate(
            str(output_tile),
            str(input_tile),
            srcWin=[crop_x_offset, crop_y_offset, width, height],
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
            )

    return output_tile


def mosaic_tiles(
        tile_paths: list,
        output_raster: Path,
        overwrite: bool = False,
        cleanup_tiles: bool = False
        ) -> Path:
    """
    Mosaic tiles into a single raster using GDAL VRT and translate.

    Parameters
    ----------
    tile_paths : list
        List of Path objects pointing to tiles to mosaic.
    output_raster : Path
        Path where the mosaiced raster will be saved.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.
    cleanup_tiles : bool, optional
        Whether to delete tiles after mosaicing. Default is False.

    Returns
    -------
    Path
        Path to the output mosaiced raster.
    """
    if output_raster.exists() and not overwrite:
        return output_raster

    # Build VRT.
    vrt_path = output_raster.with_suffix('.vrt')
    gdal.BuildVRT(str(vrt_path), tile_paths)

    # Translate to GeoTIFF.
    gdal.Translate(
        str(output_raster),
        str(vrt_path),
        creationOptions=[
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=YES',
            'NUM_THREADS=ALL_CPUS'
        ]
    )

    # Cleanup.
    vrt_path.unlink(missing_ok=True)

    if cleanup_tiles:
        for tile in tile_paths:
            tile.unlink(missing_ok=True)

    return output_raster


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
