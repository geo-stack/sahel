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
import os
import os.path as osp
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- Third party imports
import numpy as np
import geopandas as gpd
import pyproj
import rasterio
from shapely.geometry import box
from shapely.ops import transform
from rasterio.windows import Window, transform as window_transform
from rasterio.mask import mask
from rasterio.features import geometry_mask


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
        zip_path: str, tif_path: str, compress: str = 'zstd'):
    """
    Extracts the .hgt file from a zip archive and converts it to a GeoTIFF.

    Parameters
    ----------
    zip_path : str
        Path to the input ZIP archive containing the .hgt DEM tile.
    dest_dir : str, optional
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


def tile_in_boundary(
        tile_fpath: str, boundary_fpath: str, mode="intersects"
        ) -> bool:
    """
    Check if a GeoTIFF tile is within or intersects a boundary from
    a GeoJSON file.

    Parameters
    ----------
    tile_path : str
        Path to the GeoTIFF tile.
    boundary_geojson_path : str
        Path to the GeoJSON file with the boundary.
    mode : str, optional
        "intersects" (default): returns True if tile crosses or is
            within boundary.
        "within": returns True only if tile is fully within the boundary.

    Returns
    -------
    result : bool
        True if the tile matches the test, False otherwise.
    """
    # Load boundary.
    boundary_gdf = gpd.read_file(boundary_fpath)
    boundary_geom = boundary_gdf.union_all()
    boundary_crs = boundary_gdf.crs

    # Load tile bounds and CRS.
    with rasterio.open(tile_fpath) as src:
        bounds = src.bounds
        tile_crs = src.crs
        tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Reproject tile_box if CRS doesn't match.
    if boundary_crs != tile_crs:
        project = pyproj.Transformer.from_crs(
            tile_crs, boundary_crs, always_xy=True).transform
        tile_box = transform(project, tile_box)

    if mode == "intersects":
        return tile_box.intersects(boundary_geom)
    elif mode == "within":
        return tile_box.within(boundary_geom)
    else:
        raise ValueError(
            f"'mode' must be 'intersects' or 'within', but got {mode}."
            )


def extract_vertical_band(
        input_path: str, output_path: str,
        start_col: int, band_width: int, overlap: int = 0,
        boundary_geojson: str = None):
    """
    Extracts a vertical band from a raster file with optional overlap, and
    optionally masks pixels outside a boundary geometry.

    If a boundary GeoJSON is provided, only pixels inside the boundary are
    retained; others are set to NaN.

    Parameters
    ----------
    input_path : str or Path
        Path to the input raster.
    output_path : str or Path
        Path to save the extracted band.
    start_col : int
        Starting column index (0-based).
    band_width : int
        Width of the band (number of columns).
    overlap : int, optional
        Overlap (in columns) to include before and after the band.
    boundary_geojson : str or Path, optional
        Path to a GeoJSON file containing boundary geometry for masking.
        If None, no masking is performed.
    """
    with rasterio.open(input_path) as src:
        # Calculate window bounds with overlap, making sure not to
        # exceed raster size.
        left = max(start_col - overlap, 0)
        right = min(start_col + band_width + overlap, src.width)
        width = right - left
        nodata = src.nodata

        window = Window(
            col_off=left, row_off=0,
            width=width, height=src.height
            )

        profile = src.profile.copy()
        profile.update({
            'width': width,
            'height': src.height,
            'transform': rasterio.windows.transform(window, src.transform),
            'nodata': nodata
            })

        # Read data in the window
        data = src.read(window=window)

        # Mask the window if a boundary geometry is provided.
        if boundary_geojson:
            gdf = gpd.read_file(boundary_geojson)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            # Create mask (True outside polygon, False inside).
            mask_arr = geometry_mask(
                gdf.geometry,
                out_shape=(src.height, width),
                transform=profile['transform'],
                invert=True
                )

            # Apply mask to all bands.
            data = np.where(mask_arr, data, nodata)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
        else:
            with rasterio.open(output_path, 'w', **profile) as dst:
                data = src.read(window=window)
                dst.write(data)


def extract_tile_with_overlap(
        tile_fpath, vrt_fpath, out_fpath, overlap_pixels=250
        ):
    """
    Extracts a DEM chunk from a global VRT corresponding to a given tile,
    with optional padding (overlap), and saves it as a GeoTIFF.

    Parameters
    ----------
    tile_fpath : str
        Path to the input DEM tile GeoTIFF.
    vrt_fpath : str
        Path to the global VRT mosaic file.
    out_fpath : str
        Path where the extracted chunk will be saved as a GeoTIFF.
    overlap_pixels : int, optional
        Number of pixels to pad around the tile in all directions.
        Default is 250.
    """

    # Get tile bounds and shape
    with rasterio.open(tile_fpath) as tile_src:
        tile_bounds = tile_src.bounds
        tile_crs = tile_src.crs

    # Open VRT and extract DEM chunk with overlap.
    with rasterio.open(vrt_fpath) as vrt_src:
        # Map tile upper-left and lower-right to VRT indices.
        ul_row, ul_col = vrt_src.index(tile_bounds.left, tile_bounds.top)
        lr_row, lr_col = vrt_src.index(tile_bounds.right, tile_bounds.bottom)

        # Add overlap, ensuring bounds.
        col_off = max(ul_col - overlap_pixels, 0)
        row_off = max(ul_row - overlap_pixels, 0)
        col_end = min(lr_col + overlap_pixels, vrt_src.width)
        row_end = min(lr_row + overlap_pixels, vrt_src.height)

        win_width = col_end - col_off
        win_height = row_end - row_off

        window = Window(col_off, row_off, win_width, win_height)
        dem_chunk = vrt_src.read(1, window=window)
        chunk_transform = window_transform(window, vrt_src.transform)

        # Save extracted chunk to output GeoTIFF.
        profile = vrt_src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": dem_chunk.shape[0],
            "width": dem_chunk.shape[1],
            "transform": chunk_transform,
            "count": 1,
            "dtype": dem_chunk.dtype,
        })

        with rasterio.open(out_fpath, "w", **profile) as dst:
            dst.write(dem_chunk, 1)
