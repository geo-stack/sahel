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

# ---- Third party imports
import geopandas as gpd
import pyproj
import rasterio
from shapely.geometry import box
from shapely.ops import transform
from rasterio.windows import Window, transform as window_transform


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
        for fname in zf.namelist():
            if fname.lower().endswith('.hgt'):
                hgt_filename = fname
                break
        else:
            raise FileNotFoundError(f"No .hgt file found in '{zip_path}'.")

    # Open the .hgt file using rasterio (via GDAL VFS).
    vsi_path = f'/vsizip/{zip_path}/{hgt_filename}'
    with rasterio.open(vsi_path) as src:
        profile = src.profile
        data = src.read(1)
        profile.update(driver='GTiff')

    # Write to GeoTIFF.
    if compress is not None:
        profile.update(compress=compress)

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data, 1)


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
        assert vrt_src.crs == tile_crs, "CRS mismatch between VRT and tile!"

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
