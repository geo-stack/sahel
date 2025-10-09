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


