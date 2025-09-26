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
import os.path as osp
import zipfile

# ---- Third party imports
import rasterio


def convert_hgt_to_geotiff(
        zip_path: str, dest_dir: str = None, compress: str = 'zstd'):
    """
    Extracts the .hgt file from a zip archive and converts it to a GeoTIFF.

    Parameters
    ----------
    zip_path : str
        Path to the input ZIP archive containing the .hgt DEM tile.
    dest_dir : str, optional
        Directory to save the resulting GeoTIFF file. If None (default),
        saves to the same directory as the input zip file.
    compress : str or None, optional
        Compression method to use for the output GeoTIFF (e.g., 'zstd',
        'DEFLATE', 'LZW'). If None, no compression is applied.
        Default is 'zstd'.
    """
    if dest_dir is None:
        dest_dir = osp.dirname(zip_path)

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

    # Set the output GeoTIFF path.
    root, _ = osp.splitext(zip_path)
    tif_path = osp.join(dest_dir, osp.basename(root) + '.tif')

    # Write to GeoTIFF.
    if compress is not None:
        profile.update(compress=compress)

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data, 1)
