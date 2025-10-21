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

# ---- Standard imports.
import os.path as osp


# ---- Third party imports.
import pandas as pd
import geopandas as gpd


# ---- Local imports.
from sahel import __datadir__


def create_unified_geometry(output_bath: str, dst_crs: str = None):
    gadm_dirpath = osp.join(__datadir__, 'gadm')

    geojson_files = [
        "Togo_gadm41_0.json",
        "Benin_gadm41_0.json",
        "Burkina_gadm41_0.json",
        "Chad_gadm41_0.json",
        "Guinea_gadm41_0.json",
        "Mali_gadm41_0.json",
        "Mauritania_gadm41_0.json",
        "Niger_gadm41_0.json",
        "Senegal_gadm41_0.json"
        ]

    # Concatenate into a single GeoDataFrame.
    gdfs = [gpd.read_file(osp.join(gadm_dirpath, f)) for f in geojson_files]
    src_crs = gdfs[0].crs

    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=src_crs)

    if dst_crs is not None:
        gdf_all = gdf_all.to_crs(dst_crs)
        dst_crs = gdf_all.crs
    else:
        dst_crs = src_crs

    # Dissolve into a single geometry (union of all shapes).
    unified_boundary = gdf_all.union_all()

    # Save boundary to geojson file.
    gdf_unified_boundary = gpd.GeoSeries(unified_boundary, crs=dst_crs)
    gdf_unified_boundary.to_file(output_bath, driver="GeoJSON")


def buffer_geometry(
        input_path: str, output_bath: str,
        buffer_dist: int, dst_crs: str = None):
    """
    Reads a vector file, applies a buffer of given distance to each geometry,
    optionally reprojects to a target CRS, and saves the result as a GeoJSON.

    Parameters
    ----------
    input_path : str
        Path to input vector file (e.g., GeoJSON).
    output_bath : str
        Path to output buffered GeoJSON.
    buffer_dist : int
        Buffer distance (in CRS units).
    dst_crs : str, optional
        CRS to reproject geometries before buffering.
    """

    gdf = gpd.read_file(input_path)
    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)

    buffered_geom = gdf.buffer(buffer_dist)

    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geom)
    buffered_gdf.to_file(output_bath, driver="GeoJSON")


if __name__ == '__main__':
    create_unified_geometry()
