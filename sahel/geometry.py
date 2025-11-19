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
from pathlib import Path
import os.path as osp
from math import ceil, floor

# ---- Third party imports.
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


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

    return buffered_gdf


def create_buffered_bounding_box(
        points_path: Path | str,
        output_path: Path | str = None,
        buffer_distance: float = 100 * 10**3,
        ) -> Path:
    """
    Creates a rectangular study area from GeoJSON points with a buffer.

    Note that input data is automatically reprojected to ESRI:102022 if
    needed and output is saved in ESRI:102022 to maintain consistency with
    project data.

    Parameters
    ----------
    points_path : Path | str
        Path to input file containing point features (e.g., water table depth
        observation points). Supports formats readable by GeoPandas (GeoJSON,
        Shapefile, GeoPackage, etc.).
    output_path : Path | str
        Path where the output GeoJSON file will be saved.
    buffer_distance : float, optional
        Buffer distance in meters to expand the bounding box. Default is
        100,000 m (100 km). Set to 0 for no buffer.

    Returns
    -------
    Path
        Path to the created output file.
    """
    points_path = Path(points_path)

    pts_gdf = gpd.read_file(points_path)

    target_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic
    if pts_gdf.crs != target_crs:
        pts_gdf = pts_gdf.to_crs(target_crs)

    # Create a GeoDataFrame with the bounding box of the WTD obs points.
    bounds = pts_gdf.total_bounds  # (minx, miny, maxx, maxy)

    # Apply buffer in meters.
    if buffer_distance > 0:
        bounds = (
            floor(bounds[0] - buffer_distance),  # minx
            floor(bounds[1] - buffer_distance),  # miny
            ceil(bounds[2] + buffer_distance),   # maxx
            ceil(bounds[3] + buffer_distance)    # maxy
            )
    bbox = box(*bounds)
    bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox}], crs=target_crs)

    # Add metadata.
    bbox_gdf['buffer_meters'] = buffer_distance

    if output_path:
        bbox_gdf.to_file(output_path, driver='GeoJSON')

    return bbox_gdf


if __name__ == '__main__':
    from sahel import __datadir__ as datadir
    wtd_path = Path(datadir) / 'data' / 'wtd_obs_all.geojson'
    output_path = Path(datadir) / 'data' / 'wtd_obs_boundary.geojson'
    bbox_gdf = create_buffered_bounding_box(wtd_path, output_path)
