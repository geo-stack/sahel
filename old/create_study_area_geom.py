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

# ---- Standard imports
from pathlib import Path
from shapely.geometry import box

# ---- Third party imports
import geopandas as gpd

# ---- Local imports
from sahel import __datadir__
from sahel.geometry import buffer_geometry, create_unified_geometry

dst_crs = 'ESRI:102022'    # Africa Albers Equal Area Conic
buffer_dist = 100 * 10**3  # in meters

outdir = Path(__datadir__) / 'gadm'

# Create study area from Global Administrative Area (GADM) from the
# target countries.
study_area_path = outdir / 'study_area.json'
if not study_area_path.exists():
    create_unified_geometry(study_area_path, dst_crs)

# Add a buffer (in meters).
buff_study_area_path = (
    outdir / f'study_area_({int(buffer_dist/1000)}km buffer).json'
    )
if not buff_study_area_path.exists():
    buffered_gdf = buffer_geometry(
        study_area_path, buff_study_area_path, buffer_dist
        )

# Create bounding box of the buffered study area.
study_area_bbox_geo_path = (
    outdir / f'study_area_bbox_({int(buffer_dist/1000)}km buffer).json'
    )
if not study_area_bbox_geo_path.exists():
    minx, miny, maxx, maxy = buffered_gdf.total_bounds
    bbox_polygon = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=dst_crs)
    bbox_gdf.to_file(study_area_bbox_geo_path, driver='GeoJSON')
