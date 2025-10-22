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

# ---- Local imports.
from sahel import __datadir__
from sahel.geometry import buffer_geometry, create_unified_geometry

dst_crs = 'ESRI:102022'    # Africa Albers Equal Area Conic
buffer_dist = 100 * 10**3  # in meters

boundary_path = Path(__datadir__) / 'gadm' / 'unified_boundary.json'
if not boundary_path.exists():
    create_unified_geometry(boundary_path, dst_crs)

buff_geo_path = (
    boundary_path.parent / f'buffered_boundary_{int(buffer_dist/1000)}km.json'
    )
if not buff_geo_path.exists():
    buffer_geometry(boundary_path, buff_geo_path, buffer_dist)
