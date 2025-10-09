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
crs = gdfs[0].crs

gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs)

# Dissolve into a single geometry (union of all shapes).
unified_boundary = gdf_all.union_all()

# Save boundary to geojson file.
gdf_unified_boundary = gpd.GeoSeries(unified_boundary, crs=crs)

gdf_unified_boundary.to_file(
    osp.join(gadm_dirpath, "unified_boundary.json"),
    driver="GeoJSON")
