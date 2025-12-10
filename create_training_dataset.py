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

# ---- Third party imports
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---- Local imports
from hdml import __datadir__ as datadir


gwl_gdf = gpd.read_file(datadir / "data" / "wtd_obs_all.gpkg")
gwl_gdf = gwl_gdf.set_index("ID", drop=True)

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)

ndvi_daily = pd.read_csv(
    datadir / 'ndvi' / 'ndvi_vrt_index.csv',
    index_col=0, parse_dates=True, dtype={'file': str}
    )

precip_daily = pd.read_csv(
    datadir / 'precip' / 'precip_tif_index.csv',
    index_col=0, parse_dates=True, dtype={'file': str}
    )

# Extract coordinates of WTD obs as list of (x, y) tuples.
coords = [(geom.x, geom.y) for geom in gwl_gdf.geometry]


# gwl_gdf["ground_elev"] = ground_elev
# gwl_gdf["gwl_elev"] = gwl_gdf["ground_elev"] - gwl_gdf['NS']

# gwl_gdf.to_file(datadir / 'training' / "gwl_obs_all.geojson", driver="GeoJSON")
