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
from sahel import __datadir__ as datadir
from sahel.utils import read_obs_wl


# %%

countries = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo']
dfs = []
for country in countries:
    print(f'Loading WTD data for {country}...')
    filename = datadir / 'data' / f'{country}.xlsx'
    temp = read_obs_wl(filename)
    temp['country'] = country
    dfs.append(temp)

df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
df = df.reset_index(drop=True)

gwl_gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df.LON, df.LAT)],
    crs="EPSG:4326"  # WGS84
    )

# Reproject to ESRI:102022 (Africa Albers Equal Area Conic).
gwl_gdf = gwl_gdf.to_crs("ESRI:102022")

# %%

# Clip to study area, effectively removing bad points that falls in the ocean.

study_area_gdf = gpd.read_file(
    datadir / 'gadm' / 'buffered_boundary_100km.json'
    )
assert gwl_gdf.crs == study_area_gdf.crs

gwl_gdf = gwl_gdf.clip(study_area_gdf.union_all())


# %%

coords = [(point.x, point.y) for point in gwl_gdf.geometry]

dem_path = datadir / 'dem' / 'projected_mosaic_hgt.tif'
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs
    assert gwl_gdf.crs == dem_crs

    ground_elev = [
        val[0] if val[0] != dem.nodata else
        np.nan for val in dem.sample(coords)
        ]


gwl_gdf["ground_elev"] = ground_elev
gwl_gdf["gwl_elev"] = gwl_gdf["ground_elev"] - gwl_gdf['NS']

gwl_gdf.to_file(datadir / 'data', "gwl_obs_all.geojson", driver="GeoJSON")
