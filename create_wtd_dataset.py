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
import pandas as pd
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.wtd_helpers import create_wtd_obs_dataset


def recharge_period_from_basin_area(area_km2: float) -> int:
    """
    Estimates a suitable averaging period (in days) for climatic
    variables.

    This approach assumes larger basins have longer hydrological
    response times.

    Parameters
    ----------
    area_km2 : float
        Surface area of the watershed or basin in square kilometers.

    Returns
    -------
    int
        Recommended period (in days) to average precipitation, based on
        basin area.
    """
    if area_km2 < 100:
        return 30
    elif area_km2 < 500:
        return 60
    elif area_km2 < 1000:
        return 90
    elif area_km2 < 3000:
        return 120
    elif area_km2 < 5000:
        return 150
    elif area_km2 < 10000:
        return 180

    return 360


gwl_gdf = create_wtd_obs_dataset()

# Join information about sub-basin level 12 from the HydroATLAS database.
basins_path = datadir / 'hydro_atlas' / 'basins_lvl12_102022.gpkg'
basins_gdf = gpd.read_file(basins_path)
basins_gdf['basin_area_km2'] = basins_gdf.geometry.area / 1e6

joined = gpd.sjoin(gwl_gdf, basins_gdf, how='left', predicate='within')

# Remove columns that we do not want from the HydroATLAS database.

# syr: annual average in sub-basin
# tmp_dc_syr: air temperature annual average in sub-basin
# pre_mm_syr: precipitation annual average in sub-basin
# pet_mm_syr: potential evapotranspiration annual average in sub-basin
# aet_mm_syr: actual evapotranspiration annual average in sub-basin
# ari_ix_sav: global aridity index
# cmi_ix_syr: climate moisture index annual average in sub-basin

columns = list(gwl_gdf.columns)
columns += ['pre_mm_syr', 'tmp_dc_syr', 'pet_mm_syr', 'aet_mm_syr',
            'ari_ix_sav', 'cmi_ix_syr', 'basin_area_km2', 'HYBAS_ID']

gwl_gdf = joined[columns].copy()
gwl_gdf['HYBAS_ID'] = gwl_gdf['HYBAS_ID'].astype(int)

# Calculate the period for which we will need daily climatic data
# to compute the 'ndvi' and 'precipitation' features.
for index, row in gwl_gdf.iterrows():
    ndays = recharge_period_from_basin_area(row.basin_area_km2)
    date_start = row.DATE - pd.Timedelta(days=60)
    gwl_gdf.loc[index, 'climdata_period_days'] = ndays
    gwl_gdf.loc[index, 'climdata_date_start'] = date_start


# We filter out measurements that are before 2000 and after 2025.
year_min = 2000
year_max = 2025
mask = ((gwl_gdf.climdata_date_start.dt.year < year_min) |
        (gwl_gdf.DATE.dt.year > year_max))
gwl_gdf = gwl_gdf[~mask]

original_count = len(mask)
removed_count = np.sum(mask)
print(f"Removed {removed_count} points (from {original_count}) for which "
      f"daily climatic data are needed before {2000} or after {year_max}.")
print(f'Final dataset has {len(gwl_gdf)} points.')

# Save the water table observations dataset.
gwl_gdf.to_file(datadir / "data" / "wtd_obs_all.gpkg", driver="GPKG")

# Save the basins geometry (we keep only the ones with water level obs).
basins_gdf = basins_gdf.set_index('HYBAS_ID', drop=True)
basins_gdf = basins_gdf.loc[gwl_gdf['HYBAS_ID'].unique()]
basins_gdf = basins_gdf['geometry']


basins_gdf.to_file(datadir / "data" / "wtd_basin_geometry.gpkg", driver="GPKG")
