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

# ---- Third party imports
import numpy as np
import geopandas as gpd

# ---- Local imports
from sahel import __datadir__


study_area_gdf = gpd.read_file(
    Path(__datadir__) / 'gadm' / 'buffered_boundary_100km.json'
    )

basins_all_path = Path(__datadir__) / 'hydro_atlas' / 'BasinATLAS_v10.gdb'
basins_path = basins_all_path.parent / 'sahel_BasinATLAS_v10.gpkg'
if not basins_path.exists():
    for layer in range(1, 13):
        layer_name = f'BasinATLAS_v10_lev{layer:02d}'
        print(f"Clipping layer {layer:02d}...", flush=True)
        basins_gdf = gpd.read_file(str(basins_all_path), layer=layer_name)
        basins_gdf = basins_gdf.to_crs('ESRI:102022')

        basins_clipped = gpd.clip(basins_gdf, study_area_gdf)
        basins_clipped.to_file(basins_path, layer=layer_name, driver="GPKG")

# %%

for layer in range(1, 13):
    layer_name = f'BasinATLAS_v10_lev{layer:02d}'
    basins_gdf = gpd.read_file(basins_path, layer=layer_name)

    area_m2 = basins_gdf.geometry.area
    area_km2 = area_m2 / 1e6
    med_km2 = np.median(area_km2)

    print(layer_name)
    print(f'p50: {med_km2} km2')
    print('ncels:', int(med_km2 / 0.250**2))

#  2200 cells -> match basins level 10 to 12 → ~137.5 km²
#  3215 cells -> match basins level 9 → ~200.9 km²
#  7553 cells -> match basins level 8 → ~472.1 km²
# 23684 cells -> match basins level 7 → ~1480.3 km²
# 70938 cells -> match basins level 6 → ~4433.6 km²

# At 250 m resolution, even the "small" basins are regional in scale
# (tens to thousands of km²). That means using those median-based thresholds
# in flow-accumulation stream extraction will keep only fairly large rivers
# and main valley systems, which is exactly what we want when modelling
# regional groundwater / water-table depth (WTD) rather than local drains.
