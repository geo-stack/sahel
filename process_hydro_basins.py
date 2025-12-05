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
import geopandas as gpd
from shapely.geometry import box

# ---- Local imports
from hdml import __datadir__ as datadir


africa_gdf = gpd.read_file(datadir / 'coastline' / 'africa_landmass.gpkg')
africa_shape = africa_gdf.union_all()

minx, miny, maxx, maxy = africa_gdf.total_bounds
africa_bbox = box(minx, miny, maxx, maxy)

basins_all_path = datadir / 'hydro_atlas' / 'BasinATLAS_v10.gdb'

level = 12

layer_name = f'BasinATLAS_v10_lev{level:02d}'

print(f'Reading {layer_name} from {basins_all_path.name}...', flush=True)
basins_gdf = gpd.read_file(str(basins_all_path), layer=layer_name)
print('Number of basins:', len(basins_gdf), flush=True)

print('Projecting to ESRI:102022...', flush=True)
basins_gdf = basins_gdf.to_crs('ESRI:102022')

# We only clip to the bounding box of the African continent because
# clipping to the shape is too long.
print("Clipping to African continent bbox...", flush=True)
basins_gdf = gpd.clip(basins_gdf, africa_bbox)
print('Number of basins:', len(basins_gdf), flush=True)

print("Saving restuls to file...", flush=True)
basins_path = basins_all_path.parent / f'basins_lvl{level:02d}_102022.gpkg'
basins_gdf.to_file(basins_path, layer=layer_name, driver="GPKG")
