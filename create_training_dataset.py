# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/hydrodepthml
# =============================================================================

# ---- Standard imports

# ---- Third party imports
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir


gwl_gdf = gpd.read_file(datadir / "data" / "wtd_obs_all.gpkg")

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)

# Extract coordinates of WTD obs as list of (x, y) tuples.
coords = [(geom.x, geom.y) for geom in gwl_gdf.geometry]

tiles_gdf = gpd.read_file(datadir / "topo" / "tiles_geom_training.gpkg")

joined = gpd.sjoin(
    gwl_gdf, tiles_gdf[['tile_index', 'geometry']],
    how='left', predicate='within'
    )
joined = joined.drop(columns=['index_right'])

# %%

input_dir = datadir / 'topo' / 'tiles (cropped)'

ntot = len(np.unique(joined.tile_index))
count = 1
for tile_idx, group in joined.groupby('tile_index'):
    print(f"[{count}/{ntot}] Processing tile index: {tile_idx}...")

    coords = [(geom.x, geom.y) for geom in group.geometry]

    import ast
    ty, tx = ast.literal_eval(tile_idx)

    names = [
        'smoothed_dem',
        'dist_stream',
        'alt_stream',
        'dist_top',
        'alt_top'
        ]

    for name in names:
        tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
        tif_path = input_dir / name / tile_name

        with rasterio.open(tif_path) as src:
            values = np.array(list(src.sample(coords)))
            values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, name] = values[:, 0]

    stat_index_map = {
        'min': 0,
        'max': 1,
        'mean': 2,
        'var': 3,
        'skew': 4,
        'kurt': 5
        }

    name_stat = {
        'long_hessian': ['max', 'mean', 'var', 'skew', 'kurt'],
        'long_grad': ['mean', 'var'],
        'short_grad': ['max', 'var', 'mean'],
        'stream_grad': ['max', 'var', 'mean'],
        'stream_hessian': ['max']
        }

    for name, stats in name_stat.items():
        tile_name = f'{name}_stats_tile_{ty:03d}_{tx:03d}.tif'
        tif_path = input_dir / f'{name}_stats' / tile_name

        with rasterio.open(tif_path) as src:
            values = np.array(list(src.sample(coords)))
            values[values == src.nodata] = np.nan

        for stat in stats:
            index = stat_index_map[stat]
            gwl_gdf.loc[group.index, f'{name}_{stat}'] = values[:, index]

    count += 1

pixel_size = 30

gwl_gdf['ratio_dist'] = (
    gwl_gdf['dist_stream'] / (np.maximum(gwl_gdf['dist_top'], pixel_size))
    )
gwl_gdf['ratio_stream'] = (
    gwl_gdf['alt_stream'] / np.maximum(gwl_gdf['dist_stream'], pixel_size)
    )


gwl_gdf.to_file(datadir / "wtd_obs_training_dataset.gpkg", driver="GPKG")
gwl_gdf.to_csv(datadir / "wtd_obs_training_dataset.csv")


# %%

# Add precip and ndvi avg sub-basin values for each water level observation.

print("Adding NDVI and precipitation data to training dataset...")

ndvi_daily = pd.read_csv(
    datadir / 'ndvi' / 'ndvi_mosaic_index.csv',
    index_col=0, parse_dates=True, dtype={'file': str}
    )

precip_daily = pd.read_csv(
    datadir / 'precip' / 'precip_tif_index.csv',
    index_col=0, parse_dates=True, dtype={'file': str}
    )

for index, row in gwl_gdf.iterrows():
    date_range = pd.date_range(row.climdata_date_start, row.DATE)
    basin_id = str(int(row.HYBAS_ID))

    # Add mean daily NDVI values (at the basin scale).
    ndvi_values = ndvi_daily.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'ndvi'] = np.mean(ndvi_values)

    # Add mean daily PRECIP values (at the basin scale).
    precip_values = precip_daily.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'precipitation'] = np.mean(precip_values)

gwl_gdf.to_file(datadir / "wtd_obs_training_dataset.gpkg", driver="GPKG")
gwl_gdf.to_csv(datadir / "wtd_obs_training_dataset.csv")
