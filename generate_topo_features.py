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
import math
from pathlib import Path
from time import perf_counter

# ---- Third party imports.
import geopandas as gpd
import whitebox

# ---- Local imports.
from sahel import __datadir__ as datadir
from sahel.gishelpers import get_dem_filepaths
from sahel.tiling import (
    generate_tiles_bbox, extract_tile, crop_tile, mosaic_tiles)
from sahel.topo import (
    dist_to_streams, extract_ridges, dist_to_ridges, ratio_dist)


OVERWRITE = False
TILES_OVERLAP_DIR = datadir / 'training' / 'tiles (overlapped)'
TILES_CROPPED_DIR = datadir / 'training' / 'tiles (cropped)'

FEATURES = ['dem', 'filled_dem', 'smoothed_dem',
            'flow_accum', 'streams', 'geomorphons',
            'slope', 'curvature', 'dist_stream', 'ridges',
            'dist_top', 'ratio_dist']


# %% Tiling

obs_points_path = datadir / 'data' / 'wtd_obs_all.geojson'

boundary_gdf = gpd.read_file(datadir / 'data' / 'wtd_obs_boundary.geojson')
zone_bbox = tuple(boundary_gdf.total_bounds)

vrt_reprojected = datadir / 'dem' / 'nasadem_102022.vrt'

tiles_bbox_data = generate_tiles_bbox(
    input_raster=vrt_reprojected,
    tile_size=5000,
    overlap=100 * 30,  # 100 pixels at 30 meters resolution
    filter_points_path=obs_points_path
    )


# %% Computing

wbt = whitebox.WhiteboxTools()
wbt.verbose = False

tile_count = 0
total_tiles = len(tiles_bbox_data)
for tile_key, tile_bbox_data in tiles_bbox_data.items():
    ty, tx = tile_key
    tile_count += 1
    progress = f"[{tile_count:02d}/{total_tiles}]"

    crop_kwargs = {
        'crop_x_offset': tile_bbox_data['crop_x_offset'],
        'crop_y_offset': tile_bbox_data['crop_y_offset'],
        'width': tile_bbox_data['core'][2],
        'height': tile_bbox_data['core'][3],
        'overwrite': False
        }

    tile_name_template = '{name}_tile_{ty:03d}_{tx:03d}.tif'

    # Helper to process a feature.
    def process_feature(name, func, **kwargs):
        tile_name = tile_name_template.format(name=name, ty=ty, tx=tx)

        overlap_tile_path = TILES_OVERLAP_DIR / name / tile_name
        overlap_tile_path.parent.mkdir(parents=True, exist_ok=True)

        if not overlap_tile_path.exists() or OVERWRITE:
            func(output=str(overlap_tile_path), **kwargs)

            cropped_tile_path = TILES_CROPPED_DIR / name / tile_name
            cropped_tile_path.parent.mkdir(parents=True, exist_ok=True)

            crop_tile(overlap_tile_path, cropped_tile_path, **crop_kwargs)

        return overlap_tile_path

    tile_paths = {}
    for name in FEATURES:
        tile_name = tile_name_template.format(name=name, ty=ty, tx=tx)
        tile_paths[name] = TILES_OVERLAP_DIR / name / tile_name

    func_kwargs = {
        'dem': {
            'func': lambda output, **kwargs: extract_tile(
                output_tile=output, **kwargs),
            'kwargs': {'input_raster': vrt_reprojected,
                       'bbox': tile_bbox_data['overlap'],
                       'overwrite': OVERWRITE}
            },
        'filled_dem': {
            'func': wbt.fill_depressions,
            'kwargs': {'dem': tile_paths['dem']}
            },
        'smoothed_dem': {
            'func': wbt.gaussian_filter,
            'kwargs': {'i': tile_paths['filled_dem'],
                       'sigma': 1.0}
            },
        'flow_accum': {
            'func': wbt.d8_flow_accumulation,
            'kwargs': {'i': tile_paths['smoothed_dem'],
                       'out_type': 'cells'}
            },
        'streams': {
            'func': wbt.extract_streams,
            'kwargs': {'flow_accum': tile_paths['flow_accum'],
                       'threshold': 1500}
            },
        'geomorphons': {
            'func': wbt.geomorphons,
            'kwargs': {'dem': tile_paths['smoothed_dem']}
            },
        'slope': {
            'func': wbt.slope,
            'kwargs': {'dem': tile_paths['smoothed_dem']}
            },
        'curvature': {
            'func': wbt.profile_curvature,
            'kwargs': {'dem': tile_paths['smoothed_dem']}
            },
        'dist_stream': {
            'func': dist_to_streams,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'streams': tile_paths['streams']}
            },
        'ridges': {
            'func': extract_ridges,
            'kwargs': {'geomorphons': tile_paths['geomorphons'],
                       'ridge_size': 30,
                       'flow_acc': tile_paths['flow_accum'],
                       'max_flow_acc': 2}

            },
        'dist_top': {
            'func': dist_to_ridges,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'streams': tile_paths['streams'],
                       'ridges': tile_paths['ridges']}
            },
        'ratio_dist': {
            'func': ratio_dist,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'dist_stream': tile_paths['dist_stream'],
                       'dist_ridge': tile_paths['dist_top']}
            },
        }

    for name in FEATURES:
        t0 = perf_counter()
        print(f"{progress} Computing {name} for tile {tile_key}...", end='')
        func = func_kwargs[name]['func']
        kwargs = func_kwargs[name]['kwargs']
        process_feature(name, func, **kwargs)
        t1 = perf_counter()
        print(f' done in {round(t1 - t0):0.0f} sec')

    if tile_count == 1:
        break

# %% Mosaicing

MOSAIC_OUTDIR = datadir / 'training'

for i, name in enumerate(FEATURES[:1]):
    print(f"[{i+1:02d}] Mosaicing {name} tiles...")
    mosaic_path = MOSAIC_OUTDIR / f'{name}.vrt'
    if mosaic_path.exists() and OVERWRITE is False:
        continue

    mosaic_tiles(
        tile_paths=get_dem_filepaths(TILES_CROPPED_DIR / name),
        output_raster=mosaic_path,
        overwrite=False,
        cleanup_tiles=False
        )

# %%
import itertools
import rasterio
from scipy import stats
import numpy as np
from rasterio.transform import rowcol
import pandas as pd
from whitebox import WhiteboxTools
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label
from skimage.measure import regionprops
import os
import pandas as pd

from time import perf_counter

training_df = {}

print('Processing smoothed dem...')
t0 = perf_counter()
with rasterio.open(tile_paths['smoothed_dem']) as src:
    new_transform = src.transform
    new_width = src.width
    new_height = src.height

    smoothed_dem = src.read(1)

    cols, rows = np.meshgrid(np.arange(new_width), np.arange(new_height))
    xs, ys = rasterio.transform.xy(new_transform, rows, cols, offset="center")

    xs = np.array(xs).reshape(smoothed_dem.shape)
    ys = np.array(ys).reshape(smoothed_dem.shape)

t1 = perf_counter()
print(t1 - t0)

with rasterio.open(tile_paths['ridges']) as src:
    ridges = src.read(1)

with rasterio.open(tile_paths['streams']) as src:
    streams = src.read(1)
