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

# ---- Standard imports.
from time import perf_counter

# ---- Third party imports.
import geopandas as gpd
import whitebox

# ---- Local imports.
from hdml import __datadir__ as datadir
from hdml.tiling import extract_tile, crop_tile
from hdml.topo import (
    dist_to_streams, extract_ridges, dist_to_ridges, ratio_dist,
    height_above_nearest_drainage, height_below_nearest_ridge, ratio_stream,
    local_stats, stream_stats)


OVERWRITE = False

TILES_OVERLAP_DIR = datadir / 'topo' / 'tiles (overlapped)'
TILES_CROPPED_DIR = datadir / 'topo' / 'tiles (cropped)'

FEATURES = ['dem', 'filled_dem', 'smoothed_dem',
            'flow_accum', 'streams', 'geomorphons',
            'slope', 'curvature', 'dist_stream', 'ridges',
            'dist_top', 'alt_stream', 'alt_top',
            'ratio_stream', 'long_hessian_stats', 'long_grad_stats',
            'short_grad_stats', 'stream_grad_stats', 'stream_hessian_stats']

vrt_reprojected = datadir / 'dem' / 'nasadem_102022.vrt'

tiles_gdf = gpd.read_file(datadir / "topo" / "tiles_geom_training.gpkg")


# %% Computing

wbt = whitebox.WhiteboxTools()
wbt.verbose = False

tile_count = 0
total_tiles = len(tiles_gdf)
for tile_key, tile_bbox_data in tiles_gdf.iterrows():
    ty, tx = tile_key
    tile_count += 1
    progress = f"[{tile_count:02d}/{total_tiles}]"

    crop_kwargs = {
        'crop_x_offset': tile_bbox_data.crop_x_offset,
        'crop_y_offset': tile_bbox_data.crop_y_offset,
        'width': tile_bbox_data.core_bbox_pixels[2],
        'height': tile_bbox_data.core_bbox_pixels[3],
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

    all_processed = True
    tile_paths = {}
    for name in FEATURES:
        tile_name = tile_name_template.format(name=name, ty=ty, tx=tx)
        tile_paths[name] = TILES_OVERLAP_DIR / name / tile_name

        if not(TILES_CROPPED_DIR / name / tile_name).exists():
            all_processed = False

    if all_processed is True:
        print(f"{progress} Features already calculated for tile {tile_key}.")
        continue

    func_kwargs = {
        'dem': {
            'func': lambda output, **kwargs: extract_tile(
                output_tile=output, **kwargs),
            'kwargs': {'input_raster': vrt_reprojected,
                       'bbox': tile_bbox_data.ovlp_bbox_pixels,
                       'overwrite': OVERWRITE,
                       'output_dtype': 'Float32'}
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
        'alt_stream': {
            'func': height_above_nearest_drainage,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'dist_stream': tile_paths['dist_stream']}
            },
        'alt_top': {
            'func': height_below_nearest_ridge,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'dist_ridge': tile_paths['dist_top']}
            },
        'ratio_stream': {
            'func': ratio_stream,
            'kwargs': {'dem': tile_paths['smoothed_dem'],
                       'hand': tile_paths['alt_stream'],
                       'dist_stream': tile_paths['dist_stream']}
            },
        'long_hessian_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['curvature'],
                       'window': 41}
            },
        'long_grad_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'window': 41}
            },
        'short_grad_stats': {
            'func': local_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'window': 7}
            },
        'stream_grad_stats': {
            'func': stream_stats,
            'kwargs': {'raster': tile_paths['slope'],
                       'dist_stream': tile_paths['dist_stream'],
                       'fisher': False}
            },
        'stream_hessian_stats': {
            'func': stream_stats,
            'kwargs': {'raster': tile_paths['curvature'],
                       'dist_stream': tile_paths['dist_stream'],
                       'fisher': False}
            },
        }

    # max_short_distance = 7 pixels == 210 m -> halfwidth de 105 m
    # max_long_distance = 41 = 1230 m -> halfwidth = 615 m

    ttot0 = perf_counter()
    for name in FEATURES:
        t0 = perf_counter()
        print(f"{progress} Computing {name} for tile {tile_key}...", end='')
        func = func_kwargs[name]['func']
        kwargs = func_kwargs[name]['kwargs']
        process_feature(name, func, **kwargs)
        t1 = perf_counter()
        print(f' done in {round(t1 - t0):0.0f} sec')
    ttot1 = perf_counter()
    print(f"{progress} All topo feature for tile {tile_key} computed "
          f"in {round(ttot1 - ttot0):0.0f} sec")
