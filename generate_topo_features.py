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

"""
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html

# From Jensen et al. (2025)
# Topography is one of the primary drivers of WTD, with the complex
# relationship between topography and groundwater recharge (and WTD)
# depending on how topography interacts with precipitation,
# evapotranspiration, and geological formations (Moeck et al., 2020).

# Machine learning may provide a strong foundation for exploring and
# simulating such a complicated relationship. Here, we used different
# topographical attributes which are known to control recharge and WTD.
# These attributes include elevation, slope, and topographic index.
#

# Flow-direction, accumulation, TWI, HAND, and stream networks work best
# on the unsmoothed filled DEM. Gaussian smoothing should be applied after
# these indices are computed if the goal is for ML features.
# Otherwise, flow paths can get unrealistic.

# Recommended workflow

# 1. Start with raw DEM,run fill_depressions_wang_and_liu, produces filled DEM.
# 2. Compute hydrological layers: flow direction, flow accumulation,
#    TWI, HAND, streams.
# 3. Restore water mask (optional) → final DEM for AI features.
# 4. Apply gaussian_filter only on the DEM used for AI features
#    (not on the filled DEM used for flow routing).
# 5. Extract slopes, curvatures, elevation features from this smoothed DEM.

# Both breach_depressions and fill_depressions have the same
# goal — to remove spurious sinks that break flow continuity —
# but they achieve it in different ways:

# After this sequence, your dem_filled is hydrologically conditioned — i.e.
# flow directions can be computed without interruptions.
# dem = wbe.read_raster(padded_tile_fpath)

# Most large-scale hydrological studies use depression filling
# (e.g., Wang & Liu, Planchon-Darboux, or Priority-Flood algorithms),
# which is much faster and scales well.

# Breaching is typically reserved for small areas or when you have true
# hydrological sinks that must be preserved.
"""

# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html

# ---- Standard imports
from pathlib import Path

# ---- Third party imports
import whitebox

# ---- Local imports
from sahel import __datadir__ as datadir
from sahel.gishelpers import(
    generate_tiles_bbox, get_dem_filepaths, create_pyramid_overview,
    extract_tile, crop_tile, mosaic_tiles)
from sahel.topo import distance_to_stream, height_above_stream

wbt = whitebox.WhiteboxTools()

# =============================================================================
# User inputs
# =============================================================================
OVERWRITE = False

DEM_PATH = datadir / 'merit' / 'elv_mosaic.tiff'
STREAMS_PATH = datadir / 'merit' / 'river_network_var_Dd.tiff'

FEATURES_PATH = datadir / 'merit' / 'features'
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

TILES_OVERLAP_DIR = FEATURES_PATH / 'tiles (overlapped)'
TILES_OVERLAP_DIR.mkdir(exist_ok=True)

TILES_CROPPED_DIR = FEATURES_PATH / 'tiles (cropped)'
TILES_CROPPED_DIR.mkdir(exist_ok=True)

MOSAIC_OUTDIR = Path(
    "G:/Shared drives/2_PROJETS/251230_BanqueMondiale_ML_for_DWL/"
    "2_TECHNIQUE/5_CARTO/couches"
    )

tiles_bbox = generate_tiles_bbox(
    input_raster=DEM_PATH,
    tile_size=5000,
    overlap=100
    )

# %% Process topo-driven features

tile_count = 0
total_tiles = len(tiles_bbox)
wbt.verbose = False
for (ty, tx), tile_bbox_data in tiles_bbox.items():
    tile_key = (ty, tx)
    tile_count += 1
    progress = f"[{tile_count:02d}/{total_tiles}]"
    tile_paths = {}

    crop_kwargs = {
        'crop_x_offset': tile_bbox_data['crop_x_offset'],
        'crop_y_offset': tile_bbox_data['crop_y_offset'],
        'width': tile_bbox_data['core'][2],
        'height': tile_bbox_data['core'][3],
        'overwrite': OVERWRITE
        }

    # Helper to process a feature.
    def process_feature(name, func, **kwargs):
        tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'

        overlap_tile_path = TILES_OVERLAP_DIR / name / tile_name
        overlap_tile_path.parent.mkdir(parents=True, exist_ok=True)

        if not overlap_tile_path.exists() or OVERWRITE:
            func(output=str(overlap_tile_path), **kwargs)

            cropped_tile_path = TILES_CROPPED_DIR / name / tile_name
            cropped_tile_path.parent.mkdir(parents=True, exist_ok=True)

            crop_tile(overlap_tile_path, cropped_tile_path, **crop_kwargs)

        return overlap_tile_path

    # =========================================================================
    # Extract DEM tile (with overlap)
    # =========================================================================
    name = 'dem'
    print(f"{progress} Extracting {name} tile {tile_key}...")
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    overlap_tile_path = TILES_OVERLAP_DIR / name / tile_name
    overlap_tile_path.parent.mkdir(parents=True, exist_ok=True)
    extract_tile(
        input_raster=DEM_PATH,
        output_tile=overlap_tile_path,
        bbox=tile_bbox_data['overlap'],
        overwrite=OVERWRITE
        )
    tile_paths[name] = overlap_tile_path

    # =========================================================================
    # Extract streams tile (with overlap)
    # =========================================================================
    name = 'streams'
    print(f"{progress} Extracting {name} tile {tile_key}...")
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    overlap_tile_path = TILES_OVERLAP_DIR / name / tile_name
    overlap_tile_path.parent.mkdir(parents=True, exist_ok=True)
    extract_tile(
        input_raster=STREAMS_PATH,
        output_tile=overlap_tile_path,
        bbox=tile_bbox_data['overlap'],
        overwrite=OVERWRITE
        )
    tile_paths[name] = overlap_tile_path

    # =========================================================================
    # Calculate features
    # =========================================================================
    func_kwargs = {
        'slope': {
            'func': wbt.slope,
            'kwargs': {'dem': tile_paths['dem']}},
        'curvature': {
            'func': wbt.profile_curvature,
            'kwargs': {'dem': tile_paths['dem']}},
        'd8_pointer': {
            'func': wbt.d8_pointer,
            'kwargs': {'dem': tile_paths['dem']}},
        'd8_flow_acc': {
            'func': wbt.d8_flow_accumulation,
            'kwargs': {'i': tile_paths['dem'], 'out_type': 'cells'}},
        'dist_to_stream': {
            'func': distance_to_stream,
            'kwargs': {'dem': tile_paths['dem'],
                       'streams': tile_paths['streams']}},
        'height_above_stream': {
            'func': height_above_stream,
            'kwargs': {'dem': tile_paths['dem'],
                       'streams': tile_paths['streams']}},
        'dinf_flow_acc': {
            'func': wbt.d_inf_flow_accumulation,
            'kwargs': {'i': tile_paths['dem'], 'out_type': 'cells'}},
        }

    for name in func_kwargs.keys():
        print(f"{progress} Computing {name} for tile {tile_key}...")
        func = func_kwargs[name]['func']
        kwargs = func_kwargs[name]['kwargs']
        tile_paths[name] = process_feature(name, func, **kwargs)

    name = 'wetness_index'
    print(f"{progress} Computing {name} for tile {tile_key}...")
    func = wbt.wetness_index
    kwargs = {'sca': tile_paths['d8_flow_acc'],
              'slope': tile_paths['slope']}
    tile_paths[name] = process_feature(name, func, **kwargs)

    # median basins level 10 to 12 → ~137.5 km² → 16975 cells → ~15 000 cells
    # median basins level 9 → ~200.9 km² → 24691 cells → ~30 000 cells
    # median basins level 8 → ~472.1 km² → 58283 cells → ~60 000 cells
    # median basins level 7 → ~1480.3 km² → 182753 cells → ~180 000 cells
    # median basins level 6 → ~4433.6 km² → 547283 cells → ~540 000 cells

    thresholds = [15000, 540000]
    for threshold in thresholds:
        name = f'streams_{threshold}'
        print(f"{progress} Computing {name} for tile {tile_key}...")
        func = wbt.extract_streams
        kwargs = {'flow_accum': str(tile_paths['d8_flow_acc']),
                  'threshold': threshold}
        tile_paths[name] = process_feature(name, func, **kwargs)


# %% Mosaic tiles back together


names = ['slope', 'curvature', 'd8_pointer', 'd8_flow_acc', 'wetness_index',
         'dist_to_stream', 'height_above_stream', 'dinf_flow_acc',
         'streams_15000', 'streams_540000']
for i, name in enumerate(names):
    print(f"[{i+1:02d}] Mosaicing {name} tiles...")
    mosaic_path = MOSAIC_OUTDIR / f'{name}.tif'
    if mosaic_path.exists() and OVERWRITE is False:
        continue

    mosaic_tiles(
        tile_paths=get_dem_filepaths(TILES_CROPPED_DIR / name),
        output_raster=mosaic_path,
        overwrite=False,
        cleanup_tiles=False
        )

    create_pyramid_overview(mosaic_path, overwrite=True)


# %%

# TWI combines two important controls on local saturation: upstream
# contributing area (As) and local slope (β): TI = ln(As / tan β).
# That single index therefore captures both accumulation potential (As) and
# drainage ability (slope).

# TWI (implicitly) captures :
# - accumulation tendency (via As),  which correlates
#   with being “downhill / near drainage”
# - Local drainage potential (via slope)
# - Broad-scale wetness-prone locations (valleys, concavities).
