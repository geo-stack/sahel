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

# ---- Standard imports

# ---- Third party imports
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import geopandas as gpd
import whitebox

# ---- Local imports
from sahel import __datadir__ as datadir

wbt = whitebox.WhiteboxTools()

# %% Mask DEM to study area (with buffer)

print('Masking DEM to study area...')

study_area_gdf = gpd.read_file(
    datadir / 'gadm' / 'buffered_boundary_100km.json'
    )

mask_path = datadir / 'gadm' / 'buffered_boundary_100km.json'
dem_path = datadir / 'dem' / 'projected_mosaic_hgt.tif'
masked_dem_path = datadir / 'dem' / 'dem_30m_masked.tif'


if not masked_dem_path.exists():
    with rasterio.open(dem_path, 'r') as src:
        if study_area_gdf.crs != src.crs:
            raise ValueError('Boundary crs does not match that of the DEM.')

        # Create mask (True outside polygon, False inside).
        mask_arr = geometry_mask(
            study_area_gdf.geometry,
            out_shape=(src.height, src.width),
            transform=src.profile['transform'],
            invert=True
            )

        # Apply mask to all bands.
        data = src.read()
        data = np.where(mask_arr, data, src.nodata)

        with rasterio.open(masked_dem_path, 'w', **src.profile) as dst:
            dst.write(data)

# %% Generate features from topo

OVERWRITE = False

wbt = whitebox.WhiteboxTools()

# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html

feat_path = datadir / 'dem' / 'features (250m)'
feat_path.mkdir(parents=True, exist_ok=True)

# Fill deperessions.
filled_dem_path = feat_path / 'dem_filled.tif'
if not filled_dem_path.exists() or OVERWRITE:
    wbt.fill_depressions_wang_and_liu(
        dem=masked_dem_path,
        output=filled_dem_path,
        )

# Calculate local slope.
slope_path = feat_path / 'slope.tif'
if not slope_path.exists() or OVERWRITE:
    wbt.slope(
        dem=filled_dem_path,
        output=slope_path
        )

# Calculate profile curvature.
curvature_path = feat_path / 'curvature.tif'
if not curvature_path.exists() or OVERWRITE:
    wbt.profile_curvature(
        dem=filled_dem_path,
        output=curvature_path
        )

# Calculate D8 pointer.
d8_pointer_path = feat_path / 'd8_pointer.tif'
if not d8_pointer_path.exists() or OVERWRITE:
    wbt.d8_pointer(
        dem=filled_dem_path,
        output=d8_pointer_path
        )

# Calculate D8 flow accumulation.
d8_flow_acc_path = feat_path / 'd8_flow_acc.tif'
if not d8_flow_acc_path.exists() or OVERWRITE:
    wbt.d8_flow_accumulation(
        i=filled_dem_path,
        output=d8_flow_acc_path,
        out_type='cells'
        )

# Calculate the Topographic Wetness Index.

# TWI combines two important controls on local saturation: upstream
# contributing area (As) and local slope (β): TI = ln(As / tan β).
# That single index therefore captures both accumulation potential (As) and
# drainage ability (slope).

# TWI (implicitly) captures :
# - accumulation tendency (via As),  which correlates
#   with being “downhill / near drainage”
# - Local drainage potential (via slope)
# - Broad-scale wetness-prone locations (valleys, concavities).

wetness_index_path = feat_path / 'wetness_index.tif'
if not wetness_index_path.exists() or OVERWRITE:
    wbt.wetness_index(
        sca=d8_flow_acc_path,
        slope=slope_path,
        output=wetness_index_path)

# Extract streams and related features.
thresholds = [2200, 7553, 23684]  # basin level 10, 8, 7
for threshold in thresholds:
    streams_path = feat_path / f'streams_at_{threshold}.tif'
    if not streams_path.exists() or OVERWRITE:
        wbt.extract_streams(
            flow_accum=d8_flow_acc_path,
            output=streams_path,
            threshold=threshold
            )

    output = feat_path / f'downslope_distance_to_stream_at_{threshold}.tif'
    if not output.exists() or OVERWRITE:
        wbt.downslope_distance_to_stream(
            dem=filled_dem_path,
            streams=streams_path,
            output=output
            )

    output = feat_path / f'elevation_above_stream_at_{threshold}.tif'
    if not output.exists() or OVERWRITE:
        wbt.elevation_above_stream(
            dem=filled_dem_path,
            streams=streams_path,
            output=output,
            )

    output = feat_path / f'subbasins_at_{threshold}.tif'
    if not output.exists() or OVERWRITE:
        wbt.subbasins(
            d8_pntr=d8_pointer_path,
            streams=streams_path,
            output=output
            )

# # Invert DEM for ridges extraction and related features.
# inv_dem_path = feat_path / 'dem_inverted.tif'
# if not inv_dem_path.exists() or OVERWRITE:
#     wbt.multiply(
#         input1=masked_dem_path,
#         input2=-1.0,
#         output=inv_dem_path
#         )

# # Fill deperessions in inverted DEM.
# inv_filled_dem_path = feat_path / 'dem_inverted_filled.tif'
# if not inv_filled_dem_path.exists() or OVERWRITE:
#     wbt.fill_depressions_wang_and_liu(
#         dem=inv_dem_path,
#         output=inv_filled_dem_path,
#         )

# # Calculate inverted D8 flow accumulation.
# inv_d8_flow_acc_path = feat_path / 'd8_flow_acc_inverted.tif'
# if not inv_d8_flow_acc_path.exists() or OVERWRITE:
#     wbt.d8_flow_accumulation(
#         i=inv_filled_dem_path,
#         output=inv_d8_flow_acc_path,
#         out_type='cells'
#         )

# Extract ridges and related features.

# elev_relative_to_watershed_min_max
# subbasins

# thresholds = [2200, 7553, 23684]  # basin level 10, 8, 7
# for threshold in thresholds:
#     ridges_path = feat_path / f'ridges_at_{threshold}.tif'
#     if not ridges_path.exists() or OVERWRITE:
#         wbt.extract_streams(
#             flow_accum=inv_d8_flow_acc_path,
#             output=ridges_path,
#             threshold=threshold
#             )

#     # Calculate upslope distance to ridge.
#     output = feat_path / f'upslope_distance_to_ridge_at_{threshold}.tif'
#     if not output.exists() or OVERWRITE:
#         wbt.downslope_distance_to_stream(
#             dem=inv_filled_dem_path,
#             streams=ridges_path,
#             output=output
#             )

#     # Calculate elevation below ridge.
#     output = feat_path / f'elevation_below_ridge_at_{threshold}.tif'
#     if not output.exists() or OVERWRITE:
#         wbt.elevation_above_stream(
#             dem=inv_filled_dem_path,
#             streams=ridges_path,
#             output=output,
#             )

# %%

# sigma=1 is mild smoothing (~3×3 pixel influence).
# Increase to 2–3 only if DEM is very noisy.
# Default value used in WhiteboxTools is 0.75.

# print('Smoothing DEM...', flush=True)
# smooth_dem_path = feat_path / 'mosaic_hgt_250m_smooth.tif'
# wbt.gaussian_filter(
#     i=filled_dem_path,
#     output=smooth_dem_path,
#     sigma=0.75,
#     )
