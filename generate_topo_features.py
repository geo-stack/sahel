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
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import geopandas as gpd
import whitebox

# ---- Local imports
from sahel import __datadir__
from sahel.utils import read_obs_wl
from sahel.gishelpers import (
    get_dem_filepaths, tile_in_boundary, extract_tile_with_overlap,
    convert_hgt_to_geotiff, extract_vertical_band)
from sahel.geometry import buffer_geometry, create_unified_geometry

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

# https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html

# Reasoning: geomorphons describe local landform patterns (ridge, valley,
# flat, shoulder, hollow, etc.). They are sensitive to high-frequency
# noise, so compute them from a smoothed DEM (but after hydrology-critical
# steps have been run on the filled DEM). They are not a substitute for
# hydrology indices — they are complementary.


# %% Create Study Area Boundary

dst_crs = 'ESRI:102022'

boundary_path = Path(__datadir__) / 'gadm' / 'unified_boundary.json'

if not boundary_path.exists():
    create_unified_geometry(boundary_path, dst_crs)

buffer_dist = 100 * 10**3
buff_geo_path = (
    boundary_path.parent / f'buffered_boundary_{int(buffer_dist/1000)}km.json'
    )

if not buff_geo_path.exists():
    buffer_geometry(boundary_path, buff_geo_path, buffer_dist)

gdf = gpd.read_file(buff_geo_path)


# %% Mask DEM Outside Boundary

dem_dir = Path(__datadir__) / 'dem'
dem_path = dem_dir / 'projected_mosaic_hgt_250m.tif'
masked_dem_path = dem_dir / 'mosaic_hgt_250m_masked.tif'
with rasterio.open(dem_path, 'r') as src:
    if gdf.crs != src.crs:
        raise ValueError('Boundary crs does not match that of the DEM.')

    # Create mask (True outside polygon, False inside).
    mask_arr = geometry_mask(
        gdf.geometry,
        out_shape=(src.height, src.width),
        transform=src.profile['transform'],
        invert=True
        )

    # Apply mask to all bands.
    data = src.read()
    data = np.where(mask_arr, data, src.nodata)

    with rasterio.open(masked_dem_path, 'w', **src.profile) as dst:
        dst.write(data)

# %%

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

wbt = whitebox.WhiteboxTools()

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

# https://www.whiteboxgeo.com/manual/wbt_book/available_tools/
# hydrological_analysis.html#BreachDepressionsLeastCost


print('Filling deperessions...', flush=True)
filled_dem_path = dem_dir / 'mosaic_hgt_250m_filled.tif'
wbt.fill_depressions_wang_and_liu(
    dem=masked_dem_path,
    output=filled_dem_path,
    )

# %%

# sigma=1 is mild smoothing (~3×3 pixel influence).
# Increase to 2–3 only if DEM is very noisy.
# Default value used in WhiteboxTools is 0.75.

print('Smoothing DEM...', flush=True)
smooth_dem_path = dem_dir / 'mosaic_hgt_250m_smooth.tif'
wbt.gaussian_filter(
    i=filled_dem_path,
    output=smooth_dem_path,
    sigma=0.75,
    )

# %% Slope
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#slope

print('Calculating local slope...', flush=True)
slope_path = dem_dir / 'mosaic_250m_slope.tif'
wbt.slope(filled_dem_path, slope_path)

# %% Profile Curvature
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#profile_curvature

print('Calculating Profile Curvature...', flush=True)
curvature_path = dem_dir / 'mosaic_250m_curvature.tif'
wbt.profile_curvature(filled_dem_path, curvature_path)

# %% D8 Flow Accumulation
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#d8_flow_accum

print('Calculating D8 flow accumulation...', flush=True)
d8_flow_acc_path = dem_dir / 'mosaic_hgt_250m_d8_flow_acc.tif'
wbt.d8_flow_accumulation(filled_dem_path, d8_flow_acc_path,)

# %% Topographic Wetness Index
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#wetness_index

# TWI combines two important controls on local saturation: upstream
# contributing area (As) and local slope (β): TI = ln(As / tan β).
# That single index therefore captures both accumulation potential (As) and
# drainage ability (slope).

# TWI (implicitly) captures :
# - accumulation tendency (via As),  which correlates
#   with being “downhill / near drainage”
# - Local drainage potential (via slope)
# - Broad-scale wetness-prone locations (valleys, concavities).

print('Calculating the Topographic Wetness Index...', flush=True)
wetness_index_path = dem_dir / 'mosaic_hgt_250m_wetness_index.tif'
wbt.wetness_index(
    sca=d8_flow_acc_path,
    slope=slope_path,
    output=wetness_index_path)

# %% Geomorphons
# https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#geomorphons

print('Calculating the geomorphons...', flush=True)
geomorphons_path = dem_dir / 'mosaic_hgt_250m_geomorphons.tif'
wbt.geomorphons(
    smooth_dem_path,
    geomorphons_path,
    )

# %%

wbt.downslope_distance_to_stream(
    dem,
    streams,
    output,
    dinf=False,
    callback=default_callback
)

wbt.downslope_flowpath_length(
    d8_pntr,
    output,
    watersheds=None,
    weights=None,
    esri_pntr=False,
    callback=default_callback
)

wbt.elevation_above_stream(
    dem,
    streams,
    output,
    callback=default_callback
)
