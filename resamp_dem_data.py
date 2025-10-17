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
Resample the projected (ESRI:102022) NASADEM and surface water mask rasters to
a target resolution of 250 m for large-scale static water table depth (WTD)
modeling in the Sahel.

The original dataset (NASADEM, ~30 m resolution at the equator) provides
fine-scale elevation detail, but such resolution exceeds the needs of this
project, which targets semi-regional patterns relevant to agricultural
water management rather than local WTD variations.

Downsampling reduces data volume and computational cost while preserving
hydrologically meaningful terrain and surface water features. A target
resolution of 250 m compares well with similar studies conducted at the
regional scale (e.g., M. Yueling et al., 2024; J. Janssen et al., 2025).
It also ensures that DEM-derived features align more closely with the
resolution of meteorological datasets (typically >500 m) and with the
relatively coarse spatial density of water table depth observations used
to train the model.

This script should be executed after:
    1. `download_dem_data.py` — downloads the NASADEM tiles.
    2. `reproj_dem_data.py` — reprojects them to a common coordinate system.

Processing details
------------------
- Continuous rasters (e.g., DEM) are aggregated using the mean to preserve
  representative elevation values.
- Categorical rasters (e.g., surface water mask) use mode-based resampling
  to maintain class integrity.
"""

# ---- Standard imports.
from pathlib import Path

# ---- Third party imports.
from osgeo import gdal

# ---- Local imports.
from sahel import __datadir__
from sahel.gishelpers import resample_raster

gdal.UseExceptions()

from osgeo import gdal

target_res = 250  # in meters
dem_dir = Path(__datadir__) / "dem"

proj_hgt_path = dem_dir / 'projected_mosaic_hgt.tif'
resamp_hgt_path = dem_dir / f'projected_mosaic_hgt_{target_res}m.tif'

print(f"Downsampling '{proj_hgt_path.name}' to {target_res} m...")
if not resamp_hgt_path.exists():
    resample_raster(proj_hgt_path, resamp_hgt_path,
                    target_res=target_res, resample_method='average')
print(f'-> {resamp_hgt_path}\n')

proj_swb_path = dem_dir / 'projected_mosaic_swb.tif'
resamp_swb_path = dem_dir / f'projected_mosaic_swb_{target_res}m.tif'

print(f"Downsampling '{proj_swb_path.name}' to {target_res} m...")
if not resamp_swb_path.exists():
    resample_raster(proj_swb_path, resamp_swb_path,
                    target_res=target_res, resample_method='mode')
print(f'-> {resamp_swb_path}\n')
