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
This script mosaics and reprojects DEM and surface water mask raster bands
for the Sahel region using NASADEM data.

It builds VRT mosaics from all tiles, then projects:
    - DEM (band 1) using bilinear interpolation for smooth elevation
    - Surface water mask (band 2) using nearest neighbor interpolation to
      preserve classes

All outputs are reprojected to Africa Albers Equal Area Conic (ESRI:102022)
at 30-meter spatial resolution.

Projection rationale
--------------------
The NASADEM tiles are natively provided in a geographic coordinate
reference system (WGS 84, EPSG:4326), where pixel sizes and tile areas vary
with latitude. For any spatial analysis requiring accurate area or distance
calculations—such as hydrological modeling, water budget estimation, or
geomorphological analysis—it is essential to work in a projected coordinate
system with consistent units (meters) and pixel areas. The Africa Albers
Equal Area Conic projection (ESRI:102022) is specifically chosen because it
minimizes area distortion across the continent, ensuring that each pixel
represents a uniform ground area. This is particularly important for
continental-scale studies, such as modeling groundwater table depth across
the Sahel.

Analysis performed on the DEM with Whitebox Tools (such as slope, aspect,
flow accumulation, stream extraction, and terrain indices) is used to
generate key geomorphological and hydrological features for training our AI
model. These features help predict water table depth across the Sahel by
quantifying topographic controls on groundwater dynamics. Using an equal-
area projection ensures that all derived features (e.g., catchment area,
flow length, terrain ruggedness) are spatially consistent and physically
meaningful across the region.

Web Mercator (EPSG:3857) is a commonly used projection for web mapping, but
it is not suitable for scientific analysis over large regions because it
significantly distorts areas, especially as latitude increases. In contrast,
the Africa Albers Equal Area Conic projection preserves areas, making it the
appropriate choice for accurate spatial analysis and modeling in Africa.

Resolution rationale
--------------------
The 30-meter pixel size is chosen to closely match the native resolution of
the NASADEM dataset, which is provided at 1 arc-second (approximately 30
meters at the equator). This allows us to preserve the maximum spatial
detail available in the source data, ensuring that topographic and
hydrological features extracted for machine learning and water table
modeling are as accurate and representative as possible for the Sahel
region.

Requirements
------------
- NASADEM tiles produced with the script 'download_dem_data.py'.
- GDAL Python bindings (can be installed with Conda).

Outputs
-------
- VRT and projected GeoTIFF file for each band.
"""

# ---- Standard imports.
from pathlib import Path

# ---- Third party imports.
from osgeo import gdal

# ---- Local imports.
from sahel import __datadir__
from sahel.gishelpers import get_dem_filepaths

gdal.UseExceptions()

dst_crs = 'ESRI:102022'
proj_name = 'Africa Albers Equal Area Conic'
pixel_size = 30  # 1 arc-second is ~30 m at the equator

dem_dir = Path(__datadir__) / "dem"
src_paths = get_dem_filepaths(dem_dir / "raw")

vrt_hgt_path = dem_dir / 'dem_mosaic_hgt.vrt'
print('Generating mosaic for HGT band...')
if not vrt_hgt_path.exists():
    ds = gdal.BuildVRT(
        vrt_hgt_path,
        src_paths,
        options=gdal.BuildVRTOptions(bandList=[1])
        )
    ds.FlushCache()
    del ds
print(f'-> {vrt_hgt_path}\n')

proj_hgt_path = dem_dir / 'projected_mosaic_hgt.tif'
print(f'Projecting HGT data to {proj_name}...')
if not proj_hgt_path.exists():
    out_ds = gdal.Warp(
        proj_hgt_path,
        vrt_hgt_path,
        dstSRS='ESRI:102022',
        xRes=30, yRes=30, resampleAlg='bilinear',
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES']
        )
    out_ds.FlushCache()
    del out_ds
print(f'-> {proj_hgt_path}\n')

vrt_swb_path = dem_dir / 'dem_mosaic_swb.vrt'
print('Generating mosaic for SWB band...')
if not vrt_swb_path.exists():
    ds = gdal.BuildVRT(
        vrt_swb_path,
        src_paths,
        options=gdal.BuildVRTOptions(bandList=[2])
        )
    ds.FlushCache()
    del ds
print(f'-> {vrt_swb_path}\n')

proj_swb_path = dem_dir / 'projected_mosaic_swb.tif'
print(f'Projecting SWB data to {proj_name}...')
if not proj_swb_path.exists():
    out_ds = gdal.Warp(
        proj_swb_path,
        vrt_swb_path,
        dstSRS='ESRI:102022',
        xRes=30, yRes=30, resampleAlg='nearest',
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES']
    )
    out_ds.FlushCache()
    del out_ds
print(f'-> {proj_swb_path}\n')

print('All mosaicking and projection steps completed successfully.')
