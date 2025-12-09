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
import numpy as np
from osgeo import gdal
import rasterio
from scipy.ndimage import distance_transform_edt

# ---- Local imports
from sahel import __datadir__ as datadir
from sahel.gishelpers import create_pyramid_overview

gdal.UseExceptions()

adf_folder_path = (
    datadir / 'climate_zones' / 'hess-11-1633-2007-supplement' /
    'Raster files' / 'world_koppen')

dem_path = datadir / 'dem' / 'nasadem_102022.vrt'

output_path = datadir / 'climate_zones' / 'world_koppen.tiff'

# %%

print("Assign CRS to the Köppen raster (most global datasets are WGS84)...")

temp_with_crs = output_path.parent / 'koppen_with_crs.tif'

translate_options = gdal.TranslateOptions(
    format='GTiff',
    outputSRS='EPSG:4326',
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )
ds = gdal.Translate(
    str(temp_with_crs),
    str(adf_folder_path),
    options=translate_options
    )
ds.FlushCache()
del ds

# Read DEM to get target grid parameters.
with rasterio.open(dem_path) as dem:
    target_crs = dem.crs
    target_transform = dem.transform
    target_width = dem.width
    target_height = dem.height
    target_bounds = dem.bounds
    dem_nodata = dem.nodata

# %%

print("Warp Köppen raster to match DEM grid...")

temp_warped = output_path.parent / 'koppen_warped_temp.tif'

warp_options = gdal.WarpOptions(
    format='GTiff',
    dstSRS=str(target_crs),
    xRes=target_transform.a,
    yRes=abs(target_transform.e),
    outputBounds=(target_bounds.left, target_bounds.bottom,
                  target_bounds.right, target_bounds.top),
    width=target_width,
    height=target_height,
    resampleAlg='near',  # Use 'near' for categorical data like climate zones
    outputType=gdal.GDT_Byte,  # Use Byte (0-255) for climate zones
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )

ds = gdal.Warp(str(temp_warped), str(temp_with_crs), options=warp_options)
ds.FlushCache()
del ds

# %%

print("Fill gaps using nearest neighbor where DEM has data...")

with rasterio.open(dem_path) as dem:
    dem_data = dem.read(1)
    dem_mask = dem_data != dem_nodata
    dem_nodata_mask = dem_data == dem_nodata

with rasterio.open(temp_warped) as src:
    koppen_data = src.read(1)
    profile = src.profile

koppen_nodata_mask = (koppen_data == 255)
needs_filling = koppen_nodata_mask & dem_mask

if np.any(needs_filling):
    # Find nearest valid Köppen value for each nodata pixel.
    valid_mask = ~koppen_nodata_mask
    indices = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True)

    koppen_filled = koppen_data[tuple(indices)]

    koppen_data = np.where(needs_filling, koppen_filled, koppen_data)

# Set Köppen value to nodata where dem is nodata:
koppen_data[dem_nodata_mask] = 255

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(koppen_data, 1)

# Clean up temporary files.
temp_with_crs.unlink()
temp_warped.unlink()

print("Creating a pyramid overview for the  river network...")
create_pyramid_overview(output_path)
