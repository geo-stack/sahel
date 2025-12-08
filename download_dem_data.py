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
Script for downloading and converting NASADEM DEM .hgt tiles to GeoTIFF
format for the entire African continent.

To use this script, you must have a valid NASA Earthdata account. You will be
prompted to provide your Earthdata username and password for authentication.
You can create an account for free at: https://urs.earthdata.nasa.gov/

This script automates the process of acquiring, extracting, and converting
high-resolution Digital Elevation Model (DEM) data from NASA's NASADEM
dataset


Rationale for using NASADEM
---------------------------
The NASADEM dataset was selected for this project because it provides
high-resolution, globally available digital elevation data derived from
the Shuttle Radar Topography Mission (SRTM) and enhanced with additional
sources and improved processing. NASADEM improves upon the original SRTM
by offering more complete coverage, fewer voids, and enhanced vertical
accuracy, which is especially valuable for hydrological, geomorphological,
and groundwater modeling across large regions like the Sahel. Its native
resolution of 1 arc-second (~30 meters) is well-suited for regional-scale
environmental analyses, making it an ideal foundation for extracting
topographic features, surface water masks, and other terrain variables
critical to understanding and predicting groundwater resources and
surface water dynamics in West Africa.

see also https://github.com/geo-stack/hydrodepthml/pull/5


References
----------
- NASADEM project: https://www.earthdata.nasa.gov/about/competitive-programs/
  measures/new-nasa-digital-elevation-model
- USGS Earthdata: https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/
- https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-hgt-001
"""

# ---- Standard imports.
from math import floor, ceil

# ---- Third party imports.
import numpy as np
from osgeo import gdal
import geopandas as gpd

# ---- Local imports.
from hdml import __datadir__ as datadir
from hdml.gishelpers import (
    get_dem_filepaths, multi_convert_hgt_to_geotiff,
    )
from hdml.ed_helpers import earthaccess_login


print("Authenticating with NASA Earthdata...")
earthaccess = earthaccess_login()

# Define longitude and latitude ranges (covering the African continent)
africa_landmass = gpd.read_file(datadir / 'coastline' / 'africa_landmass.gpkg')
africa_landmass = africa_landmass.to_crs("EPSG:4326")
LON_MIN = floor(africa_landmass.bounds.minx[0]) - 1
LON_MAX = ceil(africa_landmass.bounds.maxx[0]) + 1
LAT_MIN = floor(africa_landmass.bounds.miny[0]) - 1
LAT_MAX = ceil(africa_landmass.bounds.maxy[0]) + 1


# Prepare output directory.
DEST_DIR = datadir / 'dem'
DEST_DIR.mkdir(exist_ok=True)

TIF_DIR = DEST_DIR / 'tif'
TIF_DIR.mkdir(exist_ok=True)

HGT_DIR = DEST_DIR / 'hgt'
HGT_DIR.mkdir(exist_ok=True)

# Generate NASADEM zip filenames for the specified tiling grid.
zip_names = []
for lat in np.arange(LAT_MIN, LAT_MAX + 1):
    for lon in np.arange(LON_MIN, LON_MAX + 1):
        zip_names.append(
            f"NASADEM_HGT_"
            f"{'n' if lat >= 0 else 's'}{abs(lat):02d}"
            f"{'w' if lon < 0 else 'e'}{abs(lon):03d}"
            ".zip")


# %%

# Get the list of available tile names from the NASADEM dataset.

granules = earthaccess.search_data(
    short_name="NASADEM_HGT",
    version="001",
    temporal="2000-02-11",
    cloud_hosted=False
    )

avail_zip_names = {}
for granule in granules:
    zip_name = granule['meta']['native-id'] + '.zip'
    url = granule['umm']['RelatedUrls'][0]['URL']
    avail_zip_names[zip_name] = url


# %%

# Download the NASADEM tiles.

missing_tiles = []
base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/"

zip_fpaths = []
tif_fpaths = []

ntot = len(zip_names)
for i, zip_name in enumerate(zip_names):
    progress = f"[{i+1:02d}/{ntot}]"
    if zip_name not in avail_zip_names:
        print(f'{progress} Skipping because tile is not valid...')
        missing_tiles.append(zip_name)
        continue

    url = base_url + zip_name
    zip_filepath = HGT_DIR / zip_name
    tif_filepath = (TIF_DIR / zip_name).with_suffix('.tif')

    # Skip if tile was already downloaded.
    if tif_filepath.exists():
        print(f'{progress} Skipping because tif file already exists...')
        continue
    if zip_filepath.exists() or tif_filepath.exists():
        print(f'{progress} Skipping because hgt file already exists...')
        zip_fpaths.append(zip_filepath)
        tif_fpaths.append(tif_filepath)
        continue

    print(f'{progress} Downloading DEM tile...')

    # Download the ZIP file.
    earthaccess.download(url, str(HGT_DIR), show_progress=False)

    zip_fpaths.append(zip_filepath)
    tif_fpaths.append(tif_filepath)

# %%

# Convert hgt files to GeoTiff.

print()
print('Converting HGT archives to geoTiff...')

if len(zip_fpaths) > 0:
    multi_convert_hgt_to_geotiff(zip_fpaths, tif_fpaths)


# %%

# Generate a GDAL virtual raster (VRT) mosaic of all DEM GeoTIFFs.
vrt_path = DEST_DIR / 'nasadem.vrt'
dem_paths = get_dem_filepaths(TIF_DIR)
ds = gdal.BuildVRT(vrt_path, dem_paths)
ds.FlushCache()
del ds

# Reprojected VRT and apply African landmass mask.

dst_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic
pixel_size = 30  # 1 arc-second is ~30 m at the equator

vrt_reprojected = DEST_DIR / 'nasadem_102022.vrt'
warp_options = gdal.WarpOptions(
    cutlineDSName=str(datadir / 'coastline' / 'africa_landmass.gpkg'),
    cropToCutline=False,
    dstSRS=dst_crs,
    format='VRT',
    resampleAlg='bilinear',
    xRes=pixel_size,
    yRes=pixel_size,
    multithread=True,
    )

ds_reproj = gdal.Warp(
    str(vrt_reprojected),
    str(vrt_path),
    options=warp_options
    )
ds_reproj.FlushCache()
del ds_reproj

print(f'Virtual dataset generated at {vrt_reprojected}.')
