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
Module for downloading and converting NASADEM DEM .hgt tiles to GeoTIFF
format for the Sahel region.

To use this script, you must have a valid NASA Earthdata account. You will be
prompted to provide your Earthdata username and password for authentication.
You can create an account for free at: https://urs.earthdata.nasa.gov/

This script automates the process of acquiring, extracting, and converting
high-resolution Digital Elevation Model (DEM) data from NASA's NASADEM
dataset

see https://github.com/geo-stack/sahel/pull/5


Main Features
-------------
- Defines a latitude/longitude bounding box covering all countries of the
  Sahel region of interest.
- Generates the list of NASADEM tile filenames needed to cover this region.
- Downloads each DEM tile as a ZIP archive directly from the NASA Earthdata/
  USGS MEASURES service, using secure authentication via the `earthaccess`
  library.
- Extracts the .hgt DEM file directly from the ZIP using GDAL's virtual file
  system (no need for manual unzipping).
- Converts each .hgt file to a compressed GeoTIFF (default: ZSTD compression)
  using rasterio, for ease of use in GIS and scientific workflows.
- Stores the resulting GeoTIFFs in the 'data/dem/Global/hgt' directory within
  your local Sahel project.
- Generate a GDAL virtual raster (VRT) mosaic of all DEM GeoTIFFs.

References
----------
- NASADEM project: https://www.earthdata.nasa.gov/about/competitive-programs/
  measures/new-nasa-digital-elevation-model
- USGS Earthdata: https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/
- https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-hgt-001
"""

# ---- Standard imports.
import os
import os.path as osp

# ---- Third party imports.
import earthaccess
from earthaccess.exceptions import LoginAttemptFailure
import numpy as np
import keyring
from osgeo import gdal

# ---- Local imports.
from sahel import __datadir__, CONF
from sahel.gishelpers import get_dem_filepaths, multi_convert_hgt_to_geotiff

# Define longitude and latitude ranges (covering West Africa)
LON_MIN = -19
LON_MAX = 25
LAT_MIN = 5
LAT_MAX = 29


blocklisted_tiles = [
    'NASADEM_HGT_n27w015.zip',
    'NASADEM_HGT_n27w016.zip',
    'NASADEM_HGT_n27w017.zip',
    'NASADEM_HGT_n27w018.zip',
    'NASADEM_HGT_n27w019.zip',
    'NASADEM_HGT_n28w014.zip',
    'NASADEM_HGT_n28w015.zip',
    'NASADEM_HGT_n28w016.zip',
    'NASADEM_HGT_n28w017.zip',
    'NASADEM_HGT_n28w018.zip',
    'NASADEM_HGT_n28w019.zip',
    'NASADEM_HGT_n29w014.zip',
    ]


# Prepare output directory.
dest_dir = osp.join(__datadir__, 'dem', 'raw', 'hgt')
os.makedirs(dest_dir, exist_ok=True)

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
# Set Earthdata credentials and login securely.

# Try to get credentials from config and keyring or prompt user
# if missing.

earthdata_username = CONF.get('main', 'earthdata_username', None)
earthdata_password = keyring.get_password("earthdata", earthdata_username)

if earthdata_username is None or earthdata_password is None:
    earthdata_username = input("Earthdata username: ")
    if not earthdata_username:
        raise ValueError(
            "No Earthdata username provided. Please rerun and enter "
            "your credentials."
            )

    earthdata_password = input("Earthdata password: ")
    if not earthdata_password:
        raise ValueError(
            "No Earthdata password provided. Please rerun and "
            "enter your credentials."
            )


# Try logging in to Earthdata and store credentials for next time.

os.environ["EARTHDATA_USERNAME"] = earthdata_username
os.environ["EARTHDATA_PASSWORD"] = earthdata_password
try:
    earthaccess.login()
except LoginAttemptFailure:
    raise LoginAttemptFailure(
        "Earthdata login failed. Please check your credentials and try again."
        )
else:
    CONF.set('main', 'earthdata_username', earthdata_username)
    keyring.set_password("earthdata", earthdata_username, earthdata_password)

print('Authentication with NASA Earthdata was successful.')


# %%

# Get the list of available tile names from the NASADEM dataset.

granules = earthaccess.search_data(
    short_name="NASADEM_HGT",
    version="001",
    temporal="2000-02-11",
    cloud_hosted=False
)
avail_zip_names = [
    granule['meta']['native-id'] + '.zip' for granule in granules
    ]


# Download the NASADEM tiles.

missing_tiles = []
base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/"
for i, zip_name in enumerate(zip_names):
    if zip_name in blocklisted_tiles:
        print(f'Skipping tile {i + 1} of {len(zip_names)} '
              f'because it is blocklisted...')
        continue
    if zip_name not in avail_zip_names:
        print(f'Skipping tile {i + 1} of {len(zip_names)} '
              f'because it does not exist...')
        continue

    print(f'Processing tile {i + 1} of {len(zip_names)}...')

    url = base_url + zip_name
    zip_filepath = osp.join(dest_dir, zip_name)

    # Skip if tile already downloaded.
    if osp.exists(zip_filepath):
        continue

    # Download the ZIP file and convert to GeoTIFF.
    try:
        earthaccess.download(
            url, osp.dirname(zip_filepath), show_progress=False)
    except Exception:
        print(f'Failed to download DEM data for tile {i + 1} ({zip_name}).')
        missing_tiles.append(zip_name)


# %%

# Convert hgt files to GeoTiff.

count = 0
progress = 0

zip_fpaths = []
tif_fpaths = []
for i, zip_name in enumerate(zip_names):
    zip_fpath = osp.join(dest_dir, zip_name)
    zip_fpaths.append(zip_fpath)

    root, _ = osp.splitext(osp.basename(zip_fpath))
    tif_fpaths.append(
        osp.join(osp.dirname(dest_dir), root + '.tif'))

multi_convert_hgt_to_geotiff(zip_fpaths, tif_fpaths)


# %%

# Generate a GDAL virtual raster (VRT) mosaic of all DEM GeoTIFFs.

dem_filepaths = get_dem_filepaths(osp.dirname(dest_dir))

vrt_filename = osp.join(__datadir__, 'dem', "dem_mosaic.vrt")
ds = gdal.BuildVRT(vrt_filename, dem_filepaths)
ds.FlushCache()
del ds

print(f'Virtual dataset generated at {vrt_filename}.')
