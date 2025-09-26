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
from sahel.gishelpers import convert_hgt_to_geotiff, get_dem_filepaths

# Define longitude and latitude ranges (covering West Africa)
LON_MIN = -19
LON_MAX = 25
LAT_MIN = 5
LAT_MAX = 29


# Prepare output directory.
dest_dir = osp.join(__datadir__, 'dem', 'Global', 'hgt')
os.makedirs(dest_dir, exist_ok=True)

# Generate NASADEM zip filenames for the specified tiling grid.
zip_names = []
for lon in np.arange(LON_MIN, LON_MAX + 1):
    for lat in np.arange(LAT_MIN, LAT_MAX + 1):
        zip_names.append(
            f"NASADEM_HGT_"
            f"{'n' if lat >= 0 else 's'}{abs(lat):02d}"
            f"{'w' if lon <= 0 else 'e'}{abs(lon):03d}"
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

missing_tiles = []

base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/"
for i, zip_name in enumerate(zip_names):
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
    else:
        convert_hgt_to_geotiff(zip_filepath, osp.dirname(dest_dir))


# %%

dem_filepaths = get_dem_filepaths(osp.dirname(dest_dir))

ds = gdal.BuildVRT(
    osp.join(__datadir__, 'dem', "Global.vrt"),
    dem_filepaths)
ds.FlushCache()
del ds
