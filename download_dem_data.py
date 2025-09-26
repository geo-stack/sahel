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
Download and convert NASADEM DEM .hgt tiles to GeoTIFF format.

This script:
- Generates a list of NASADEM SRTM tile filenames covering a specified
  lat/lon bounding box.
- Downloads each tile (as a ZIP) from the NASA Earthdata/USGS MEASURES archive,
  using earthaccess authentication.
- Extracts the .hgt file from the ZIP (using GDAL VFS for direct access).
- Converts the .hgt file to a compressed (ZSTD) GeoTIFF using rasterio.

See:
    https://www.earthdata.nasa.gov/about/competitive-programs/measures/
    new-nasa-digital-elevation-model
"""

# ---- Standard imports.
import os
import os.path as osp

# ---- Third party imports.
import earthaccess
from earthaccess.exceptions import LoginAttemptFailure
import numpy as np
import keyring

# ---- Local imports.
from sahel import __datadir__, CONF
from sahel.gishelpers import convert_hgt_to_geotiff


# Prepare output directory.
dest_dir = osp.join(__datadir__, 'dem', 'Global', 'hgt')
os.makedirs(dest_dir, exist_ok=True)

# Define longitude and latitude ranges (covering West Africa)
lon_min = -19
lon_max = 25
lat_min = 5
lat_max = 29

# Generate NASADEM zip filenames for the specified tiling grid.
zip_names = []
for lon in np.arange(lon_min, lon_max + 1):
    for lat in np.arange(lat_min, lat_max + 1):
        zip_names.append(
            f"NASADEM_HGT_"
            f"{'n' if lat >= 0 else 's'}{abs(lat):02d}"
            f"{'e' if lon <= 0 else 'w'}{abs(lon):03d}"
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
            'Please run this cell again and provide valid '
            'Earthdata credentials.'
            )

    earthdata_password = input("Earthdata password: ")
    if not earthdata_password:
        raise ValueError(
            'Please run this cell again and provide valid '
            'Earthdata credentials.'
            )


# Try logging in to Earthdata and store credentials for next time.

os.environ["EARTHDATA_USERNAME"] = earthdata_username
os.environ["EARTHDATA_PASSWORD"] = earthdata_password
try:
    earthaccess.login()
except LoginAttemptFailure:
    raise ValueError(
        'Please run this cell again and provide valid '
        'Earthdata credentials.'
        )
else:
    CONF.set('main', 'earthdata_username', earthdata_username)
    keyring.set_password("earthdata", earthdata_username, earthdata_password)

print('Authentication with NASA Earthdata was successful.')


# %%

base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/"
for i, zip_name in enumerate(zip_names):
    print(f'Processing tile {i + 1} of {len(zip_names)}...')

    url = base_url + zip_name
    zip_filepath = osp.join(dest_dir, zip_name)

    # Skip if tile already downloaded.
    if osp.exists(zip_filepath):
        continue

    # Download the ZIP file and convert to GeoTIFF.
    earthaccess.download(url, osp.dirname(zip_filepath), show_progress=False)
    convert_hgt_to_geotiff(zip_filepath, osp.dirname(dest_dir))

    break
