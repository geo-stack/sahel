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

"""Download climatic data ."""

# Standard imports.
import os
import os.path as osp

# Third party imports.
import numpy as np
import netCDF4
import requests
from bs4 import BeautifulSoup
import rasterio
from rasterio.transform import from_origin

# Local imports.
from sahel import __datadir__


# %% Download Monthly Precip Data

# CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
# is a gridded rainfall dataset providing daily, pentadal, and monthly
# precipitation estimates at ~0.05° resolution, starting from 1981. It
# combines satellite imagery with in-situ station data to support climate
# and drought monitoring applications, especially in data-scarce regions.

# Here we download all monthly CHIRPS precipitation GeoTIFF files for the
# entire African continent from 1981 to today (one .tif file per month) from
# the official UCSB CHC data server.

# See https://www.chc.ucsb.edu/data/chirps.

dest_folder = osp.join(__datadir__, 'chirps')
os.makedirs(dest_folder, exist_ok=True)

base_url = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/"
    )

# Get the list of monthly tif files available for download.
resp = requests.get(base_url)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")
files = [a['href'] for a in soup.find_all("a") if a['href'].endswith(".tif")]

for file in files:
    file_url = base_url + file
    local_path = osp.join(dest_folder, file)

    # Skip if already downloaded.
    if osp.exists(local_path):
        print(f"Skipping {file}, already exists.")
        continue

    try:
        resp = requests.get(file_url, stream=False, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as fp:
            fp.write(resp.content)
    except Exception as e:
        print(f"Could not download {file}: {e}")
    else:
        print(f"Downloaded {file} sucessfully.")


# %% Download HydroATLAS layers

# HydroATLAS is a global, sub-basin hydrology dataset providing physical,
# ecological, climatic and socio-economic attributes for river basins and
# catchments at high resolution. It offers standardized basin geometries
# (HydroBASINS) and >700 attributes (e.g. flow accumulation, land cover,
# precipitation) for multiple nested basin levels to support water resources
# management, modeling, and environmental assessment.

# Here we download all three layers of HydroATLAS (BasinATLAS, RiverATLAS,
# and LakeATLAS) in geodatabase format.

# See https://www.hydrosheds.org/products/hydrobasins.

dest_folder = osp.join(__datadir__, 'hydro_atlas')
os.makedirs(dest_folder, exist_ok=True)

urls = {'BasinATLAS': 'https://figshare.com/ndownloader/files/20082137',
        'RiverATLAS': 'https://figshare.com/ndownloader/files/20087321',
        'LakeATLAS': 'https://figshare.com/ndownloader/files/35959544'}

for key, url in urls.items():

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        # Get the filename from the response headers.
        cd = r.headers.get("Content-Disposition", "")
        filename = cd.split("filename=")[-1].strip('"')

        local_path = osp.join(dest_folder, filename)

        # Skip if already downloaded.
        if osp.exists(local_path):
            print(f"Skipping {filename}, already exists.")
            continue

        print(f"Downloading {filename}...")
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {file} sucessfully.")


# %% Download NOAA NDVI data

# MODIS NDVI (MOD13A1) data from NASA EarthData is only available since the
# year 2000 because the MODIS sensors aboard Terra and Aqua satellites were
# launched in late 1999 and 2002, respectively. Therefore we are using
# NDVI data from the NOAA Climate Data Record (CDR) instead.

# The NOAA Climate Data Record (CDR) of the Normalized Difference Vegetation
# Index (NDVI) provides a long-term, consistent record of global vegetation
# greenness derived from AVHRR satellite observations, spanning from 1981 to
# present.

# see: https://www.ncei.noaa.gov/products/climate-data-records

from sahel.dataio.noaa import (
    download_noaa_ndvi_daily, stack_daily_ndvi_to_month,
    calc_ndvi_monthly_stats
    )

year = 1981

base_datadir = osp.join(__datadir__, 'noaa_ndvi')
daily_datadir = osp.join(base_datadir, 'daily', str(year))

download_noaa_ndvi_daily(year, daily_datadir)

lat_min = -40.0
lat_max = 40.0
lon_min = -20.0
lon_max = 55.0

for month in range(1, 13):

    print(f"Processing daily NDVI data for month {month} of year {year}...")
    ndvi_mth = stack_daily_ndvi_to_month(
        year=year, month=month, datadir=daily_datadir,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max
        )

    if ndvi_mth is None:
        print(f"There is no daily NDVI data for month {month} of "
              f"year {year}, skipping.")
        continue

    print(f"Calculating monthly NDVI stats for "
          f"month {month} of year {year}...")
    mth_stats, band_names = calc_ndvi_monthly_stats(ndvi_mth)

    print(f"Write monthly stats to GeoTIFF for "
          f"month {month} of year {year}...")

    filename = f'noaa_ndvi_monthly_{year}_{month:0.2d}.tif'
    filepath = osp.join(__datadir__, 'noaa_ndvi', filename)

    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=mth_stats.shape[1],
        width=mth_stats.shape[2],
        count=mth_stats.shape[0],
        dtype=mth_stats.dtype,
        crs="EPSG:4326",  # lat/lon grids (WGS84)
        transform=from_origin(lon_min, lat_max, 0.05, 0.05),
        compress='zstd'
    ) as dst:
        for i, (band_name, band_values) in enumerate(mth_stats.items):
            dst.write(band_values, i + 1)
            dst.set_band_description(i + 1, band_name)

    break
