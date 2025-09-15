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

"""Download climatic data from NOAA Climate Data Record (CDR)"""

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


BASE_NOAA_NDVI_URL = (
    "https://www.ncei.noaa.gov/data/land-normalized-difference-"
    "vegetation-index/access/"
    )


def get_noaa_ndvi_avail_years() -> list[int]:
    """
    Retrieve the list of available years for NOAA NDVI data from
    the remote server.

    Returns
    -------
    years : list of str
        List of available years as strings.
    """
    resp = requests.get(BASE_NOAA_NDVI_URL)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    years = [a.text.strip('/') for a in soup.find_all("a") if
             a.text.strip('/').isdigit()]
    return years


def download_noaa_ndvi_daily(year: int, dest_folder: str):
    """
    Download daily NOAA NDVI NetCDF files for a given year into a
    specified folder.

    Parameters
    ----------
    year : int
        The year for which NDVI data should be downloaded (e.g., 2022).
    dest_folder : str
        The path to the local directory where the downloaded files should be
        stored. The folder will be created if it does not already exist.
    """
    if year not in get_noaa_ndvi_avail_years():
        print(f'There is nothing to download for year {year}.')
        return

    os.makedirs(dest_folder, exist_ok=True)

    # Get filenames available for download for current year.
    year_url = BASE_NOAA_NDVI_URL + f'{year}/'

    resp = requests.get(year_url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = [a['href'] for a in soup.find_all("a") if
             a['href'].endswith(".nc")]

    # Download data files.
    for file in files:
        local_path = osp.join(dest_folder, file)

        # Skip if already downloaded.
        if osp.exists(local_path):
            print(f"Skipping {file}, already exists.")
            continue

        file_url = year_url + f'{file}'
        resp = requests.get(file_url, stream=False, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as fp:
            fp.write(resp.content)

        print(f"Downloaded {file} sucessfully.")


def stack_daily_ndvi_to_month(
        year: int, month: int, datadir: str,
        lat_min: float = -np.inf, lat_max: float = np.inf,
        lon_min: float = -np.inf, lon_max: float = np.inf
        ):
    """
    Load and concatenate daily NDVI data for a given month and spatial subset.

    This function searches a directory for daily NDVI NetCDF files
    matching the specified year and month, extracts the NDVI data within
    the provided latitude and longitude bounds, and stacks all daily arrays
    into a single NumPy array.

    Parameters
    ----------
    year : int
        The year for which to load NDVI data (e.g., 1981).
    month : int
        The month for which to load NDVI data (1-12).
    datadir : str
        Path to the directory containing daily NDVI NetCDF files.
    lat_min : float
        Minimum latitude (southern boundary) of the spatial subset (degrees).
    lat_max : float
        Maximum latitude (northern boundary) of the spatial subset (degrees).
    lon_min : float
        Minimum longitude (western boundary) of the spatial subset (degrees).
    lon_max : float
        Maximum longitude (eastern boundary) of the spatial subset (degrees).

    Returns
    -------
    ndvi_mth : np.ndarray
        A 3D NumPy array of shape (n_days, n_lat, n_lon) containing the
        stacked daily NDVI values for the specified month and region.
        Missing values (originally -9999) are replaced with np.nan.
    """
    idx_x = None
    idx_y = None

    filenames = os.listdir(datadir)

    arrays = []
    for day in range(1, 32):
        for filename in filenames:
            if f"{year}{month:02d}{day:02d}" in filename:
                print('day:', day)
                dset = netCDF4.Dataset(osp.join(datadir, filename))
                if idx_x is None:
                    lat = np.array(dset['latitude'])
                    lon = np.array(dset['longitude'])

                    idx_x = np.nonzero((lon_min < lon) & (lon < lon_max))[0]
                    idx_y = np.nonzero((lat_min < lat) & (lat < lat_max))[0]

                arrays.append(
                    np.array(dset['NDVI'])[:, idx_y, :][:, :, idx_x].copy()
                    )

    ndvi_mth = np.concatenate(arrays)
    ndvi_mth[ndvi_mth == -9999] = np.nan

    return ndvi_mth
