# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel_water_table_ml
# =============================================================================

# ---- Standard imports
import os
from pathlib import Path

# ---- Third party imports
import keyring
import earthaccess
from earthaccess.exceptions import LoginAttemptFailure
from osgeo import gdal
import rasterio

# ---- Local imports.
from hdml import CONF

gdal.UseExceptions()

"""
ed_helpers.py

Helper utilities for the 'earthdata' plateform.
"""


def earthaccess_login():
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
        earthdata_username = CONF.set('main', 'earthdata_username', None)
        raise LoginAttemptFailure(
            "Earthdata login failed. Please check your credentials and "
            "try again."
            )
    else:
        CONF.set('main', 'earthdata_username', earthdata_username)
        keyring.set_password(
            "earthdata", earthdata_username, earthdata_password
            )

    print('Authentication with NASA Earthdata was successful.')
    return earthaccess


def get_MOD13Q1_hdf_metadata(hdf_file: Path, subdataset_index: int = 0):
    """
    Extract metadata from a MODIS MOD13Q1 HDF subdataset.

    Parameters
    ----------
    hdf_file : Path
        Path to the MODIS HDF file (e.g., MOD13Q1.A2020001.h21v08.*. hdf).
    subdataset_index :  int, optional
        Index of the subdataset to read (0-based). Default is 0.
        MOD13Q1 typically contains multiple subdatasets:
        - 0: 250m 16 days NDVI
        - 1: 250m 16 days EVI
        - 2: 250m 16 days VI Quality
        - etc.

    Returns
    -------
    dict
        Dictionary containing subdataset metadata with keys such as:
        - 'RANGEBEGINNINGDATE': Start date of the 16-day composite
        - 'RANGEENDINGDATE': End date of the 16-day composite
        - 'HORIZONTALTILENUMBER':  MODIS tile h-coordinate
        - 'VERTICALTILENUMBER': MODIS tile v-coordinate
        - 'long_name': Description of the variable
        - 'subdataset_name': Full GDAL subdataset identifier string

    Notes
    -----
    Requires GDAL compiled with HDF4 support. Install via:
        conda install -c conda-forge libgdal-hdf4
    """
    # Open HDF file and get list of subdatasets
    hdf_ds = gdal.Open(str(hdf_file), gdal.GA_ReadOnly)

    subdatasets = hdf_ds.GetSubDatasets()
    hdf_ds = None

    # Open subdataset and get metadata.
    subdataset_name = subdatasets[subdataset_index][0]
    sds = gdal.Open(subdataset_name)

    metadata = sds.GetMetadata().copy()
    metadata['subdataset_name'] = subdataset_name

    verticaltilenumber = metadata['VERTICALTILENUMBER']
    horizontaltilenumber = metadata['HORIZONTALTILENUMBER']
    metadata['tile_name'] = f'h{horizontaltilenumber}v{verticaltilenumber}'

    sds = None

    return metadata


def MOD13Q1_hdf_to_geotiff(
        hdf_file: Path, subdataset_index: int, output_file: Path
        ):
    """
    Convert a MODIS MOD13Q1 HDF subdataset to GeoTIFF format.

    Extracts the specified subdataset from a MODIS MOD13Q1 HDF file,
    converts it to GeoTIFF with compression, and returns metadata about
    the tile and temporal coverage.

    Parameters
    ----------
    hdf_file : Path
        Path to the MODIS MOD13Q1 HDF file.
    subdataset_index : int
        Index of subdataset to extract (0-based). Use 0 for 250m 16-day NDVI.
    output_file : Path
        Output GeoTIFF file path where the converted raster will be saved.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing:
        - tile_name :  str
            MODIS tile identifier (e.g., 'h21v08')
        - rangebeginningdate : str
            Start date of the 16-day composite (YYYY-MM-DD)
        - rangeendingdate :  str
            End date of the 16-day composite (YYYY-MM-DD)
    """
    metadata = get_MOD13Q1_hdf_metadata(str(hdf_file))

    # Validate subdataset type.
    long_name = metadata['long_name']
    assert long_name == '250m 16 days NDVI', (
        f"Expected '250m 16 days NDVI' but got '{long_name}'.  "
        f"Check subdataset_index or update validation logic."
        )

    # Convert to GeoTIFF with compression.
    result = gdal.Translate(
        str(output_file),
        metadata['subdataset_name'],
        format='GTiff',
        creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
        )
    del result

    with rasterio.open(output_file, 'r+') as ds:
        ds.update_tags(
            tile_name=metadata['tile_name'],
            date_start=metadata['RANGEBEGINNINGDATE'],
            date_end=metadata['RANGEENDINGDATE']
            )

    return (metadata['tile_name'],
            metadata['RANGEBEGINNINGDATE'],
            metadata['RANGEENDINGDATE'])
