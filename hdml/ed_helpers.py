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


def MOD13Q1_hdf_to_geotiff(
        hdf_file: Path, subdataset_index: int, output_file: Path
        ):
    """
    Convert MODIS HDF subdataset to GeoTIFF.

    Need to install:
        > conda install -c conda-forge libgdal-hdf4

    Parameters
    ----------
    hdf_file : str or Path
        Path to HDF file.
    subdataset_index : int
        Index of subdataset to extract (0-based).
    output_file : str or Path, optional
        Output GeoTIFF path. If None, auto-generate from input name.
    """
    hdf_file = Path(hdf_file)

    # Open HDF and get subdatasets
    hdf_ds = gdal.Open(str(hdf_file), gdal.GA_ReadOnly)

    subdatasets = hdf_ds.GetSubDatasets()

    # Open subdataset and get metadata.
    subdataset_name = subdatasets[subdataset_index][0]
    sds = gdal.Open(subdataset_name)

    rangebeginningdate = sds.GetMetadata()['RANGEBEGINNINGDATE']
    rangeendingdate = sds.GetMetadata()['RANGEENDINGDATE']
    verticaltilenumber = sds.GetMetadata()['VERTICALTILENUMBER']
    horizontaltilenumber = sds.GetMetadata()['HORIZONTALTILENUMBER']
    long_name = sds.GetMetadata()['long_name']
    metadata = sds.GetMetadata().copy()
    sds = None

    assert long_name == '250m 16 days NDVI'

    # Convert to GeoTIFF
    gdal.Translate(
        str(output_file),
        subdataset_name,
        format='GTiff',
        creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
        )

    hdf_ds = None

    tile_name = f'h{horizontaltilenumber}v{verticaltilenumber}'

    return tile_name, rangebeginningdate, rangeendingdate, metadata
