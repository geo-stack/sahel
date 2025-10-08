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

# ---- Standard import
import os
import os.path as osp

# ---- Third party import
import cdsapi
import keyring

# ---- Local import
from sahel import __datadir__, CONF

# %%

ecmwf_username = CONF.get('main', 'ecmwf_username', None)
ecmwf_key = keyring.get_password("ECMWF", ecmwf_username)


if ecmwf_username is None or ecmwf_key is None:
    ecmwf_username = input("ECMWF username: ")
    if not ecmwf_username:
        raise ValueError(
            "No ECMWF username provided. Please rerun and enter "
            "your credentials."
            )

    ecmwf_key = input("ECMWF API key: ")
    if not ecmwf_key:
        raise ValueError(
            "No ECMWF API key provided. Please rerun and "
            "enter your credentials."
            )

    CONF.set('main', 'ecmwf_username', ecmwf_username)
    keyring.set_password("ECMWF", ecmwf_username, ecmwf_key)

api_url = 'https://cds.climate.copernicus.eu/api'
with open(osp.join(osp.expanduser('~'), '.cdsapirc'), mode='w') as fp:
    fp.write(f"url: {api_url}\nkey: {ecmwf_key}\n")


# %%

LON_MIN = -19
LON_MAX = 25
LAT_MIN = 5
LAT_MAX = 29

target_dir = osp.join(__datadir__, 'meteo', 'airtemp_2m')
os.makedirs(target_dir, exist_ok=True)

client = cdsapi.Client()

target_file = osp.join(target_dir, 'airtemp_monthly_2000.grib')

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["2m_temperature"],
    "year": ["2000"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]
}

client.retrieve(dataset, request).download(target=target_file)

# %%
import netCDF4
import numpy as np

dset = netCDF4.Dataset(target_file)

np.array(dset['longitude'])
np.array(dset['latitude'])
t2m = np.array(dset['t2m'])


import matplotlib.pyplot as plt

plt.plot(t2m[:, 1, 1] - 273.15)
