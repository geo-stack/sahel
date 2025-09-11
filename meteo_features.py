# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2024 (C) Aziz Agrebi
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# Originally developed by Aziz Agrebi as part of his master's project.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel
# =============================================================================

"""Création des features météorologiques."""

# Standard imports.
import os
import os.path as osp
from datetime import datetime, timedelta

# Third party imports.
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Local imports.
from sahel import __datadir__
from sahel.utils import read_obs_wl


def duration(area_km2):
    """
    Constante de temps associée à différentes tailles de bassin versant.
    """
    if area_km2 < 100:
        return 30
    elif area_km2 < 500:
        return 60
    elif area_km2 < 1000:
        return 90
    elif area_km2 < 3000:
        return 120
    elif area_km2 < 5000:
        return 150
    elif area_km2 < 10000:
        return 180
    return 360


dem_countries = ["Benin", "Burkina", "Guinee", "Mali", "Niger", "Togo"]
date_methods = ["datetime", "str", "datetime", "datetime", "datetime", "int"]

dem_to_inference = {
    "Benin": "Benin",
    "Burkina": "BF",
    "Guinee": "gui",
    "Mali": "Mali",
    "Niger": "Niger",
    "Togo": "Togo",
}

training_num = 3
dem_country = dem_countries[training_num]
inference_country = dem_to_inference[dem_country]


# %%
training_df = read_obs_wl(osp.join(__datadir__, 'data', f'{dem_country}.xlsx'))

# On filtre les dates supérieures à 2002 (car pas de données sur ee avant ça).
mask = ((training_df['DATE'].dt.year > 2002) &
        (training_df['DATE'].dt.year < 2025))

training_df = training_df.loc[mask, :]

print('Nbr. of WL observations :', len(training_df))


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

# Hydrobassin qui permet de délimiter les bassins
# versants (niveau 12, i.e. résolution max)
hydrobasins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_12')

# Dataset pour la pluviométrie.
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

# Dataset pour le NDVI
modis_ndvi = ee.ImageCollection('MODIS/006/MOD13A1').select("NDVI")

SAVE_PATH = f"Meteo_features_{dem_country}_time_series.csv"


def get_time_series(lon, lat, date_str):
    if type(date_str) is str:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            date = datetime.strptime(date_str, "%d/%m/%Y")
    else:
        date = date_str

    start_date = date - timedelta(days=150)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = date.strftime('%Y-%m-%d')

    point = ee.Geometry.Point([lon, lat])
    bassin = hydrobasins.filterBounds(point).first()
    if bassin is None:
        return None, None

    bassin_geom = bassin.geometry()

    chirps_filtered = chirps.filterDate(start_date_str, end_date_str)

    def extract_rainfall(img):
        date = ee.Date(img.date()).format("YYYY-MM-dd")
        mean_rainfall = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bassin_geom,
            scale=5000,
            maxPixels=1e9
        ).get("precipitation")
        return ee.Feature(None, {"date": date, "precipitation": mean_rainfall})

    rainfall_series = (
        chirps_filtered.map(extract_rainfall)
        .aggregate_array("date")
        .getInfo())
    rainfall_values = (
        chirps_filtered.map(extract_rainfall)
        .aggregate_array("precipitation")
        .getInfo())

    ndvi_filtered = modis_ndvi.filterDate(start_date_str, end_date_str)

    def extract_ndvi(img):
        date = ee.Date(img.date()).format("YYYY-MM-dd")
        mean_ndvi = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bassin_geom,
            scale=500,
            maxPixels=1e9
        ).get("NDVI")
        return ee.Feature(None, {"date": date, "NDVI": mean_ndvi})

    ndvi_series = (
        ndvi_filtered.map(extract_ndvi)
        .aggregate_array("date")
        .getInfo())
    ndvi_values = (
        ndvi_filtered.map(extract_ndvi)
        .aggregate_array("NDVI")
        .getInfo())

    ndvi_values = [v * 0.0001 if v is not None else None for v in ndvi_values]

    return (list(zip(rainfall_series, rainfall_values)),
            list(zip(ndvi_series, ndvi_values)))


if osp.exists(SAVE_PATH):
    training_df = pd.read_csv(SAVE_PATH, index_col=0)
    start_index = (
        training_df[training_df["ndvi_series"] ==
                    training_df["ndvi_series"]].index[-1]
        )
    print(f"Reprise depuis l'index {start_index}")
else:
    training_df["precipitation_series"] = None
    training_df["ndvi_series"] = None
    start_index = -1

for k, i in enumerate(training_df.index):
    if i > start_index:
        lon = training_df.loc[i, "LON"]
        lat = training_df.loc[i, "LAT"]
        date_str = training_df.loc[i, "DATE"]

        print(f"Traitement de l'index {k}/{len(training_df)} : "
              f"LON={lon}, LAT={lat}, DATE={date_str}")

        precipitation_series, ndvi_series = get_time_series(lon, lat, date_str)

        training_df.at[i, "precipitation_series"] = str(precipitation_series)
        training_df.at[i, "ndvi_series"] = str(ndvi_series)

        if k % 10 == 0 or k == len(training_df) - 1:
            training_df.to_csv(SAVE_PATH, index=True)
            print(f"Sauvegarde à l'index {i}")
