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

import os
import os.path as osp
import pandas as pd
from datetime import datetime, timedelta
import ee  # L'API de Google Earth Engine
ee.Initialize(project="ee-azizagrebi4")

from sahel import __datadir__
from sahel.utils import read_obs_wl


# %%

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
training_df = read_obs_wl(osp.join(__datadir__, f'{dem_country}.xlsx'))

# On filtre les dates supérieures à 2002 (car pas de données sur ee avant ça).
mask = ((training_df['DATE'].dt.year > 2002) &
        (training_df['DATE'].dt.year < 2025))

training_df = training_df.loc[mask, :]

print('Nbr. of WL observations :', len(training_df))


# %%

def duration(area_km2): # Constante de temps associée à différentes tailles de bassin versant
    if area_km2 < 100:
        return 30
    elif area_km2 < 500:
        return 60
    elif area_km2 < 1000:
        return 90
    elif area_km2 < 3000 :
        return 120
    elif area_km2 < 5000:
        return 150
    elif area_km2 < 10000:
        return 180
    return 360


# In[ ]:


# On crée ici les features météorologiques

hydrobasins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_12') # Hydrobassin qui permet de délimiter les bassins versants (niveau 12, i.e. résolution max)
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") # Dataset pour la pluviométrie
modis_ndvi = ee.ImageCollection('MODIS/006/MOD13A1').select("NDVI") # Dataset pour le NDVI

SAVE_PATH = f"Meteo_features_{dem_country}_time_series.csv"

def get_time_series(lon, lat, date_str):
    if type(date_str)==str:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            date = datetime.strptime(date_str, "%d/%m/%Y")
    else:
        date = date_str

    start_date = date - timedelta(days=150)
    start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')

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

    rainfall_series = chirps_filtered.map(extract_rainfall).aggregate_array("date").getInfo()
    rainfall_values = chirps_filtered.map(extract_rainfall).aggregate_array("precipitation").getInfo()

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

    ndvi_series = ndvi_filtered.map(extract_ndvi).aggregate_array("date").getInfo()
    ndvi_values = ndvi_filtered.map(extract_ndvi).aggregate_array("NDVI").getInfo()

    ndvi_values = [v * 0.0001 if v is not None else None for v in ndvi_values]

    return list(zip(rainfall_series, rainfall_values)), list(zip(ndvi_series, ndvi_values))

if os.path.exists(SAVE_PATH):
    training_df = pd.read_csv(SAVE_PATH, index_col=0)
    start_index = training_df[training_df["ndvi_series"]==training_df["ndvi_series"]].index[-1]
    print(f"Reprise depuis l'index {start_index}")
else:
    training_df["precipitation_series"] = None
    training_df["ndvi_series"] = None
    start_index = -1

for k, i in enumerate(training_df.index):
    if i > start_index:
        lon, lat, date_str = training_df.loc[i, "LON"], training_df.loc[i, "LAT"], training_df.loc[i, "DATE"]
        
        print(f"Traitement de l'index {k}/{len(training_df)} : LON={lon}, LAT={lat}, DATE={date_str}")

        precipitation_series, ndvi_series = get_time_series(lon, lat, date_str)
        
        training_df.at[i, "precipitation_series"] = str(precipitation_series)
        training_df.at[i, "ndvi_series"] = str(ndvi_series)

        if k % 10 == 0 or k == len(training_df) - 1:
            training_df.to_csv(SAVE_PATH, index=True)
            print(f"Sauvegarde à l'index {i}")

