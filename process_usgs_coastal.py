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

# You need to download the 'USGSEsriWCMC_GlobalIslands_v3_mpk.zip'
# archive from:
# https://www.sciencebase.gov/catalog/item/63bdf25dd34e92aad3cda273
# and copy it to the folder './sahel/data/coastline'

# See also:
# https://data.usgs.gov/datacatalog/data/USGS:63bdf25dd34e92aad3cda273
# https://pubs.usgs.gov/publication/70202401


# ---- Standard imports.
import shutil
import subprocess

# ---- Third party imports.
import geopandas as gpd
import pandas as pd

# ---- Local imports.
from sahel import __datadir__ as datadir


# %% Extract global islands database

print("Extract USGS global islands database...")
coast_dir = datadir / 'coastline'

# Extract with 7zip (because zipfile does not support the 'mpk' format)
exepath = datadir / '7za.exe'

# Extract the .zip archive.
zip_path = coast_dir / 'USGSEsriWCMC_GlobalIslands_v3_mpk.zip'
command = f'"{exepath}" x "{zip_path}" -o"{coast_dir}"'
result = subprocess.run(
    command, capture_output=True, text=True, shell=True, check=True
    )

# Extract the .mpk archive.
mpk_path = coast_dir / 'USGSEsriWCMC_GlobalIslands_v3.mpk'
extract_dir = coast_dir / 'USGSEsriWCMC_GlobalIslands_v3'
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True, exist_ok=True)

command = f'"{exepath}" x "{mpk_path}" -o"{extract_dir}"'
result = subprocess.run(
    command, capture_output=True, text=True, shell=True, check=True
    )

mpk_path.unlink()

# %% Extract African continent from global dataset

print('Extract African continent from global dataset...')

gdb_path = extract_dir / 'v108/globalislandsfix.gdb'

ADD_BIG_ISLANDS = False

# Fetch the shapefile for the African continent.
gdf_africa = gpd.read_file(
    gdb_path, layer='USGSEsriWCMC_GlobalIslandsv2_Continents'
    )
gdf_africa = gdf_africa.loc[gdf_africa.OBJECTID == 2]

# Add big islands.
if ADD_BIG_ISLANDS:
    africa_bbox = gdf_africa.total_bounds
    gdf_big_isl = gpd.read_file(
        gdb_path, layer='USGSEsriWCMC_GlobalIslandsv2_BigIslands')
    gdf_big_isl = gdf_big_isl.cx[
        africa_bbox[0] - 10**6:africa_bbox[2] + 10**6,
        africa_bbox[1] - 10**6:africa_bbox[3] + 10**6
    ]

    gdf_africa = gpd.GeoDataFrame(
        pd.concat([gdf_africa, gdf_big_isl], ignore_index=True),
        crs=gdf_africa.crs
        )

# Reproject and save.
gdf_africa = gdf_africa.to_crs('ESRI:102022')  # Africa Albers Equal Area Conic
gdf_africa.to_file(coast_dir / 'africa_landmass.gpkg', driver='GPKG')

shutil.rmtree(extract_dir)
