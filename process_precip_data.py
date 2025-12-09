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

"""Download daily precipitation data ."""

# CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
# is a gridded rainfall dataset providing daily, pentadal, and monthly
# precipitation estimates at ~0.05° resolution, starting from 1981.
# It combines satellite imagery with in-situ station data to support climate
# and drought monitoring applications, especially in data-scarce regions.


# ---- Standard imports.
from datetime import datetime

# ---- Third party imports.
import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
import numpy as np

# ---- Local imports.
from hdml import __datadir__ as datadir
from hdml.gishelpers import clip_and_project_raster
from hdml.zonal_extract import build_zonal_index_map, extract_zonal_means

# See https://www.chc.ucsb.edu/data/chirps.

DEST_DIR = datadir / 'precip'
DEST_DIR.mkdir(parents=True, exist_ok=True)

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

tif_index_fpath = DEST_DIR / 'tif_index.csv'

# Read Africa landmass and get bounding box.
africa_gdf = gpd.read_file(datadir / 'coastline' / 'africa_landmass.gpkg')
africa_shape = africa_gdf.union_all()
africa_bbox = africa_shape.bounds  # (minx, miny, maxx, maxy)

base_url = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat"


# %%

# Download the CHIRPS daily sat data.

if not tif_index_fpath.exists():
    tif_index = pd.DataFrame(
        columns=['file'] + list(basins_gdf.index),
        index=pd.date_range('2000-01-01', '2025-12-31')
        )
    tif_index.index.name = 'date'
else:
    tif_index = pd.read_csv(
        tif_index_fpath, index_col=0, parse_dates=True, dtype={'file': str}
        )

ndownload = 0
for year in range(2025, 2026):
    year_url = base_url + f'/{year}'

    # Get the list of tif files available for download.
    resp = requests.get(year_url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = [a['href'] for a in
             soup.find_all("a") if
             a['href'].endswith(".tif")]

    for file in files:
        dtime = datetime.strptime(file[-14:-4], '%Y.%m.%d')
        file_url = year_url + f'/{file}'

        global_tif_fpath = DEST_DIR / file

        tif_fpath = DEST_DIR / f'{year}' / file
        tif_fpath.parent.mkdir(parents=True, exist_ok=True)

        tif_index.loc[dtime, 'file'] = f'{year}/{file}'

        # Skip if already downloaded.
        if tif_fpath.exists():
            continue

        print(f"[{str(dtime)[:10]}] Downloading tif file...", end='')

        resp = requests.get(file_url, stream=False, timeout=60)
        resp.raise_for_status()
        with open(global_tif_fpath, "wb") as fp:
            fp.write(resp.content)

        clip_and_project_raster(
            global_tif_fpath, tif_fpath,
            output_crs='ESRI:102022', clipping_bbox=africa_bbox
            )

        global_tif_fpath.unlink()

        ndownload += 1

        print(' done')

print()
print('All precip tif file downloaded successfully.')
print()
print('Saving tif index dataframe to file...', end='')
tif_index.to_csv(tif_index_fpath)
print('done')

# %%

# Generate the basin zonal index map.

tif_index = pd.read_csv(
    tif_index_fpath, index_col=0, parse_dates=True, dtype={'file': str}
    )

tif_fnames = tif_index.file
tif_fnames = tif_fnames[~pd.isnull(tif_fnames)]
tif_fnames = np.unique(tif_fnames)

zonal_index_map, small_basin_ids = build_zonal_index_map(
    DEST_DIR / tif_fnames[0], basins_gdf
    )

# Extract precip means for each basin.

ntot = len(tif_index)
for index, row in tif_index.iterrows():
    if pd.isnull(row.file):
        continue

    tif_path = DEST_DIR / row.file

    print(f"[{str(index)[:10]}] Extracing basin mean precip...")

    mean_precip, basin_ids = extract_zonal_means(tif_path, zonal_index_map)

    for i, basin_id in enumerate(basins_gdf.index):
        tif_index.loc[index, str(basin_id)] = mean_precip[i]

tif_index.to_csv(tif_index_fpath)
