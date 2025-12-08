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

# ---- Standard imports
from pathlib import Path
import shutil

# ---- Third party imports
from osgeo import gdal
import pandas as pd
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.ed_helpers import earthaccess_login, MOD13Q1_hdf_to_geotiff

MODIS_TILE_NAMES = ['h16v07', 'h17v07', 'h18v07', 'h16v08', 'h17v08', 'h18v08']

HDF_DIR = datadir / 'ndvi'

TIF_DIR = HDF_DIR / 'tiles'
TIF_DIR.mkdir(parents=True, exist_ok=True)

VRT_DIR = HDF_DIR / 'vrt'
VRT_DIR.mkdir(parents=True, exist_ok=True)

# %%

# Authenticate to Earthdata and get available datasets

print("Authenticating with NASA Earthdata...")
earthaccess = earthaccess_login()

# Get the list of available hDF names from the NDVI MODIS dataset.
print("Getting the list of MODIS datasets from earthdata (might take a few "
      "minutes)...")

granules = earthaccess.search_data(
    short_name="MOD13Q1",
    version="061",
    cloud_hosted=False,
    )

avail_hdf_names = [
    granule['meta']['native-id'] for granule in granules
    ]

# Only keep the hdf for the tiles listed in MODIS_TILE_NAMES.
hdf_names = []
for avail_hdf_name in avail_hdf_names:
    for tile_name in MODIS_TILE_NAMES:
        if tile_name in avail_hdf_name:
            hdf_names.append(avail_hdf_name)
            break

# %%

# Download the NDVI MODIS tiles and convert to GeoTIFF.

index_fpath = TIF_DIR.parent / 'tiles_index.csv'
if not index_fpath.exists():
    index = pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    index_df = pd.DataFrame(index=index)
else:
    index_df = pd.read_csv(index_fpath, index_col=[0, 1])


base_url = ('https://data.lpdaac.earthdatacloud.nasa.gov/'
            'lp-prod-protected/MOD13Q1.061')
i = 0
n = len(hdf_names)
for hdf_name in hdf_names:
    progress = f"[{i+1:02d}/{n}]"

    hdf_fpath = HDF_DIR / (hdf_name + '.hdf')
    tif_fpath = TIF_DIR / (hdf_name + '.tif')
    bck_fpath = Path('E:/MODIS NDVI 250m/') / hdf_fpath.name

    # Skip if tile already downloaded and processed file.
    if tif_fpath.exists():
        i += 1
        continue

    print(f'{progress} Downloading MODIS HDF file...')

    # Download the MODIS HDF file and convert to GeoTIFF.
    if not hdf_fpath.exists():
        url = base_url + '/' + hdf_name + '/' + hdf_name + '.hdf'
        try:
            earthaccess.download(url, HDF_DIR, show_progress=False)
        except Exception:
            print(f'{progress} Failed to download NDVI data for {hdf_name}.')
            break

    print(f'{progress} Converting to GeoTIFF...')
    tile_name, date_start, date_end, metadata = MOD13Q1_hdf_to_geotiff(
        hdf_fpath, 0, tif_fpath)

    index_df.loc[(date_start, date_end), tile_name] = tif_fpath.name
    index_df.to_csv(index_fpath)

    # Move file to external hard drive for backup.
    print(f'{progress} Moving HDF file to backup drive...')
    if hdf_fpath.exists():
        shutil.move(str(hdf_fpath), str(bck_fpath))

    i += 1


# %%

# Generate GDAL virtual raster (VRT).

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

tif_file_index_path = datadir / "ndvi" / "tiles_index.csv"
tif_file_index = pd.read_csv(tif_file_index_path, index_col=[0, 1])

vrt_index_path = datadir / 'ndvi' / 'vrt_index.csv'
if not vrt_index_path.exists():
    vrt_index = pd.DataFrame(
        columns=['file'] + list(basins_gdf.index),
        index=pd.date_range('2000-01-01', '2025-12-31')
        )
else:
    vrt_index = pd.read_csv(vrt_index_path, index_col=0, parse_dates=True)

ntot = len(tif_file_index)
i = 0
for index, row in tif_file_index.iterrows():
    print(f"[{i+1:02d}/{ntot}] Producing VRT for {index[0]}...")

    # Define the name of the VRT file.
    start = index[0].replace('-', '')
    end = index[1].replace('-', '')
    vrt_path = VRT_DIR / f"NDVI_MOD13Q1_{start}_{end}.vrt"

    # Define the list of tiles to add to the VRT.
    tif_paths = [TIF_DIR / tif_fname for tif_fname in row.values]
    assert len(tif_paths) == 6

    # Build the VRT.
    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, tif_paths)
        ds.FlushCache()
        del ds

    # Reprojected VRT.
    dst_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic

    vrt_reprojected = (
        datadir / 'ndvi' / f"NDVI_MOD13Q1_{start}_{end}_ESRI102022.vrt"
        )

    if not vrt_reprojected.exists():
        warp_options = gdal.WarpOptions(
            dstSRS='ESRI:102022',
            format='VRT',
            resampleAlg='bilinear',
            multithread=True,
            )

        ds_reproj = gdal.Warp(
            str(vrt_reprojected),
            str(vrt_path),
            options=warp_options
            )
        ds_reproj.FlushCache()
        del ds_reproj

    # Update the VRT file index.
    vrt_index.loc[pd.date_range(*index), 'file'] = vrt_reprojected.name
    i += 1

vrt_index.to_csv(vrt_index_path)

# %%
import numpy as np
from hdml.zonal_extract import build_zonal_index_map, extract_zonal_means

vrt_index_path = datadir / 'ndvi' / 'vrt_index.csv'
vrt_index = pd.read_csv(vrt_index_path, index_col=0, parse_dates=True)

wtd_gdf = gpd.read_file(datadir / "data" / "wtd_obs_all.gpkg")
wtd_gdf = wtd_gdf.set_index("ID", drop=True)

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

vrt_fnames = vrt_index.file
vrt_fnames = vrt_fnames[~pd.isnull(vrt_fnames)]
vrt_fnames = np.unique(vrt_fnames)


zonal_index_map, bad_basin_ids = build_zonal_index_map(
    datadir / 'ndvi' / vrt_fnames[0], basins_gdf
    )
zonal_index_map['indexes'][basins_gdf.index[0]]

# %%

basin_geom = basins_gdf.geometry.iloc[0]

ntot = len(tif_file_index)
count = 0
for vrt_name in vrt_fnames:
    print(f"[{count+1:02d}/{ntot}] Processing {vrt_name}...")
    vrt_path = datadir / 'ndvi' / vrt_name

    mean_ndvi = extract_zonal_means(vrt_path, basins_gdf.geometry)
    mean_ndvi = mean_ndvi * 0.0001  # MODIS Int16 scale to physical NDVI

    mask_index = vrt_index.file == vrt_path.name
    for i, basin_id in enumerate(basins_gdf.index):
        vrt_index.loc[mask_index, int(basin_id)] = mean_ndvi[i]

    count += 1

vrt_index.to_csv(vrt_index_path)
