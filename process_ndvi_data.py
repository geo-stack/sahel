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

# MODIS NDVI (MOD13A1) data from NASA EarthData is only available since the
# year 2000 because the MODIS sensors aboard Terra and Aqua satellites were
# launched in late 1999 and 2002, respectively.

# https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-006

# ---- Standard imports
from pathlib import Path
import shutil

# ---- Third party imports
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.ed_helpers import (
    earthaccess_login, MOD13Q1_hdf_to_geotiff, get_MOD13Q1_hdf_metadata)
from hdml.zonal_extract import build_zonal_index_map, extract_zonal_means

MODIS_TILE_NAMES = ['h16v07', 'h17v07', 'h18v07', 'h19v07', 'h20v07',
                    'h16v08', 'h17v08', 'h18v08', 'h19v08']

NDVI_DIR = datadir / 'ndvi'

HDF_DIR = Path("E:/Banque Mondiale (HydroDepthML)/MODIS NDVI 250m")
HDF_DIR.mkdir(parents=True, exist_ok=True)

TIF_DIR = NDVI_DIR / 'tiles'
TIF_DIR.mkdir(parents=True, exist_ok=True)

VRT_DIR = NDVI_DIR / 'vrt'
VRT_DIR.mkdir(parents=True, exist_ok=True)

tif_file_index_path = datadir / "ndvi" / "ndvi_tiles_index.csv"
vrt_index_path = datadir / 'ndvi' / 'ndvi_vrt_index.csv'

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)


# %%

# Authenticate to Earthdata and get available datasets

print("Authenticating with NASA Earthdata...")
earthaccess = earthaccess_login()

# Get the list of available hDF names from the NDVI MODIS dataset.

all_granules = []
for tile in MODIS_TILE_NAMES:
    print(f"Searching for tile {tile}...")
    granules = earthaccess.search_data(
        short_name="MOD13Q1",
        version="061",
        cloud_hosted=False,
        granule_name=f"*{tile}*"  # Wildcard pattern to match tile name
    )
    all_granules.extend(granules)
    print(f"  Found {len(granules)} granules for {tile}")


hdf_names = {}
for granule in all_granules:
    tile_id = granule['meta']['native-id']
    for url_data in granule['umm']['RelatedUrls']:
        url = url_data['URL']
        if url.endswith('hdf'):
            break
    else:
        raise ValueError("Cannot find a URL ending with '.hdf'.")

    hdf_names[tile_id] = url


# %%

# Download the NDVI MODIS tiles and convert to GeoTIFF.

index_fpath = TIF_DIR.parent / 'tiles_index.csv'
if not index_fpath.exists():
    index = pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    index_df = pd.DataFrame(index=index)
else:
    index_df = pd.read_csv(index_fpath, index_col=[0, 1])

i = 0
n = len(hdf_names)
for hdf_name, url in hdf_names.items():
    progress = f"[{i+1:02d}/{n}]"

    # Download the MODIS HDF file.

    hdf_fpath = HDF_DIR / (hdf_name + '.hdf')

    if not hdf_fpath.exists():
        print(f'{progress} Downloading MODIS HDF file...')
        try:
            earthaccess.download(url, HDF_DIR, show_progress=False)
        except Exception:
            print(f'{progress} Failed to download NDVI data for {hdf_name}.')
            index_df.to_csv(index_fpath)
            break

    # Convert MODIS HDF file to GeoTIFF.

    tif_fpath = TIF_DIR / (hdf_name + '.tif')

    if not tif_fpath.exists():
        print(f'{progress} Converting to GeoTIFF...')
        tile_name, date_start, date_end = MOD13Q1_hdf_to_geotiff(
            hdf_fpath, 0, tif_fpath)
    else:
        print(f'{progress} Fetching MODIS HDF metadata...')
        with rasterio.open(tif_fpath) as src:
            meta_dict = src.tags()
            tile_name = meta_dict.get('tile_name')
            date_start = meta_dict.get('date_start')
            date_end = meta_dict.get('date_end')

    index_df.loc[(date_start, date_end), tile_name] = tif_fpath.name
    i += 1

index_df.to_csv(index_fpath)


# %%

# Generate GDAL virtual raster (VRT).

tif_file_index = pd.read_csv(tif_file_index_path, index_col=[0, 1])

if not vrt_index_path.exists():
    vrt_index = pd.DataFrame(
        columns=['file'] + list(basins_gdf.index.astype(str)),
        index=pd.date_range('2000-01-01', '2025-12-31')
        )
else:
    vrt_index = pd.read_csv(
        vrt_index_path, index_col=0, parse_dates=True, dtype={'file': str}
        )

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

    vrt_reprojected = VRT_DIR / f"NDVI_MOD13Q1_{start}_{end}_ESRI102022.vrt"

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

# Generate the basin zonal index map.

vrt_index = pd.read_csv(
    vrt_index_path, index_col=0, parse_dates=True, dtype={'file': str}
    )

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

vrt_fnames = vrt_index.file
vrt_fnames = vrt_fnames[~pd.isnull(vrt_fnames)]
vrt_fnames = np.unique(vrt_fnames)

zonal_index_map, bad_basin_ids = build_zonal_index_map(
    VRT_DIR / vrt_fnames[0], basins_gdf
    )

# Extract NDVI means for each basin.

ntot = len(vrt_fnames)
count = 0
for vrt_name in vrt_fnames:
    print(f"[{count+1:02d}/{ntot}] Processing {vrt_name}...")
    vrt_path = VRT_DIR / vrt_name

    mean_ndvi, basin_ids = extract_zonal_means(vrt_path, zonal_index_map)
    mean_ndvi = mean_ndvi * 0.0001  # MODIS Int16 scale to physical NDVI

    mask_index = vrt_index.file == vrt_name
    for i, basin_id in enumerate(basins_gdf.index):
        vrt_index.loc[mask_index, str(basin_id)] = mean_ndvi[i]

    count += 1

vrt_index.to_csv(vrt_index_path)
