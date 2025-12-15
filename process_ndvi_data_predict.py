# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/hydrodepthml
# =============================================================================

# MODIS NDVI (MOD13A1) data from NASA EarthData is only available since the
# year 2000 because the MODIS sensors aboard Terra and Aqua satellites were
# launched in late 1999 and 2002, respectively.

# https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-006

# ---- Standard imports
from pathlib import Path
from time import perf_counter

# ---- Third party imports
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.ed_helpers import earthaccess_login, MOD13Q1_hdf_to_geotiff
from hdml.zonal_extract import build_zonal_index_map, extract_zonal_means

MODIS_TILE_NAMES = [
              'h17v05', 'h18v05', 'h19v05', 'h20v05',
    'h16v06', 'h17v06', 'h18v06', 'h19v06', 'h20v06', 'h21v06',
    'h16v07', 'h17v07', 'h18v07', 'h19v07', 'h20v07', 'h21v07', 'h22v07',
    'h16v08', 'h17v08', 'h18v08', 'h19v08', 'h20v08', 'h21v08', 'h22v08',
                        'h18v09', 'h19v09', 'h20v09', 'h21v09', 'h22v09',
                                  'h19v10', 'h20v10', 'h21v10', 'h22v10',
                                  'h19v11', 'h20v11', 'h21v11',
                                  'h19v12', 'h20v12'
    ]


NDVI_DIR = datadir / 'ndvi_predict'
NDVI_DIR.mkdir(parents=True, exist_ok=True)

HDF_DIR = Path("E:/Banque Mondiale (HydroDepthML)/MODIS MOD13Q1 HDF 250m")
HDF_DIR.mkdir(parents=True, exist_ok=True)

TIF_DIR = Path("E:/Banque Mondiale (HydroDepthML)/MODIS NDVI TIF 250m")
TIF_DIR.mkdir(parents=True, exist_ok=True)

VRT_DIR = NDVI_DIR / 'vrt'
VRT_DIR.mkdir(parents=True, exist_ok=True)

MOSAIC_DIR = NDVI_DIR / 'mosaic'
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

tif_file_index_path = NDVI_DIR / "ndvi_tiles_index.csv"
mosaic_index_path = NDVI_DIR / 'ndvi_mosaic_index.csv'

basins_path = datadir / 'hydro_atlas' / 'basins_lvl12_102022.gpkg'
basins_gdf = gpd.read_file(basins_path)
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
        temporal=("2023-12-31", "2026-01-01"),
        granule_name=f"*{tile}*"
    )
    all_granules.extend(granules)
    print(f"  Found {len(granules)} granules for {tile}")


hdf_names = {}
for granule in all_granules:
    tile_id = granule['meta']['native-id']
    if 'A2025' not in tile_id and 'A2024' not in tile_id:
        continue

    for url_data in granule['umm']['RelatedUrls']:
        url = url_data['URL']
        if url.endswith('hdf'):
            break
    else:
        raise ValueError("Cannot find a URL ending with '.hdf'.")

    hdf_names[tile_id] = url


# %%

# Download the NDVI MODIS tiles and convert to GeoTIFF.

if not tif_file_index_path.exists():
    index = pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    index_df = pd.DataFrame(index=index)
else:
    index_df = pd.read_csv(tif_file_index_path, index_col=[0, 1])

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
            index_df.to_csv(tif_file_index_path)
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

    if i > 500:
        break

index_df.to_csv(tif_file_index_path)


# %%

# Generate the tiled GeoTIFF mosaic.

tif_file_index = pd.read_csv(tif_file_index_path, index_col=[0, 1])

if not mosaic_index_path.exists():
    mosaic_index = pd.DataFrame(
        columns=['file'] + list(basins_gdf.index.astype(str)),
        index=pd.date_range('2000-01-01', '2025-12-31')
        )
else:
    mosaic_index = pd.read_csv(
        mosaic_index_path, index_col=0, parse_dates=True, dtype={'file': str}
        )


ntot = len(tif_file_index)
i = 0
for index, row in tif_file_index.iterrows():
    t0 = perf_counter()
    print(f"[{i+1:02d}/{ntot}] Producing a mosaic for {index[0]}...", end=' ')

    # Define the name of the VRT file.
    start = index[0].replace('-', '')
    end = index[1].replace('-', '')
    vrt_path = VRT_DIR / f"NDVI_MOD13Q1_{start}_{end}.vrt"

    # Define the list of tiles to add to the VRT.
    tif_paths = [TIF_DIR / tif_fname for tif_fname in row.values]
    assert len(tif_paths) == 9

    # Build a VRT first.
    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, tif_paths)
        ds.FlushCache()
        del ds

    # Reprojected and assemble the tiles into a mosaic.
    mosaic_path = MOSAIC_DIR / f"NDVI_MOD13Q1_{start}_{end}_ESRI102022.tif"
    if not mosaic_path.exists():
        warp_options = gdal.WarpOptions(
            dstSRS='ESRI:102022',  # Africa Albers Equal Area Conic
            format='GTiff',
            resampleAlg='bilinear',
            creationOptions=[
                'COMPRESS=DEFLATE',
                'TILED=YES',
                'BIGTIFF=YES'
                ]
            )

        ds_reproj = gdal.Warp(
            str(mosaic_path),
            str(vrt_path),
            options=warp_options
            )
        ds_reproj.FlushCache()
        del ds_reproj

    # Update the VRT file index.
    mosaic_index.loc[pd.date_range(*index), 'file'] = mosaic_path.name

    i += 1
    t1 = perf_counter()
    print(f'done in {t1 - t0:0.1f} sec')

mosaic_index.to_csv(mosaic_index_path)

# %%

# Generate the basin zonal index map.

mosaic_index = pd.read_csv(
    mosaic_index_path, index_col=0, parse_dates=True, dtype={'file': str}
    )

basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

mosaic_fnames = mosaic_index.file
mosaic_fnames = mosaic_fnames[~pd.isnull(mosaic_fnames)]
mosaic_fnames = np.unique(mosaic_fnames)

zonal_index_map, bad_basin_ids = build_zonal_index_map(
    MOSAIC_DIR / mosaic_fnames[0], basins_gdf
    )

# %%

# Extract NDVI means for each basin.

ntot = len(mosaic_fnames)
count = 0
for mosaic_name in mosaic_fnames:
    t0 = perf_counter()
    mask_index = mosaic_index.file == mosaic_name

    if not np.any(pd.isnull(mosaic_index.loc[mask_index])):
        print(f"[{count+1:02d}/{ntot}] Skipping "
              f"already processed {mosaic_name}...")

        count += 1
        continue

    print(f"[{count+1:02d}/{ntot}] Processing {mosaic_name}...", end=' ')

    mean_ndvi, basin_ids = extract_zonal_means(
        MOSAIC_DIR / mosaic_name, zonal_index_map)
    mean_ndvi = mean_ndvi * 0.0001  # MODIS Int16 scale to physical NDVI

    for i, basin_id in enumerate(basins_gdf.index):
        mosaic_index.loc[mask_index, str(basin_id)] = mean_ndvi[i]

    count += 1
    t1 = perf_counter()
    print(f'done in {t1 - t0:0.1f} sec')

mosaic_index.to_csv(mosaic_index_path)
