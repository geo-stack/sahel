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
from pandas.io.parquet import read_parquet
import geopandas as gpd
import numpy as np
import rasterio

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.ed_helpers import (
    earthaccess_login, MOD13Q1_hdf_to_geotiff, get_mod13q1_hdf_urls)
from hdml.zonal_extract import build_zonal_index_map, extract_zonal_means

# MODIS_TILE_NAMES specifies which MODIS tiles to download NDVI data for,
# along with the year ranges:
# - For 2024 and 2025: Download all tiles covering Africa for static
#   water depth prediction.
# - For tiles where water level observations are available, download data
#   for 2000–2025 for model training.
# If more observation sites are added to the training dataset, update this
# list to include the corresponding tiles and years.
# See https://modis-land.gsfc.nasa.gov/MODLAND_grid.html

predict_year_range = (2024, 2025)
training_year_range = (2000, 2025)

MODIS_TILE_NAMES = [
    # row 05
    ('h17v05', *predict_year_range),
    ('h18v05', *predict_year_range),
    ('h19v05', *predict_year_range),
    ('h20v05', *predict_year_range),
    # row 06
    ('h16v06', *predict_year_range),
    ('h17v06', *predict_year_range),
    ('h18v06', *predict_year_range),
    ('h19v06', *predict_year_range),
    ('h20v06', *predict_year_range),
    ('h21v06', *predict_year_range),
    # row 07
    ('h16v07', *training_year_range),
    ('h17v07', *training_year_range),
    ('h18v07', *training_year_range),
    ('h19v07', *training_year_range),
    ('h20v07', *training_year_range),
    ('h21v07', *predict_year_range),
    ('h22v07', *predict_year_range),
    # row 08
    ('h16v08', *training_year_range),
    ('h17v08', *training_year_range),
    ('h18v08', *training_year_range),
    ('h19v08', *training_year_range),
    ('h20v08', *predict_year_range),
    ('h21v08', *predict_year_range),
    ('h22v08', *predict_year_range),
    # row 09
    ('h18v09', *predict_year_range),
    ('h19v09', *predict_year_range),
    ('h20v09', *predict_year_range),
    ('h21v09', *predict_year_range),
    ('h22v09', *predict_year_range),
    # row 10
    ('h19v10', *predict_year_range),
    ('h20v10', *predict_year_range),
    ('h21v10', *predict_year_range),
    ('h22v10', *predict_year_range),
    # row 11
    ('h19v11', *predict_year_range),
    ('h20v11', *predict_year_range),
    ('h21v11', *predict_year_range),
    # row 12
    ('h19v12', *predict_year_range),
    ('h20v12', *predict_year_range)
    ]

NDVI_DIR = datadir / 'ndvi'
NDVI_DIR.mkdir(parents=True, exist_ok=True)

HDF_DIR = Path("E:/Banque Mondiale (HydroDepthML)/MODIS MOD13Q1 HDF 250m")
HDF_DIR.mkdir(parents=True, exist_ok=True)

TIF_DIR = Path("E:/Banque Mondiale (HydroDepthML)/MODIS NDVI TIF 250m")
TIF_DIR.mkdir(parents=True, exist_ok=True)

VRT_DIR = NDVI_DIR / 'vrt'
VRT_DIR.mkdir(parents=True, exist_ok=True)

MOSAIC_DIR = NDVI_DIR / 'mosaic'
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

tif_file_index_path = NDVI_DIR / 'ndvi_tiles_index.csv'
mosaic_index_path = NDVI_DIR / 'ndvi_mosaic_index.csv'


# %%

# Authenticate to Earthdata and get available datasets

print("Authenticating with NASA Earthdata...")
earthaccess = earthaccess_login()

# Get the list of available hDF names from the NDVI MODIS dataset for the
# entire African continent.

hdf_urls = {}
for tile_name, year_from, year_to in MODIS_TILE_NAMES:
    print(f"Getting HDF urls for tile {tile_name}...")
    tile_hdf_urls = get_mod13q1_hdf_urls(tile_name, year_from, year_to)
    hdf_urls.update(tile_hdf_urls)
    print(f"  Found {len(tile_hdf_urls)} granules for {tile_name}")


# %%

# Download the NDVI MODIS tiles and convert to GeoTIFF (skip if they exist).

tif_file_index = pd.DataFrame(
    index=pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    )

i = 0
n = len(hdf_urls)
for hdf_name, url in hdf_urls.items():
    progress = f"[{i+1:02d}/{n}]"

    tif_fpath = TIF_DIR / (hdf_name + '.tif')
    if tif_fpath.exists():
        with rasterio.open(tif_fpath) as src:
            print(f'{progress} NDVI data already downloaded and processed.')
            meta_dict = src.tags()
            tile_name = meta_dict.get('tile_name')
            date_start = meta_dict.get('date_start')
            date_end = meta_dict.get('date_end')
        tif_file_index.loc[(date_start, date_end), tile_name] = tif_fpath.name
        i += 1
        continue

    # Download the MODIS HDF file.

    hdf_fpath = HDF_DIR / (hdf_name + '.hdf')
    if not hdf_fpath.exists():
        print(f'{progress} Downloading MODIS HDF file...')
        try:
            earthaccess.download(url, HDF_DIR, show_progress=False)
        except Exception:
            print(f'{progress} Failed to download NDVI data for {hdf_name}.')
            break

    # Convert MODIS HDF file to GeoTIFF.
    print(f'{progress} Converting to GeoTIFF...')
    tile_name, date_start, date_end = MOD13Q1_hdf_to_geotiff(
        hdf_fpath, 0, tif_fpath)

    tif_file_index.loc[(date_start, date_end), tile_name] = tif_fpath.name
    i += 1


tif_file_index = tif_file_index.sort_index()
tif_file_index.to_csv(tif_file_index_path)


# %%

# Generate the tiled GeoTIFF mosaic.

tif_file_index = pd.read_csv(tif_file_index_path, index_col=[0, 1])

mosaic_index = pd.DataFrame(
    columns=['file', 'ntiles'],
    index=pd.date_range('2000-01-01', '2025-12-31')
    )

ntot = len(tif_file_index)
i = 0
for index, row in tif_file_index.iterrows():
    start = index[0].replace('-', '')
    end = index[1].replace('-', '')

    # Define the list of tiles to add to the mosaic.
    tif_paths = [
        TIF_DIR / tif_fname for tif_fname in row.values if
        not pd.isnull(tif_fname)
        ]

    # Define the name of the final mosaic and check if it exists.
    mosaic_path = MOSAIC_DIR / f"NDVI_MOD13Q1_{start}_{end}_ESRI102022.tif"
    if mosaic_path.exists():
        print(f"[{i+1:02d}/{ntot}] Mosaic already exists for {index[0]}.")
        mosaic_index.loc[pd.date_range(*index), 'file'] = mosaic_path.name
        mosaic_index.loc[pd.date_range(*index), 'ntiles'] = len(tif_paths)
        i += 1
        continue

    t0 = perf_counter()
    print(f"[{i+1:02d}/{ntot}] Producing a mosaic for {index[0]}...", end=' ')

    # Define the name of the VRT file.
    vrt_path = VRT_DIR / f"NDVI_MOD13Q1_{start}_{end}.vrt"

    # Build a VRT first.
    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, tif_paths)
        ds.FlushCache()
        del ds

    # Reprojected and assemble the tiles into a mosaic.
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
    mosaic_index.loc[pd.date_range(*index), 'ntiles'] = len(tif_paths)

    i += 1
    t1 = perf_counter()
    print(f'done in {t1 - t0:0.1f} sec')

mosaic_index = mosaic_index.dropna(how='all')
mosaic_index.to_csv(mosaic_index_path)


# %%

def extract_basins_zonal_means(
        mosaic_index_path: Path,
        basins_path: Path,
        year_start: int, year_end: int
        ):

    mosaic_index = pd.read_csv(
        mosaic_index_path,
        index_col=0,
        parse_dates=True,
        dtype={'file': str}
        )

    basins_path = Path(basins_path)
    if not basins_path.exists():
        raise FileNotFoundError(
            "Make sure to run 'process_hydro_basins.py' to generate the "
            "the 'basins_lvl12_102022.gpkg' file."
            )

    print('Loading hydro atlas basins...', flush=True)
    basins_gdf = gpd.read_file(basins_path)
    basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
    basins_gdf.index = basins_gdf.index.astype(int)

    # Generate the basin zonal index map for the PREDICT dataset, which
    # includes data for the whole African continent for the years defined
    # in 'predict_year_range'.

    print('Building the basins zonal index map...', flush=True)

    years = list(range(year_start, year_end + 1))
    mask = (
        (np.isin(mosaic_index.index.year, years)) &
        (~pd.isnull(mosaic_index.file))
        )

    mosaic_fnames = np.unique(mosaic_index.file[mask])
    zonal_index_map, bad_basin_ids = build_zonal_index_map(
        MOSAIC_DIR / mosaic_fnames[0], basins_gdf
        )

    # Initiating the basin ndvi dataframe.
    index = mosaic_index.index[mask]
    columns = list(basins_gdf.index)
    basin_ndvi_means = pd.DataFrame(
        data=np.full((len(index), len(columns)), np.nan, dtype='float32'),
        index=index,
        columns=columns
        )

    ntot = len(mosaic_fnames)
    count = 0
    for mosaic_name in mosaic_fnames:
        t0 = perf_counter()
        dates = mosaic_index.loc[mosaic_index.file == mosaic_name].index

        print(f"[{count+1:02d}/{ntot}] Processing {mosaic_name}...", end=' ')

        mean_ndvi, basin_ids = extract_zonal_means(
            MOSAIC_DIR / mosaic_name, zonal_index_map)
        mean_ndvi = mean_ndvi * 0.0001  # MODIS Int16 scale to physical NDVI

        assert list(basin_ids) == list(basin_ndvi_means.columns)
        basin_ndvi_means.loc[dates] = mean_ndvi.astype('float32')

        count += 1
        t1 = perf_counter()
        print(f'done in {t1 - t0:0.1f} sec')

    return basin_ndvi_means


ndvi_means_africa_basins = extract_basins_zonal_means(
    mosaic_index_path=mosaic_index_path,
    basins_path=datadir / 'hydro_atlas' / 'basins_lvl12_102022.gpkg',
    year_start=predict_year_range[0],
    year_end=predict_year_range[1]
    )
ndvi_means_africa_basins.to_hdf(
    NDVI_DIR / 'ndvi_means_africa_basins.h5', key='ndvi', mode='w'
    )

ndvi_means_wtd_basins = extract_basins_zonal_means(
    mosaic_index_path=mosaic_index_path,
    basins_path=datadir / 'data' / 'wtd_basin_geometry.gpkg',
    year_start=training_year_range[0],
    year_end=training_year_range[1]
    )
ndvi_means_wtd_basins.to_hdf(
    NDVI_DIR / 'ndvi_means_wtd_basins.h5', key='ndvi', mode='w'
    )
