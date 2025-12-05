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

# ---- Third party imports
from osgeo import gdal
import pandas as pd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.gishelpers import get_dem_filepaths
from hdml.ed_helpers import earthaccess_login, MOD13Q1_hdf_to_geotiff

earthaccess = earthaccess_login()

MODIS_TILE_NAMES = ['h16v07', 'h17v07', 'h18v07', 'h16v08', 'h17v08', 'h18v08']

# %%

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

dest_dir = datadir / 'ndvi'
dest_dir.mkdir(exist_ok=True)

index_fpath = dest_dir / 'index.csv'
if not index_fpath.exists():
    index = pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    index_df = pd.DataFrame(index=index)
else:
    index_df = pd.read_csv(index_fpath, index_col=[0, 1])


base_url = ('https://data.lpdaac.earthdatacloud.nasa.gov/'
            'lp-prod-protected/MOD13Q1.061')
i = 1
for hdf_name in hdf_names:
    print(f'Processing hdf file {i + 1} of {len(hdf_names)}...')

    url = base_url + '/' + hdf_name + '/' + hdf_name + '.hdf'

    hdf_fpath = dest_dir / (hdf_name + '.hdf')
    tif_fpath = hdf_fpath.with_suffix('.tif')

    # Skip if tile already downloaded and processed file.
    if tif_fpath.exists():
        i += 1
        continue

    if not hdf_fpath.exists():
        # Download the MODIS HDF file and convert to GeoTIFF.
        try:
            earthaccess.download(url, dest_dir, show_progress=False)
        except Exception:
            print(f'Failed to download NDVI data for {hdf_name}.')
            break

    tile_name, date_start, date_end, metadata = MOD13Q1_hdf_to_geotiff(
        hdf_fpath, 0, tif_fpath)

    index_df.loc[(date_start, date_end), tile_name] = tif_fpath.name
    index_df.to_csv(index_fpath)

    # Delete the HDF file.
    hdf_fpath.unlink()

    i += 1
    if i > 100:
        break


# %%

# Generate a GDAL virtual raster (VRT) mosaic of all DEM GeoTIFFs.
vrt_path = datadir / 'dem' / 'nasadem.vrt'
dem_paths = get_dem_filepaths(dest_dir.parent)
ds = gdal.BuildVRT(vrt_path, dem_paths)
ds.FlushCache()
del ds

# Reprojected VRT and apply African landmass mask.

dst_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic
pixel_size = 30  # 1 arc-second is ~30 m at the equator

vrt_reprojected = datadir / 'dem' / 'nasadem_102022.vrt'
warp_options = gdal.WarpOptions(
    cutlineDSName=str(datadir / 'coastline' / 'africa_landmass.gpkg'),
    cropToCutline=False,
    dstSRS=dst_crs,
    format='VRT',
    resampleAlg='bilinear',
    xRes=pixel_size,
    yRes=pixel_size,
    multithread=True,
    )

ds_reproj = gdal.Warp(
    str(vrt_reprojected),
    str(vrt_path),
    options=warp_options
    )
ds_reproj.FlushCache()
del ds_reproj

print(f'Virtual dataset generated at {vrt_reprojected}.')
