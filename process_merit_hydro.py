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

# https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/
# https://hydro.iis.u-tokyo.ac.jp/%7Eyamadai/MERIT_Hydro/
# https://springernature.figshare.com/collections/A_new_vector-based_global_river_network_dataset_accounting_for_variable_drainage_density/5052635
# https://springernature.figshare.com/articles/dataset/F4_watersheds/13256741?file=34469414

# ---- Standard imports.
from pathlib import Path
import tarfile
import os.path as osp
import shutil

# ---- Third party imports.
from osgeo import gdal

# ---- Local imports.
from sahel import __datadir__ as datadir
from sahel.gishelpers import get_dem_filepaths, create_pyramid_overview

gdal.UseExceptions()

merit_dir = Path(datadir) / "merit"

FEATURES = ['elv']
# dem: multi-error-removed improved-terrain DEM
# hnd: height above nearest drainage
# wth: river width
# upg: number of upstream drainage pixels
# upa: upstream drainage area
# elv: adjusted elevation
# dir: flow direction map

# %%

tarpaths = []
tardir = merit_dir / '__raw__'
for p in tardir.iterdir():
    if p.is_file() and p.suffix == ".tar":
        tarpaths.append(p)

for i, p in enumerate(tarpaths):
    print(f'Processing tar file {i + 1} of {len(tarpaths)}...')

    feature = p.name[:3]
    if feature not in FEATURES:
        continue

    subdir = merit_dir / feature

    with tarfile.open(p, mode='r:*') as tf:
        for member in tf.getmembers():
            if not member.name.endswith('.tif'):
                continue

            fp = tf.extractfile(member)

            out = subdir / osp.basename(member.name)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue

            with out.open("wb") as dst:
                shutil.copyfileobj(fp, dst)


# %%

for feature in FEATURES:

    vrt_path = merit_dir / f'{feature}_mosaic.vrt'
    src_paths = get_dem_filepaths(merit_dir / f'{feature}')

    print(f"Generating mosaic for '{feature}'...")
    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, src_paths)
        ds.FlushCache()
        del ds

# %%

dst_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic
pixel_size = 90  # 3 arc-second is ~92.77 m at the equator


for feature in FEATURES:
    input_vrt = merit_dir / f'{feature}_mosaic.vrt'
    geojson_file = datadir / 'gadm' / 'buffered_boundary_100km.json'
    output_tif = merit_dir / f'{feature}_mosaic.tiff'

    # Set up warp options
    warp_options = gdal.WarpOptions(
        format='GTiff',
        cutlineDSName=geojson_file,
        cropToCutline=True,
        dstSRS=dst_crs,
        dstNodata=-9999,
        xRes=pixel_size, yRes=pixel_size, resampleAlg='bilinear',
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES']
        )

    if not output_tif.exists():
        print(f"Clipping and projecting {feature} mosaic...")
        out_ds = gdal.Warp(output_tif, input_vrt, options=warp_options)
        out_ds.FlushCache()
        del out_ds

        print(f"Creating a pyramid overview for the {feature} mosaic...")
        create_pyramid_overview(output_tif)


# %%
filename = "D:/Projets/sahel/data/dem/raw/NASADEM_HGT_n05e000.tif"
with rasterio.open(filename) as src:
    print(f"Data type: {src.dtypes[0]}")  # First band
    print(f"NoData value: {src.nodata}")
    print(f"Number of bands: {src.count}")

    # For all bands:
    for i, dtype in enumerate(src.dtypes, 1):
        print(f"Band {i}: {dtype}")
