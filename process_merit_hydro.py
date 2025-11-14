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
import zipfile

# ---- Third party imports.
from osgeo import gdal
import pandas as pd
import geopandas as gpd

# ---- Local imports.
from sahel import __datadir__ as datadir
from sahel.gishelpers import (
    get_dem_filepaths, create_pyramid_overview, rasterize_streams)

gdal.UseExceptions()

merit_dir = Path(datadir) / "merit"
study_area_path = datadir / 'gadm' / 'study_area_bbox_(100km buffer).json'
african_landmass = datadir / 'coastline' / 'africa_landmass.gpkg'

FEATURES = ['elv']
# dem: multi-error-removed improved-terrain DEM
# hnd: height above nearest drainage
# wth: river width
# upg: number of upstream drainage pixels
# upa: upstream drainage area
# elv: adjusted elevation
# dir: flow direction map

# %%  Extract merit data

tarpaths = []
tardir = merit_dir / '__src__'
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


# %% Clip and project merit data.

dst_crs = 'ESRI:102022'  # Africa Albers Equal Area Conic
pixel_size = 90  # 3 arc-second is ~92.77 m at the equator

for feature in FEATURES:
    print(f"Clipping and projecting {feature} mosaic...")

    # Create a mosaic first since it is faster.
    vrt_path = merit_dir / f'{feature}_mosaic.vrt'
    src_paths = get_dem_filepaths(merit_dir / f'{feature}')

    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, src_paths)
        ds.FlushCache()
        del ds

    # Clip, project and save in a geotiff.
    output_tif = merit_dir / f'{feature}_mosaic.tiff'

    warp_options = gdal.WarpOptions(
        format='GTiff',
        cutlineDSName=study_area_path,
        cropToCutline=True,
        dstSRS=dst_crs,
        dstNodata=-9999,
        xRes=pixel_size, yRes=pixel_size, resampleAlg='bilinear',
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES']
        )

    if not output_tif.exists():
        out_ds = gdal.Warp(output_tif, vrt_path, options=warp_options)
        out_ds.FlushCache()
        del out_ds

    # Delete virtual mosaics since we won't need them anymore.
    vrt_path.unlink()

# %% Apply African landmask

for feature in FEATURES:
    print(f"Masking '{feature}' with African landmass (coastline)...")

    input_tif = merit_dir / f'{feature}_mosaic.tiff'
    output_tif = merit_dir / f'{feature}_mosaic_masked.tiff'

    warp_options = gdal.WarpOptions(
        cutlineDSName=str(african_landmass),
        cropToCutline=False,  # Keep original extent
        dstNodata=-9999,
        srcNodata=-9999,
        format='GTiff',
        creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )

    out_ds = gdal.Warp(str(output_tif), str(input_tif), options=warp_options)
    out_ds.FlushCache()
    del out_ds

    output_tif.replace(input_tif)

    print(f"Creating a pyramid overview for the {feature} mosaic...")
    create_pyramid_overview(input_tif)


# %% Pre-process river network data

# Selected zone for wich river network will be extracted.
# See 'lev02_basin_numbers.jpg' in './sahel/data/merit/lin_et_al_2021'
RIVERS_ZONE_IDS = [13, 14, 15, 16, 17]

# Paths
rivers_zip_path = merit_dir / 'lin_et_al_2021/river_network_variable_Dd.zip'
study_area = gpd.read_file(study_area_path)

# Extract files related to each target zone.
extract_dir = rivers_zip_path.parent

print('Processing river network vector layer...')

shp_to_merge = []
with zipfile.ZipFile(rivers_zip_path, 'r') as zf:
    for zone_id in RIVERS_ZONE_IDS:
        # Extract shapefile and all related files (.shx, .dbf, .prj, etc.).
        base_name = f'pfaf_variable_Dd_{zone_id}'
        parent_dir = Path('river_network_variable_Dd')

        for fname in zf.namelist():
            fname_path = Path(fname)
            if (fname_path.stem, fname_path.parent) == (base_name, parent_dir):
                dst_path = extract_dir / fname
                if not dst_path.exists():
                    zf.extract(fname, str(extract_dir))

        # Read the shapefile.
        shp_to_merge.append(
            extract_dir / 'river_network_variable_Dd' / (base_name + '.shp')
            )

# Read, project, clip and merge the river shp.
gdf_to_merge = []
for shp_fpath in shp_to_merge:
    gdf = gpd.read_file(shp_fpath)

    # Reproject if needed.
    if gdf.crs != study_area.crs:
        gdf = gdf.to_crs(study_area.crs)

    gdf_to_merge.append(gdf)

# Merge and save
rivers_output_gpkg = merit_dir / 'river_network_var_Dd.gpkg'
merged_rivers = gpd.GeoDataFrame(pd.concat(gdf_to_merge, ignore_index=True))
merged_rivers.to_file(rivers_output_gpkg, driver='GPKG')

print('Rasterizing river network...')

# rasterize river network data.
output_path = rasterize_streams(
    vector_path=rivers_output_gpkg,
    template_raster=merit_dir / 'elv_mosaic.tiff',
    output_raster=merit_dir / 'river_network_var_Dd.tiff',
    background_value=0,
    attribute='strmOrder',
    all_touched=True,
    overwrite=True
    )

print("Creating a pyramid overview for the  river network...")
create_pyramid_overview(output_path)
