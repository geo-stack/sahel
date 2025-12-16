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

# HydroATLAS is a global, sub-basin hydrology dataset providing physical,
# ecological, climatic and socio-economic attributes for river basins and
# catchments at high resolution. It offers standardized basin geometries
# (HydroBASINS) and >700 attributes (e.g. flow accumulation, land cover,
# precipitation) for multiple nested basin levels to support water resources
# management, modeling, and environmental assessment.

# Here we download all three layers of HydroATLAS (BasinATLAS, RiverATLAS,
# and LakeATLAS) in geodatabase format.

# See https://www.hydrosheds.org/products/hydrobasins.

# ---- Standard imports
import zipfile
import shutil

# ---- Third party imports
import requests
import geopandas as gpd
from shapely.geometry import box

# ---- Local imports
from hdml import __datadir__ as datadir

DEST_FOLDER = datadir / 'hydro_atlas'
DEST_FOLDER.mkdir(parents=True, exist_ok=True)

# %%

# Download ATLAS databases.

key = 'BasinATLAS'
url = 'https://figshare.com/ndownloader/files/20082137'

with requests.get(url, stream=True) as r:
    r.raise_for_status()

    # Get the filename from the response headers.
    cd = r.headers.get("Content-Disposition", "")
    filename = cd.split("filename=")[-1].strip('"')

    local_path = DEST_FOLDER / filename

    # Skip if already downloaded.
    if not local_path.exists():
        print(f"Downloading {filename}...")
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {local_path.name} sucessfully.")
    else:
        print(f"'{filename}' already exists.")


# %%

# Extract basins level 12 from the BasinATLAS.

level = 12
layer_name = f'BasinATLAS_v10_lev{level:02d}'

zip_path = DEST_FOLDER / 'BasinATLAS_Data_v10.gdb.zip'
extract_dir = DEST_FOLDER / 'BasinATLAS_Data_v10'
if not extract_dir.exists():
    print("Extrating zip archive...", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# %%

# Clip the basins to the African continent.

africa_simple_path = datadir / 'coastline' / 'africa_landmass_simple.gpkg'

if not africa_simple_path.exists():
    africa_gdf = gpd.read_file(
        datadir / 'coastline' / 'africa_landmass_simple.gpkg')
    africa_simple = africa_gdf.buffer(5000)
    africa_simple = africa_simple.buffer(-5000)
    africa_simple = africa_simple.simplify(1000, preserve_topology=False)
    africa_simple.to_file(africa_simple_path)

africa_gdf = gpd.read_file(africa_simple_path)

basins_all_path = extract_dir / 'BasinATLAS_v10.gdb'

print(f'Reading {layer_name} from {basins_all_path.name}...', flush=True)
basins_gdf = gpd.read_file(str(basins_all_path), layer=layer_name)
print('Number of basins:', len(basins_gdf), flush=True)

print('Projecting to ESRI:102022...', flush=True)
basins_gdf = basins_gdf.to_crs('ESRI:102022')

# Clipping to the non-simplifed African continent shape is way too long,
# so we need to do this in two steps and use a simplified shape.

print("Clipping to African continent bbox...", flush=True)
basins_gdf_bbox = gpd.clip(basins_gdf, box(*africa_gdf.total_bounds))
print('Number of basins:', len(basins_gdf_bbox), flush=True)

print("Clipping to simplified African continent shape...", flush=True)
basins_africa = gpd.clip(basins_gdf_bbox, africa_gdf.union_all())
print('Number of basins:', len(basins_africa), flush=True)

print("Saving results to file...", flush=True)
basins_path = DEST_FOLDER / f'basins_lvl{level:02d}_102022.gpkg'
basins_africa.to_file(basins_path, layer=layer_name, driver="GPKG")

# Clean up temp files.
shutil.rmtree(extract_dir)
