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


urls = {'BasinATLAS': 'https://figshare.com/ndownloader/files/20082137',
        'RiverATLAS': 'https://figshare.com/ndownloader/files/20087321',
        'LakeATLAS': 'https://figshare.com/ndownloader/files/35959544'}

for key, url in urls.items():

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        # Get the filename from the response headers.
        cd = r.headers.get("Content-Disposition", "")
        filename = cd.split("filename=")[-1].strip('"')

        local_path = DEST_FOLDER / filename

        # Skip if already downloaded.
        if local_path.exists():
            print(f"Skipping {filename}, already exists.")
            continue

        print(f"Downloading {filename}...")
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {local_path.name} sucessfully.")


# %%

# Extract basins level 12 from the BasinATLAS.

africa_gdf = gpd.read_file(datadir / 'coastline' / 'africa_landmass.gpkg')
africa_shape = africa_gdf.union_all()

minx, miny, maxx, maxy = africa_gdf.total_bounds
africa_bbox = box(minx, miny, maxx, maxy)

level = 12
layer_name = f'BasinATLAS_v10_lev{level:02d}'

zip_path = DEST_FOLDER / 'BasinATLAS_Data_v10.gdb.zip'
extract_dir = DEST_FOLDER / 'BasinATLAS_Data_v10'
extract_dir.mkdir(parents=True, exist_ok=True)

print("Extrating zip archive...", flush=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

basins_all_path = extract_dir / 'BasinATLAS_v10.gdb'

print(f'Reading {layer_name} from {basins_all_path.name}...', flush=True)
basins_gdf = gpd.read_file(str(basins_all_path), layer=layer_name)
print('Number of basins:', len(basins_gdf), flush=True)

print('Projecting to ESRI:102022...', flush=True)
basins_gdf = basins_gdf.to_crs('ESRI:102022')

# We only clip to the bounding box of the African continent because
# clipping to the shape is too long.
print("Clipping to African continent bbox...", flush=True)
basins_gdf = gpd.clip(basins_gdf, africa_bbox)
print('Number of basins:', len(basins_gdf), flush=True)

print("Saving results to file...", flush=True)
basins_path = DEST_FOLDER / f'basins_lvl{level:02d}_102022.gpkg'
basins_gdf.to_file(basins_path, layer=layer_name, driver="GPKG")

# Clean up temp files.
shutil.rmtree(extract_dir)
