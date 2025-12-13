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

# ---- Standard imports
from pathlib import Path

# ---- Third party imports
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.topo import generate_topo_features_for_tile


TILES_OVERLAP_DIR = datadir / 'topo' / 'tiles (cropped)'
TILES_CROPPED_DIR = datadir / 'topo' / 'tiles (overlapped)'

tiles_gdf = gpd.read_file(datadir / "topo" / "tiles_geom_training.gpkg")

tile_count = 0
total_tiles = len(tiles_gdf)
for _, tile_bbox_data in tiles_gdf.iterrows():
    tile_count += 1

    if total_tiles >= 100:
        progress = f"[{tile_count:03d}/{total_tiles}]"
    elif total_tiles >= 10:
        progress = f"[{tile_count:03d}/{total_tiles}]"
    else:
        progress = f"[{tile_count}/{total_tiles}]"

    generate_topo_features_for_tile(
        tile_bbox_data=tile_bbox_data,
        dem_path=datadir / 'dem' / 'nasadem_102022.vrt',
        crop_tile_dir=TILES_CROPPED_DIR,
        ovlp_tile_dir=TILES_OVERLAP_DIR,
        print_affix=progress,
        extract_streams_treshold=500,
        gaussian_filter_sigma=1,
        ridge_size=30,
        )
