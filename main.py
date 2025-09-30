# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2024 (C) Aziz Agrebi
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# Originally developed by Aziz Agrebi as part of his master's project.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel
# =============================================================================
from __future__ import annotations

# Standard imports
import os
import os.path as osp

# Third party imports
import rasterio
import numpy as np
from pysheds.grid import Grid
import pandas as pd
import cv2
import whitebox_workflows as wbw
from scipy.ndimage import label
from skimage.measure import regionprops

# Local imports
from sahel import __datadir__


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def convert_coord_to_indices(lon, lat, grid_affine):
    col = round((lon - grid_affine[2]) / grid_affine[0])
    row = round((lat - grid_affine[5]) / grid_affine[4])
    return col, row


min_size = 70
kernel_size = (5, 5)
sigma = 1.0

# Hyperparameter for acccumlation threshold to extract streams.
threshold = 1500

size = 200

temp_folder = osp.join(__datadir__, 'results', 'temp')
os.makedirs(temp_folder, exist_ok=True)

csv_files = [f for f in os.listdir(temp_folder) if f.endswith('.csv')]

map_folder = osp.join(__datadir__, 'results')
map_csv_files = [f for f in os.listdir(map_folder) if f.endswith('.csv')]

dem_path = osp.join(__datadir__, 'dem')

# The rainfedcropland rasters (.tif) of the areas where we want to perform
# groundwater table depth calculations, for the 6 countries of interest:
# Burkina Faso, Chad, Mali, Mauritania, Niger, and Senegal. These rasters
# are aligned with the SRTM 30 m. The expected calculation is the groundwater
# table level at the center of the activated pixels.

COUNTRIES = ['Burkina', 'Tchad', 'Mali', 'Mauritania', 'Niger', 'Senegal']


# %%

# Compute the geomorphon map for each DEM tile of each country and write the
# resulting map to disk as a GeoTIFF.

def create_geomorphon_map(dem_filepath, geomorphon_filepath):
    """
    Generate a geomorphon classification map from a DEM and write the
    resulting map to disk as a GeoTIFF.

    Parameters
    ----------
    dem_filepath : str
        Path to the input DEM raster file.
    geomorphon_filepath : str
        Path to the output geomorphon classification raster file.
    """
    wbe = wbw.WbEnvironment()

    dem = wbe.read_raster(dem_filepath)
    dem = wbe.fill_missing_data(
        dem, filter_size=35, exclude_edge_nodata=True)
    dem = wbe.gaussian_filter(dem, sigma=2)
    dem = wbe.fill_missing_data(
        dem, filter_size=35, exclude_edge_nodata=True)

    # Note that the double filling is a common pattern in GIS data processing
    # to ensure a clean, complete raster for subsequent analysis.
    # The first filling is to fill original gaps and ensure smoothing
    # operates on valid data. The second filling is to fill any new or
    # residual gaps created by the smoothing process.

    wbe.write_raster(
        wbe.geomorphons(
            dem,
            search_distance=100,
            flatness_threshold=1.0,
            flatness_distance=0,
            skip_distance=0,
            output_forms=True,
            analyze_residuals=True),
        geomorphon_filepath,
        compress=True,
        )


for country in COUNTRIES:
    dem_folderpath = osp.join(
        __datadir__, 'dem', f'{country}'
        )

    geomorphon_folderpath = osp.join(
        __datadir__, 'results', 'geomorphons', f'{country}'
        )
    os.makedirs(geomorphon_folderpath, exist_ok=True)

    dem_filepaths = get_dem_filepaths(country)

    for index, dem_filepath in enumerate(dem_filepaths):
        geomorphon_filepath = osp.join(
            geomorphon_folderpath, f'{country}_geomorphon_{index:03d}.tif'
            )

        if osp.exists(geomorphon_filepath):
            continue

        print(f"Generating geomorphon map for tile {index:03d} of {country}.")
        create_geomorphon_map(dem_filepath, geomorphon_filepath)

    break


# %%

for country in COUNTRIES:

    # Extracts the geographic coordinates (lat/lon) of rainfed cropland pixels
    # for the specified country (aligned to SRTM, 30 m resolution).
    filepath = osp.join(
        __datadir__, 'pixels', f"{country}_rainfedcropland_ls.tif"
        )
    with rasterio.open(filepath) as src:
        transform = src.transform
        pixels_of_interest = src.read(1)
        pixels_of_interest = np.maximum(pixels_of_interest, 0)

    rows, cols = np.where(pixels_of_interest == 1)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    df = pd.DataFrame({'LON': xs, 'LAT': ys})

    dem_filepaths = get_dem_filepaths(country)
    for index, dem_filepath in enumerate(dem_filepaths):

        if f"map_{country}_{index}.csv" in csv_files:
            continue

        grid = Grid.from_raster(dem_filepath)

        part_df = df[
            (df["LAT"] < grid.bbox[3]) &
            (df["LAT"] > grid.bbox[1]) &
            (df["LON"] < grid.bbox[2]) &
            (df["LON"] > grid.bbox[0])
            ]
        if len(part_df) == 0:
            continue

        dem = grid.read_raster(dem_filepath)
        pit_filled_dem = grid.fill_pits(dem)
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        flowdir = grid.flowdir(inflated_dem)
        acc = grid.accumulation(flowdir)
        inflated_dem = cv2.GaussianBlur(inflated_dem, kernel_size, sigma)

        res = part_df.copy()
        res["stream_row"] = np.nan
        res["stream_col"] = np.nan
        res["ridge_row"] = np.nan
        res["ridge_col"] = np.nan
        res["alt_stream"] = np.nan
        res["dist_stream"] = np.nan
        res["alt_top"] = np.nan
        res["dist_top"] = np.nan
        res["ratio_alt"] = np.nan
        res["ratio_dist"] = np.nan
        res["ratio_stream"] = np.nan
        res["ratio_top"] = np.nan
        res["altitude"] = np.nan
        res["accumulation"] = np.nan

        geomorphon_filepath = osp.join(
            __datadir__, 'results', 'geomorphons', f'{country}',
            f'{country}_geomorphon_{index:03d}.tif'
            )
        with rasterio.open(geomorphon_filepath) as dataset:
            geomorphon = dataset.read(1)
            geomorphon = np.maximum(geomorphon, 0)

        mask_black = geomorphon < 4
        labeled_array, num_features = label(mask_black)

        filtered_mask = np.zeros_like(mask_black)

        for region in regionprops(labeled_array):
            if region.area >= min_size:
                coords = region.coords
                acc_values = acc[coords[:, 0], coords[:, 1]]
                valid_coords = coords[acc_values < 2]
                filtered_mask[valid_coords[:, 0], valid_coords[:, 1]] = 1

        ridges = geomorphon * filtered_mask
        ridges = np.minimum(ridges, 1)

        streams = np.where(acc > threshold, 1, 0)

        n, p = inflated_dem.shape
        num = len(res.index)
        for i, indice in enumerate(res.index):
            try:
                col_point, row_point = convert_coord_to_indices(
                    res.loc[indice, "LON"], res.loc[indice, "LAT"], grid.affine
                )

                col_min = max(0, col_point - size)
                row_max = min(n, row_point + size)
                row_min = max(0, row_point - size)
                col_max = min(p, col_point + size)
                ones_indices = np.argwhere(
                    ridges[row_min:row_max, col_min:col_max] == 1
                )
                distances = np.sqrt(
                    (ones_indices[:, 0] - row_point + row_min) ** 2 +
                    (ones_indices[:, 1] - col_point + col_min) ** 2
                )
                nearest_index = np.argmin(distances)
                nearest_point = ones_indices[nearest_index]
                ridge_point_row, ridge_point_col = (
                    nearest_point[0] + row_min,
                    nearest_point[1] + col_min,
                    )

                res.at[indice, "ridge_row"] = ridge_point_row
                res.at[indice, "ridge_col"] = ridge_point_col

                ones_indices = np.argwhere(
                    streams[row_min:row_max, col_min:col_max] == 1
                    )
                distances = np.sqrt(
                    (ones_indices[:, 0] - row_point + row_min) ** 2 +
                    (ones_indices[:, 1] - col_point + col_min) ** 2
                    )
                nearest_index = np.argmin(distances)
                nearest_point = ones_indices[nearest_index]
                stream_point_row, stream_point_col = (
                    nearest_point[0] + row_min,
                    nearest_point[1] + col_min,
                    )

                res.at[indice, "stream_row"] = stream_point_row
                res.at[indice, "stream_col"] = stream_point_col

                stream_points = bresenham_line(
                    x0=row_point, y0=col_point,
                    x1=ridge_point_row, y1=ridge_point_col
                    )
                stream_line_points = np.array(
                    [[row, col] for row, col in stream_points]
                    )
                top_points = bresenham_line(
                    x0=row_point, y0=col_point,
                    x1=stream_point_row, y1=stream_point_col
                    )
                top_line_points = np.array(
                    [[row, col] for row, col in top_points]
                    )

                dem_point = inflated_dem[row_point, col_point]
                dem_stream = inflated_dem[stream_point_row, stream_point_col]
                dem_ridge = inflated_dem[ridge_point_row, ridge_point_col]

                res.at[indice, "alt_stream"] = dem_point - dem_stream
                res.at[indice, "alt_top"] = dem_ridge - dem_point

                dist_stream = (
                    np.sqrt(
                        (row_point - stream_point_row) ** 2 +
                        (col_point - stream_point_col) ** 2) + 1
                )

                dist_ridge = (
                    np.sqrt(
                        (row_point - ridge_point_row) ** 2 +
                        (col_point - ridge_point_col) ** 2) + 1
                )

                res.at[indice, "dist_stream"] = dist_stream
                res.at[indice, "dist_top"] = dist_ridge

                res.at[indice, "ratio_alt"] = (
                    (dem_point - dem_stream) / (dem_ridge - dem_point + 0.1))
                res.at[indice, "ratio_dist"] = (
                    dist_stream / dist_ridge)
                res.at[indice, "ratio_stream"] = (
                    (dem_point - dem_stream) / dist_stream)
                res.at[indice, "ratio_top"] = (
                    (dem_ridge - dem_point) / dist_ridge)

                res.at[indice, "altitude"] = inflated_dem[row_point, col_point]
                res.at[indice, "accumulation"] = acc[row_point, col_point]
            except Exception as e:
                print(e)
                continue
