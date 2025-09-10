#!/usr/bin/env python
# coding: utf-8

import rasterio
from scipy import stats
import numpy as np
from rasterio.transform import rowcol
import pandas as pd
from whitebox import WhiteboxTools
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label
from skimage.measure import regionprops
import os
import pandas as pd


def array_to_cord(transform, row, col):
    """
    Convert array indices to geographic coordinates.

    Parameters:
    transform (Affine): The affine transformation object.
    row (int): The row index in the array.
    col (int): The column index in the array.

    Returns:
    tuple: The longitude and latitude coordinates.
    """
    lon, lat = transform * (col, row)
    return lon, lat


def coord_to_array(transform, lon, lat):
    """
    Convert geographic coordinates to array indices.

    Parameters:
    transform (Affine): The affine transformation object.
    lon (float): The longitude coordinate.
    lat (float): The latitude coordinate.

    Returns:
    tuple: The column and row indices in the array.
    """
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    return col, row


def bresenham_line(row0, col0, row1, col1):
    """
    Generate points along a line using Bresenham's algorithm.

    Parameters:
    row0 (int): The starting row index.
    col0 (int): The starting column index.
    row1 (int): The ending row index.
    col1 (int): The ending column index.

    Returns:
    list: A list of points (row, col) along the line.
    """
    points = []
    drow = abs(row1 - row0)
    dcol = abs(col1 - col0)
    srow = 1 if row0 < row1 else -1
    scol = 1 if col0 < col1 else -1
    err = drow - dcol

    while True:
        points.append((row0, col0))
        if row0 == row1 and col0 == col1:
            break
        e2 = err * 2
        if e2 > -dcol:
            err -= dcol
            row0 += srow
        if e2 < drow:
            err += drow
            col0 += scol

    return points


def new_bresenham_line(row0, col0, row1, col1, thickness=1):
    """
    Generate points along a thick line using Bresenham's algorithm.

    Parameters:
    row0 (int): The starting row index.
    col0 (int): The starting column index.
    row1 (int): The ending row index.
    col1 (int): The ending column index.
    thickness (int): The thickness of the line.

    Returns:
    list: A list of points (row, col) along the thick line.
    """
    points = []
    drow = abs(row1 - row0)
    dcol = abs(col1 - col0)
    srow = 1 if row0 < row1 else -1
    scol = 1 if col0 < col1 else -1
    err = drow - dcol

    # Flag to avoid adding the contour of the first point.
    is_first_point = True

    while True:
        if is_first_point:
            # Add only the central point for the first point
            is_first_point = False
        else:
            # Add neighboring points for subsequent points
            for dr in range(-thickness, thickness + 1):
                for dc in range(-thickness, thickness + 1):
                    points.append((row0 + dr, col0 + dc))

        if row0 == row1 and col0 == col1:
            break

        e2 = err * 2
        if e2 > -dcol:
            err -= dcol
            row0 += srow
        if e2 < drow:
            err += drow
            col0 += scol

    return list(set(points))


# %%
# List of countries and corresponding date methods.
dem_countries = ["Benin", "Burkina", "Guinee"]

# Indicates the type of the column "DATE" in the csv.
date_methods = ["datetime", "str", "datetime"]


# Main processing loop for each country.
for training_num in range(len(dem_countries)):
    dem_country = dem_countries[training_num]

    # Load training data.
    training_df = pd.read_excel(f"Training_data/{dem_country}.xlsx")

    # Filter data based on date method (Meteorological data
    # are available from 2002).
    if date_methods[training_num] == "datetime":
        training_df["DATE"] = pd.to_datetime(
            training_df["DATE"], errors="coerce")
        training_df = training_df[
            (training_df["DATE"].dt.year > 2002) &
            (training_df["DATE"].dt.year < 2025)
        ]
    elif date_methods[training_num] == "str":
        training_df = training_df[
            training_df["DATE"].apply(
                lambda row: int(row.split("/")[2])
                ) > 2002
        ]
    else:
        training_df = training_df[training_df["DATE"] > 2002]

    # Load DEM data
    with rasterio.open(f"DEM/{dem_country}/{dem_country}.tif") as src:
        transform = src.transform
        area = src.read(1)
        width = src.width
        height = src.height

    # Define maximum dimensions and padding to load arrays of
    # reasonable size in RAM.
    max_width = 5000
    max_height = 5000
    padding = 100

    # Calculate the number of tiles.
    i_max = height // max_height + int(height % max_height != 0)
    j_max = width // max_width + int(width % max_width != 0)
    print(i_max, j_max)

    # Initialize dictionaries for training points and borders.
    training_points = {}
    training_borders = {}

    for i in range(i_max):
        for j in range(j_max):
            row_min = max(0, i * max_height)
            col_min = max(0, j * max_width)
            row_max = min(height, row_min + max_height)
            col_max = min(width, col_min + max_width)
            training_borders[row_min, row_max, col_min, col_max] = (i, j)
            training_points[i, j] = []

    # Assign training points to tiles
    coords = zip(training_df.index, training_df["LON"], training_df["LAT"])
    for index, lon, lat in coords:
        col, row = coord_to_array(transform=transform, lon=lon, lat=lat)
        for row_min, row_max, col_min, col_max in training_borders.keys():
            if row_min <= row < row_max and col_min <= col < col_max:
                training_points[
                    training_borders[row_min, row_max, col_min, col_max]
                ].append((index, lon, lat))

    for key in training_points.keys():
        print(f"Number of points in area {key}: {len(training_points[key])}")

    # Process each tile
    for i in range(i_max):
        for j in range(j_max):
            print(f"Currently working country: {dem_country} and area: {i, j}")
            if len(training_points[i, j]) == 0:
                # There is no point in this tile.
                continue

            for accumulation_threshold in [1500, 3000]:
                # Hyperparameter for acccumlation threshold to extract streams.
                for ridge_size in [30]:
                    # Hyperparameter to identify the ridges (The higher
                    # this parameter is, the fewer points are identified as
                    # ridges).

                    file_path = (
                        f"Topo_features_{dem_country}_{accumulation_threshold}"
                        "_{ridge_size}_{i}_{j}.csv")
                    if os.path.exists(file_path):
                        print("The file already exists")
                        continue

                    row_min = max(0, i * max_height)
                    col_min = max(0, j * max_width)
                    row_max = min(height, row_min + max_height)
                    col_max = min(width, col_min + max_width)

                    # Load DEM data for the tile
                    tifname = f"DEM/{dem_country}/{dem_country}.tif"
                    with rasterio.open(tifname) as src:
                        transform = src.transform

                        # The window that loads only the tile with a
                        # padding around to remove border effects.
                        window = rasterio.windows.Window(
                            max(0, col_min - padding),
                            max(0, row_min - padding),
                            (col_min - max(0, col_min - padding))
                            + col_max
                            - col_min
                            + (min(width, col_max + padding) - col_max),
                            (row_min - max(0, row_min - padding))
                            + row_max
                            - row_min
                            + (min(height, row_max + padding) - row_max),
                        )
                        dem = src.read(1, window=window)

                        results = []

                        output_path = f"Temporary_dem_{dem_country}_{i}_{j}.tif"

                        new_transform = src.window_transform(window)

                        new_meta = src.meta.copy()
                        new_meta.update(
                            {
                                "height": dem.shape[0],
                                "width": dem.shape[1],
                                "transform": new_transform,
                            }
                        )

                        with rasterio.open(output_path, "w", **new_meta) as dst:
                            dst.write(dem, 1)

                    # Initialize WhiteboxTools
                    wbt = WhiteboxTools()

                    # Fill depressions
                    wbt.fill_depressions(
                        f"C:/Users/aziza/Desktop/Map/Temporary_dem_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_filled_dem_{dem_country}_{i}_{j}.tif",
                    )

                    # Apply Gaussian filter
                    wbt.gaussian_filter(
                        i=f"C:/Users/aziza/Desktop/Map/Temporary_filled_dem_{dem_country}_{i}_{j}.tif",
                        output=f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif",
                        sigma=4.0,
                    )

                    # Calculate flow accumulation
                    wbt.d8_flow_accumulation(
                        f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_flow_accum_{dem_country}_{i}_{j}.tif",
                        out_type="cells",
                    )

                    # Extract streams
                    wbt.extract_streams(
                        f"C:/Users/aziza/Desktop/Map/Temporary_flow_accum_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_streams_{dem_country}_{i}_{j}.tif",
                        threshold=accumulation_threshold,
                    )

                    # Calculate geomorphons
                    wbt.geomorphons(
                        f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_geomorphons_{dem_country}_{i}_{j}.tif",
                    )

                    # Calculate slope
                    wbt.slope(
                        f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_slope_{dem_country}_{i}_{j}.tif",
                    )

                    # Calculate profile curvature
                    wbt.profile_curvature(
                        f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif",
                        f"C:/Users/aziza/Desktop/Map/Temporary_curvature_{dem_country}_{i}_{j}.tif",
                    )

                    # Load smoothed DEM and other derived layers
                    with rasterio.open(
                        f"C:/Users/aziza/Desktop/Map/Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif"
                    ) as src:
                        new_transform = src.transform
                        new_width = src.width
                        new_height = src.height
                        smoothed_dem = src.read(1)
                        cols, rows = np.meshgrid(
                            np.arange(new_width), np.arange(new_height)
                        )
                        longitudes, latitudes = rasterio.transform.xy(
                            new_transform, rows, cols, offset="center"
                        )
                        longitudes = np.array(longitudes).reshape(smoothed_dem.shape)
                        latitudes = np.array(latitudes).reshape(smoothed_dem.shape)

                    with rasterio.open(
                        f"C:/Users/aziza/Desktop/Map/Temporary_streams_{dem_country}_{i}_{j}.tif"
                    ) as dataset:
                        streams = dataset.read(1)

                    with rasterio.open(
                        f"C:/Users/aziza/Desktop/Map/Temporary_slope_{dem_country}_{i}_{j}.tif"
                    ) as dataset:
                        grad = dataset.read(1)

                    with rasterio.open(
                        f"C:/Users/aziza/Desktop/Map/Temporary_curvature_{dem_country}_{i}_{j}.tif"
                    ) as dataset:
                        hessian = dataset.read(1)

                    with rasterio.open(
                        f"C:/Users/aziza/Desktop/Map/Temporary_geomorphons_{dem_country}_{i}_{j}.tif"
                    ) as dataset:
                        band1 = dataset.read(1)
                        band1 = np.maximum(band1, 0)

                        mask_black = band1 < 4
                        labeled_array, num_features = label(mask_black)

                        filtered_mask = np.zeros_like(mask_black)

                        for region in regionprops(labeled_array): #Some filtering is applied to correctly identified the ridges
                            if region.area >= ridge_size:
                                for coord in region.coords:
                                    filtered_mask[coord[0], coord[1]] = 1

                        ridges = band1 * filtered_mask

                        ridges = np.minimum(ridges, 1)

                        binary_ridges = ridges > 0

                        thin_ridges = skeletonize(binary_ridges)

                        ridges = thin_ridges.astype(int)

                        thin_ridges_cleaned = remove_small_objects(
                            ridges.astype(bool), min_size=2
                        )

                        ridges = thin_ridges_cleaned.astype(int)

                    # Initialize new columns in the training dataframe
                    training_df["precipitation"] = 0
                    training_df["ndvi"] = 0
                    training_df["ridge_row"] = 0
                    training_df["ridge_col"] = 0
                    training_df["stream_row"] = 0
                    training_df["stream_col"] = 0
                    training_df["alt_stream"] = 0
                    training_df["alt_top"] = 0
                    training_df["dist_stream"] = 0
                    training_df["dist_top"] = 0
                    training_df["ratio_alt"] = 0
                    training_df["ratio_stream"] = 0
                    training_df["ratio_alt"] = 0
                    training_df["ratio_dist"] = 0

                    training_df["stream_grad_mean"] = 0
                    training_df["stream_grad_var"] = 0
                    training_df["stream_grad_skew"] = 0
                    training_df["stream_grad_kurt"] = 0
                    training_df["stream_grad_max"] = 0
                    training_df["stream_hessian_mean"] = 0
                    training_df["stream_hessian_var"] = 0
                    training_df["stream_hessian_skew"] = 0
                    training_df["stream_hessian_kurt"] = 0
                    training_df["stream_hessian_max"] = 0

                    training_df["ridge_grad_mean"] = 0
                    training_df["ridge_grad_var"] = 0
                    training_df["ridge_grad_skew"] = 0
                    training_df["ridge_grad_kurt"] = 0
                    training_df["ridge_grad_max"] = 0
                    training_df["ridge_hessian_mean"] = 0
                    training_df["ridge_hessian_var"] = 0
                    training_df["ridge_hessian_skew"] = 0
                    training_df["ridge_hessian_kurt"] = 0
                    training_df["ridge_hessian_max"] = 0

                    training_df["short_grad_mean"] = 0
                    training_df["short_grad_var"] = 0
                    training_df["short_grad_skew"] = 0
                    training_df["short_grad_kurt"] = 0
                    training_df["short_grad_max"] = 0
                    training_df["short_hessian_mean"] = 0
                    training_df["short_hessian_var"] = 0
                    training_df["short_hessian_skew"] = 0
                    training_df["short_hessian_kurt"] = 0
                    training_df["short_hessian_max"] = 0

                    training_df["long_grad_mean"] = 0
                    training_df["long_grad_var"] = 0
                    training_df["long_grad_skew"] = 0
                    training_df["long_grad_kurt"] = 0
                    training_df["long_grad_max"] = 0
                    training_df["long_hessian_mean"] = 0
                    training_df["long_hessian_var"] = 0
                    training_df["long_hessian_skew"] = 0
                    training_df["long_hessian_kurt"] = 0
                    training_df["long_hessian_max"] = 0

                    size = 100
                    n, p = smoothed_dem.shape

                    error_points = set()

                    features = []

                    # Process each training point
                    for index, lon, lat in training_points[i, j]:
                        try:
                            row_point, col_point = rowcol(new_transform, lon, lat)

                            col_mi, row_ma = max(0, col_point - size), min(
                                n, row_point + size
                            )
                            row_mi, col_ma = max(0, row_point - size), min(
                                p, col_point + size
                            )

                            ones_indices = np.argwhere(
                                ridges[row_mi:row_ma, col_mi:col_ma] == 1
                            )
                            distances = np.sqrt(
                                (ones_indices[:, 0] - row_point + row_mi) ** 2
                                + (ones_indices[:, 1] - col_point + col_mi) ** 2
                            )
                            sorted_indices = np.argsort(distances)

                            ridge_point_row, ridge_point_col = None, None

                            for idx in sorted_indices:
                                nearest_point = ones_indices[idx]
                                candidate_row, candidate_col = (
                                    nearest_point[0] + row_mi,
                                    nearest_point[1] + col_mi,
                                )

                                if streams[row_point, col_point] == 1:
                                    ridge_point_row, ridge_point_col = (
                                        candidate_row,
                                        candidate_col,
                                    )
                                    break

                                ridge_points = np.array(
                                    [
                                        [row, col]
                                        for row, col in new_bresenham_line(
                                            row0=row_point,
                                            col0=col_point,
                                            row1=candidate_row,
                                            col1=candidate_col,
                                        )
                                    ]
                                )

                                # Check if the line crosses a stream point
                                if not any(
                                    streams[row, col] == 1
                                    for row, col in ridge_points[1:]
                                ):
                                    ridge_point_row, ridge_point_col = (
                                        candidate_row,
                                        candidate_col,
                                    )
                                    break

                            if ridge_point_row is None or ridge_point_col is None:
                                error_points.add(index)
                                ridge_point_row, ridge_point_col = 0, 0

                            ones_indices = np.argwhere(
                                streams[row_mi:row_ma, col_mi:col_ma] == 1
                            )
                            distances = np.sqrt(
                                (ones_indices[:, 0] - row_point + row_mi) ** 2
                                + (ones_indices[:, 1] - col_point + col_mi) ** 2
                            )
                            nearest_index = np.argmin(distances)
                            nearest_point = ones_indices[nearest_index]
                            stream_point_row, stream_point_col = (
                                nearest_point[0] + row_mi,
                                nearest_point[1] + col_mi,
                            )

                            ridge_points = np.array(
                                [
                                    [row, col]
                                    for row, col in bresenham_line(
                                        row0=row_point,
                                        col0=col_point,
                                        row1=candidate_row,
                                        col1=candidate_col,
                                    )
                                ]
                            )

                            stream_points = np.array(
                                [
                                    [row, col]
                                    for row, col in bresenham_line(
                                        row0=row_point,
                                        col0=col_point,
                                        row1=stream_point_row,
                                        col1=stream_point_col,
                                    )
                                ]
                            )

                            training_df.at[index, "ridge_row"] = ridge_point_row
                            training_df.at[index, "ridge_col"] = ridge_point_col

                            training_df.at[index, "stream_row"] = stream_point_row
                            training_df.at[index, "stream_col"] = stream_point_col

                            dem_point = smoothed_dem[row_point, col_point]
                            dem_stream = smoothed_dem[
                                stream_point_row, stream_point_col
                            ]
                            dem_ridge = smoothed_dem[ridge_point_row, ridge_point_col]

                            training_df.at[index, "alt_stream"] = dem_point - dem_stream
                            training_df.at[index, "alt_top"] = dem_ridge - dem_point

                            dist_stream = (
                                np.sqrt(
                                    (row_point - stream_point_row) ** 2
                                    + (col_point - stream_point_col) ** 2
                                )
                                + 1
                            )
                            dist_ridge = (
                                np.sqrt(
                                    (row_point - ridge_point_row) ** 2
                                    + (col_point - ridge_point_col) ** 2
                                )
                                + 1
                            )

                            training_df.at[index, "dist_stream"] = dist_stream
                            training_df.at[index, "dist_top"] = dist_ridge

                            training_df.at[index, "ratio_alt"] = (
                                dem_point - dem_stream
                            ) / (dem_ridge - dem_point + 0.1)
                            training_df.at[index, "ratio_dist"] = (
                                dist_stream / dist_ridge
                            )
                            training_df.at[index, "ratio_stream"] = (
                                dem_point - dem_stream
                            ) / dist_stream
                            training_df.at[index, "ratio_top"] = (
                                dem_ridge - dem_point
                            ) / dist_ridge

                            center_row, center_col = row_point, col_point
                            max_short_distance = 5
                            max_long_distance = 20
                            row_mi = max(center_row - max_long_distance, 0)
                            row_ma = min(
                                center_row + max_long_distance + 2,
                                smoothed_dem.shape[0],
                            )
                            col_mi = max(center_col - max_long_distance, 0)
                            col_ma = min(
                                center_col + max_long_distance + 2,
                                smoothed_dem.shape[1],
                            )

                            ROWS, COLS = np.meshgrid(
                                np.arange(row_mi, row_ma),
                                np.arange(col_mi, col_ma),
                                indexing="ij",
                            )
                            distances = np.sqrt(
                                (ROWS - center_row) ** 2 + (COLS - center_col) ** 2
                            )
                            short_distance = np.argwhere(
                                distances <= max_short_distance
                            ) + np.array([row_mi, col_mi])
                            long_distance = np.argwhere(
                                distances <= max_long_distance
                            ) + np.array([row_mi, col_mi])

                            grad_stream = grad[stream_points[:, 0], stream_points[:, 1]]
                            hessian_stream = hessian[
                                stream_points[:, 0], stream_points[:, 1]
                            ]
                            training_df.at[index, "stream_grad_mean"] = np.mean(
                                grad_stream
                            )
                            training_df.at[index, "stream_grad_var"] = np.mean(
                                grad_stream
                            )
                            training_df.at[index, "stream_grad_skew"] = stats.skew(
                                grad_stream
                            )
                            training_df.at[index, "stream_grad_kurt"] = stats.kurtosis(
                                grad_stream
                            )
                            training_df.at[index, "stream_grad_max"] = np.max(
                                grad_stream
                            )
                            training_df.at[index, "stream_hessian_mean"] = np.mean(
                                hessian_stream
                            )
                            training_df.at[index, "stream_hessian_var"] = np.mean(
                                hessian_stream
                            )
                            training_df.at[index, "stream_hessian_skew"] = stats.skew(
                                hessian_stream
                            )
                            training_df.at[
                                index, "stream_hessian_kurt"
                            ] = stats.kurtosis(hessian_stream)
                            training_df.at[index, "stream_hessian_max"] = np.max(
                                hessian_stream
                            )

                            grad_ridge = grad[ridge_points[:, 0], ridge_points[:, 1]]
                            hessian_ridge = hessian[
                                ridge_points[:, 0], ridge_points[:, 1]
                            ]
                            training_df.at[index, "ridge_grad_mean"] = np.mean(
                                grad_ridge
                            )
                            training_df.at[index, "ridge_grad_var"] = np.mean(
                                grad_ridge
                            )
                            training_df.at[index, "ridge_grad_skew"] = stats.skew(
                                grad_ridge
                            )
                            training_df.at[index, "ridge_grad_kurt"] = stats.kurtosis(
                                grad_ridge
                            )
                            training_df.at[index, "ridge_grad_max"] = np.max(grad_ridge)
                            training_df.at[index, "ridge_hessian_mean"] = np.mean(
                                hessian_ridge
                            )
                            training_df.at[index, "ridge_hessian_var"] = np.mean(
                                hessian_ridge
                            )
                            training_df.at[index, "ridge_hessian_skew"] = stats.skew(
                                hessian_ridge
                            )
                            training_df.at[
                                index, "ridge_hessian_kurt"
                            ] = stats.kurtosis(hessian_ridge)
                            training_df.at[index, "ridge_hessian_max"] = np.max(
                                hessian_ridge
                            )

                            grad_short = grad[
                                short_distance[:, 0], short_distance[:, 1]
                            ]
                            hessian_short = hessian[
                                short_distance[:, 0], short_distance[:, 1]
                            ]
                            training_df.at[index, "short_grad_mean"] = np.mean(
                                grad_short
                            )
                            training_df.at[index, "short_grad_var"] = np.mean(
                                grad_short
                            )
                            training_df.at[index, "short_grad_skew"] = stats.skew(
                                grad_short
                            )
                            training_df.at[index, "short_grad_kurt"] = stats.kurtosis(
                                grad_short
                            )
                            training_df.at[index, "short_grad_max"] = np.max(grad_short)
                            training_df.at[index, "short_hessian_mean"] = np.mean(
                                hessian_short
                            )
                            training_df.at[index, "short_hessian_var"] = np.mean(
                                hessian_short
                            )
                            training_df.at[index, "short_hessian_skew"] = stats.skew(
                                hessian_short
                            )
                            training_df.at[
                                index, "short_hessian_kurt"
                            ] = stats.kurtosis(hessian_short)
                            training_df.at[index, "short_hessian_max"] = np.max(
                                hessian_short
                            )

                            grad_long = grad[long_distance[:, 0], long_distance[:, 1]]
                            hessian_long = hessian[
                                long_distance[:, 0], long_distance[:, 1]
                            ]
                            training_df.at[index, "long_grad_mean"] = np.mean(grad_long)
                            training_df.at[index, "long_grad_var"] = np.mean(grad_long)
                            training_df.at[index, "long_grad_skew"] = stats.skew(
                                grad_long
                            )
                            training_df.at[index, "long_grad_kurt"] = stats.kurtosis(
                                grad_long
                            )
                            training_df.at[index, "long_grad_max"] = np.max(grad_long)
                            training_df.at[index, "long_hessian_mean"] = np.mean(
                                hessian_long
                            )
                            training_df.at[index, "long_hessian_var"] = np.mean(
                                hessian_long
                            )
                            training_df.at[index, "long_hessian_skew"] = stats.skew(
                                hessian_long
                            )
                            training_df.at[index, "long_hessian_kurt"] = stats.kurtosis(
                                hessian_long
                            )
                            training_df.at[index, "long_hessian_max"] = np.max(
                                hessian_long
                            )
                        except Exception as e:
                            error_points.add(index)

                    # Filter out rows with zero values for ridge_row and ridge_col
                    training_df = training_df[
                        (training_df["ridge_row"] != 0)
                        | (training_df["ridge_row"] != 0)
                    ]

                    # Save the processed data to a CSV file
                    training_df.to_csv(
                        f"Topo_features_{dem_country}_{accumulation_threshold}_{ridge_size}_{i}_{j}.csv"
                    )

                    # Remove temporary files
                    os.remove(f"Temporary_dem_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_filled_dem_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_smoothed_dem_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_flow_accum_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_streams_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_geomorphons_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_slope_{dem_country}_{i}_{j}.tif")
                    os.remove(f"Temporary_curvature_{dem_country}_{i}_{j}.tif")

