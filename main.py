import os
import rasterio
import numpy as np
from pysheds.grid import Grid
import pandas as pd
import cv2
from whitebox_workflows import WbEnvironment
from scipy.ndimage import label
from skimage.measure import regionprops
import pickle
import glob


def list_folders(directory):
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]


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


def convert_indices_to_coords(col, row, grid_affine):
    lon = grid_affine[0] * col + grid.affine[2]
    lat = grid_affine[4] * row + grid_affine[5]
    return lon, lat


def convert_coord_to_indices(lon, lat, grid_affine):
    col = round((lon - grid_affine[2]) / grid_affine[0])
    row = round((lat - grid_affine[5]) / grid_affine[4])
    return col, row


min_size = 70
kernel_size = (5, 5)
sigma = 1.0
threshold = 1500
size = 200

with open("./models/model.pkl", "rb") as file:
    model = pickle.load(file)

folder = "./data/results/temp/"
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
map_folder = "./data/results/"
map_csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
directory_path = "./data/dem/"
countries = list_folders(directory_path)

for country in countries:
    with rasterio.open(
        f"./data/pixels/{country}/{country}_rainfedcropland_ls.tif"
    ) as dataset:
        pixels_of_interest = dataset.read(1)
        pixels_of_interest = np.maximum(pixels_of_interest, 0)

    grid = Grid.from_raster(
        f"./data/pixels/{country}/{country}_rainfedcropland_ls.tif", data_name="map"
    )

    rows, cols = np.where(pixels_of_interest == 1)
    data = convert_indices_to_coords(rows, cols, grid.affine)
    df = pd.DataFrame(columns=["LON", "LAT"])
    df["LON"] = data[0]
    df["LAT"] = data[1]

    folder_path = f"./data/dem/{country}/"

    dem_files = os.listdir(folder_path)
    dem_files = [f for f in dem_files if os.path.isfile(os.path.join(folder_path, f))]

    for index in range(len(dem_files)):
        if f"map_{country}_{index}.csv" in csv_files:
            continue
        grid = Grid.from_raster(folder_path + dem_files[index])
        part_df = df[
            (df["LAT"] < grid.bbox[3])
            & (df["LAT"] > grid.bbox[1])
            & (df["LON"] < grid.bbox[2])
            & (df["LON"] > grid.bbox[0])
        ]
        if len(part_df) == 0:
            continue
        dem = grid.read_raster(folder_path + dem_files[index])
        pit_filled_dem = grid.fill_pits(dem)
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem)
        acc = grid.accumulation(fdir)
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

        wbe = WbEnvironment()

        dem = wbe.read_raster(folder_path + dem_files[index])

        dem = wbe.fill_missing_data(dem, filter_size=35, exclude_edge_nodata=True)

        dem = wbe.gaussian_filter(dem, sigma=2)

        dem = wbe.fill_missing_data(dem, filter_size=35, exclude_edge_nodata=True)

        wbe.write_raster(
            wbe.geomorphons(
                dem,
                search_distance=100,
                flatness_threshold=1.0,
                flatness_distance=0,
                skip_distance=0,
                output_forms=True,
                analyze_residuals=True,
            ),
            f"./data/results/temp/{country}_map_geomorphons.tif",
            compress=True,
        )

        with rasterio.open(
            f"./data/results/temp/{country}_map_geomorphons.tif"
        ) as dataset:
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

                col_min, row_max = max(0, col_point - size), min(n, row_point + size)
                row_min, col_max = max(0, row_point - size), min(p, col_point + size)
                ones_indices = np.argwhere(
                    ridges[row_min:row_max, col_min:col_max] == 1
                )
                distances = np.sqrt(
                    (ones_indices[:, 0] - row_point + row_min) ** 2
                    + (ones_indices[:, 1] - col_point + col_min) ** 2
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
                    (ones_indices[:, 0] - row_point + row_min) ** 2
                    + (ones_indices[:, 1] - col_point + col_min) ** 2
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
                    x0=row_point, y0=col_point, x1=ridge_point_row, y1=ridge_point_col
                )
                stream_line_points = np.array(
                    [[row, col] for row, col in stream_points]
                )

                top_points = bresenham_line(
                    x0=row_point, y0=col_point, x1=stream_point_row, y1=stream_point_col
                )
                top_line_points = np.array([[row, col] for row, col in top_points])

                dem_point = inflated_dem[row_point, col_point]
                dem_stream = inflated_dem[stream_point_row, stream_point_col]
                dem_ridge = inflated_dem[ridge_point_row, ridge_point_col]

                res.at[indice, "alt_stream"] = dem_point - dem_stream
                res.at[indice, "alt_top"] = dem_ridge - dem_point

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

                res.at[indice, "dist_stream"] = dist_stream
                res.at[indice, "dist_top"] = dist_ridge

                res.at[indice, "ratio_alt"] = (dem_point - dem_stream) / (
                    dem_ridge - dem_point + 0.1
                )
                res.at[indice, "ratio_dist"] = dist_stream / dist_ridge
                res.at[indice, "ratio_stream"] = (dem_point - dem_stream) / dist_stream
                res.at[indice, "ratio_top"] = (dem_ridge - dem_point) / dist_ridge

                res.at[indice, "altitude"] = inflated_dem[row_point, col_point]
                res.at[indice, "accumulation"] = acc[row_point, col_point]
            except Exception as e:
                print(e)
                continue

        new_res = res.dropna()
        X = new_res[
            [
                "alt_stream",
                "dist_stream",
                "alt_top",
                "dist_top",
                "ratio_alt",
                "ratio_dist",
                "ratio_stream",
                "ratio_top",
                "altitude",
                "accumulation",
            ]
        ].astype("float")
        X["dist_stream_inverse"] = 1 / (X["dist_stream"] + 1)
        X = X.drop(columns=["dist_stream"])

        Y = model.predict(X)

        new_res["Predicted_NS"] = Y

        new_res.to_csv("./data/results/temp/" + f"map_{country}_{index}.csv")

    if f"map_{country}.csv" in map_csv_files:
        continue
    fichiers_csv = glob.glob(f"{folder}/*{country}*.csv")

    dfs = [pd.read_csv(fichier, index_col=0) for fichier in fichiers_csv]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df[["LON", "LAT", "Predicted_NS"]]

    merged_df.to_csv("./data/results/" + f"map_{country}.csv")
