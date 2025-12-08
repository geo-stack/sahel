# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel_water_table_ml
# =============================================================================

"""Zonal data and stats extraction capability."""

# ---- Standard imports
from pathlib import Path

# ---- Third party imports
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from osgeo import gdal

gdal.UseExceptions()


def build_zonal_index_map(
        raster_path: Path,
        basin_gdf: gpd.GeoDataFrame,
        ) -> tuple[dict, list]:
    """
    Build an index map for basin geometries relative to a raster grid.

    For each basin, this computes the absolute pixel indices (rows, cols) that
    fall inside the basin geometry.  The returned dictionary contains raster
    metadata necessary to validate that other rasters are on the same grid.

    By default, only pixels whose centers fall within the basin geometry
    are included (all_touched=False), which is standard practice for
    hydrological zonal statistics. This ensures accurate areal weighting
    and prevents double-counting of pixels at basin boundaries.

    For very small basins that do not contain any pixel centers, the function
    falls back to all_touched=True, which includes any pixel intersected by
    the basin polygon.  This avoids empty results at the cost of minor edge
    over-sampling.  Basins using this fallback are flagged in the returned
    metadata.

    Parameters
    ----------
    raster_path : Path
        Path to a representative raster (VRT/TIFF) that defines the grid.
    basin_gdf : gpd.GeoDataFrame
        GeoDataFrame containing basin geometries, indexed by basin ID.
        Must be in the same CRS as the raster.

    Returns
    -------
    zonal_index_map : dict
        Dictionary with the following structure:
        {
            'width': int,
                Raster width in pixels.
            'height': int,
                Raster height in pixels.
            'crs': str,
                Raster CRS as string (WKT or PROJ).
            'indices': dict[int, np.ndarray],
                Mapping of basin_id -> Nx2 array of [row, col] indices.
                Each array contains absolute row/col indices for pixels inside
                the basin. Basins that don't intersect the raster are omitted.
        }
    basin_metadata : dict
        Dictionary with keys:
        {
            'bad': list[int],
                Basin IDs that did not intersect the raster or had invalid
                geometry.
            'small': list[int],
                Basin IDs that required all_touched=True fallback due to
                small size relative to grid resolution.
        }
    """
    import rasterio
    from rasterio.features import geometry_window

    indices_for_geoms = {}

    # Basins that don't intersect or have invalid geometry.
    bad_basin_ids = []

    # Basins that needed all_touched=True fallback.
    small_basin_ids = []

    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        crs = src.crs

        for index, row in basin_gdf.iterrows():

            geom = row.geometry

            try:
                # Get the minimal window covering the geometry (speeds up
                # rasterize)
                win = geometry_window(src, [geom], pad_x=0, pad_y=0)
            except Exception:
                # Geometry does not intersect raster or other failure.
                bad_basin_ids.append(index)
                continue

            win_height = int(win.height)
            win_width = int(win.width)

            if win_height <= 0 or win_width <= 0:
                bad_basin_ids.append(index)
                continue

            # Rasterize the geometry into the window coordinates.
            win_transform = src.window_transform(win)

            mask = rasterize(
                [(geom, 1)],
                out_shape=(win_height, win_width),
                transform=win_transform,
                fill=0,
                all_touched=False,
                dtype='uint8'
                )

            if not mask.any():
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(win_height, win_width),
                    transform=win_transform,
                    fill=0,
                    all_touched=True,
                    dtype='uint8'
                    )
                small_basin_ids.append(index)

            if not mask. any():
                bad_basin_ids.append(index)
                continue

            # rows/cols relative to window.
            rows, cols = np.nonzero(mask)

            # Convert to absolute row/col on the full raster
            abs_rows = rows + int(win.row_off)
            abs_cols = cols + int(win.col_off)

            indices_for_geoms[int(index)] = (
                np.column_stack((abs_rows, abs_cols))
                )

    zonal_index_map = {
        'width': width,
        'height': height,
        'crs': crs.to_string(),
        'indices': indices_for_geoms
        }

    if len(small_basin_ids):
        print(f"Warning: we used 'all_touched=True' for "
              f"{len(small_basin_ids)} small basins.")

    return zonal_index_map, {'bad': bad_basin_ids, 'small': small_basin_ids}


def extract_zonal_means(
        raster_path: Path, zonal_index_map: dict
        ) -> np.ndarray:
    """
    Extract mean raster values for a list of geometries.

    Computes the spatial mean of raster values (e.g., NDVI, precipitation)
    within each provided geometry (e.g., watershed polygons, administrative
    boundaries). Nodata values are excluded from the mean calculation.
    Geometries that do not intersect the raster or contain only nodata will
    return NaN.

    This implementation keeps the raster file open for the entire loop,
    which is highly efficient for VRT files and large numbers of geometries.

    Parameters
    ----------
    raster_path : Path
        Path to the raster file (GeoTIFF, VRT, etc.).
    geometries : list of shapely.Geometry
        List of geometries (polygons, multipolygons) for which to extract
        raster values.  Must be in the same CRS as the raster.

    Returns
    -------
    np.ndarray
        Array of mean values, one per geometry.
    """
    n_geoms = len(zonal_index_map['indices'])
    mean_values = np.empty(n_geoms, dtype=np.float32)
    basin_ids = np.empty(n_geoms, dtype=np.int64)

    with rasterio.open(raster_path) as src:
        assert src.width == zonal_index_map['width']
        assert src.height == zonal_index_map['height']
        assert src.crs.to_string() == zonal_index_map['crs']

        data = src.read(1)
        nodata = src.nodata

        for i, basin_id in enumerate(zonal_index_map['indices'].keys()):
            basin_ids[i] = basin_id

            # Extract values.
            indices = zonal_index_map['indices'].get(basin_id)
            rows, cols = indices[:, 0], indices[:, 1]
            array = data[rows, cols]

            # Compute mean, excluding nodata
            if nodata is not None:
                valid_pixels = array[array != nodata]
            else:
                valid_pixels = array

            if valid_pixels.size > 0:
                mean_values[i] = np.mean(valid_pixels)
            else:
                # Geometry doesn't intersect raster.
                mean_values[i] = np.nan

    return mean_values, basin_ids


if __name__ == '__main__':
    import rasterio
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from hdml import __datadir__ as datadir

    # Validate 'build_zonal_index_map' function.

    vrt_index_path = datadir / 'ndvi' / 'vrt_index.csv'
    vrt_index = pd.read_csv(vrt_index_path, index_col=0, parse_dates=True)

    wtd_gdf = gpd.read_file(datadir / "data" / "wtd_obs_all.gpkg")
    wtd_gdf = wtd_gdf.set_index("ID", drop=True)

    basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
    basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
    basins_gdf.index = basins_gdf.index.astype(int)

    vrt_fnames = vrt_index.file
    vrt_fnames = vrt_fnames[~pd.isnull(vrt_fnames)]
    vrt_fnames = np.unique(vrt_fnames)

    vrt_path = datadir / 'ndvi' / vrt_fnames[0]

    zonal_index_map, bad_basin_ids = build_zonal_index_map(
        vrt_path, basins_gdf
        )

    # Select a couple of basin ids to test (e.g. [101, 205, 399])
    example_basin_ids = basins_gdf.index[[0, 3, 5]]
    print(example_basin_ids)

    example_basins_gdf = basins_gdf.loc[example_basin_ids]
    example_basins_gdf.to_file(
        datadir / "data" / "example_basin_geometry.gpkg",
        layer='example basin',
        driver="GPKG"
        )

    out_tif_path = vrt_path.with_suffix('.tif')
    with rasterio.open(vrt_path) as src:
        data = src.read(1)
        out_profile = src.profile.copy()
        nodata = src.nodata

        # For each basin, set all extracted pixels to nodata
        for basin_id in example_basin_ids:
            indices = zonal_index_map['indices'].get(basin_id)

            # indices: Nx2 array of [row, col]
            rows, cols = indices[:, 0], indices[:, 1]

            # Apply nodata value
            data[rows, cols] = nodata

        # Write result to a new GeoTIFF
        out_profile.update(driver='GTiff')
        with rasterio.open(out_tif_path, 'w', **out_profile) as dst:
            dst.write(data, 1)
