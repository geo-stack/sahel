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

"""
Functions for generating and managing raster tiles with overlap.
"""

# ---- Standard imports.
from pathlib import Path
import math
import itertools

# ---- Third party imports.
from osgeo import gdal


# ---- Local imports.
from sahel import __datadir__ as datadir

gdal.UseExceptions()


def generate_tiles_bbox(
        input_raster: Path,
        tile_size: int = 5000,
        overlap: int = 0,
        zone_bbox: tuple = None
        ) -> dict:
    """
    Generate bounding box information for tiling a zone within a raster with
    overlap.

    Pre-computes the pixel coordinates for a grid of tiles covering either the
    full input raster or a specific rectangular zone within it, including both
    the core (non-overlapping) tile extents and the overlapped extents needed
    for accurate edge processing.

    Parameters
    ----------
    input_raster : Path
        Path to the input raster file to be tiled.
    tile_size : int, optional
        Size of each tile in pixels (width and height of the core tile area,
        excluding overlap). Default is 5000.
    overlap : float, optional
        Overlap distance between adjacent tiles in the same units as the
        raster's CRS (e.g., meters for ESRI:102022). This overlap ensures
        that edge effects are minimized when processing tiles independently.
        Default is 0.
    zone_bbox : tuple, optional
        Rectangular zone to tile in the raster's coordinate system as
        (minx, miny, maxx, maxy). Coordinates should match the raster's CRS
        (e.g., ESRI:102022 coordinates in meters). If None, tiles the entire
        raster.

    Returns
    -------
    dict
        Dictionary mapping tile indices (ty, tx) to tile geometry information.

        Each entry contains:
        - 'core': [x_start, y_start, width, height] - Core tile extent without
          overlap, in pixel coordinates relative to the input raster. Used for
          the final mosaic.
        - 'overlap': [x_start, y_start, width, height] - Extended tile extent
          with overlap, in pixel coordinates. Used for processing to ensure
          accurate calculations at tile edges.
        - 'crop_x_offset': int - Number of pixels to skip from the left edge
          of the overlapped tile to extract the core tile.
        - 'crop_y_offset': int - Number of pixels to skip from the top edge
          of the overlapped tile to extract the core tile.

        Note: Tiles smaller than 10x10 pixels are excluded from the output.
    """
    import rasterio
    from rasterio.windows import from_bounds

    # Open raster to get dimensions and transform, then close immediately.
    with rasterio.open(input_raster) as src:
        raster_width = src.width
        raster_height = src.height
        transform = src.transform

        # We assume square pixels and uses only x resolution.
        pixel_size = abs(transform.a)

    # Convert overlap from geographic/projected units to pixels.
    overlap_pixels = int(round(overlap / pixel_size))

    print(f"Raster resolution: {pixel_size:.2f} units/pixel")
    print(f"Overlap: {overlap:.2f} units = {overlap_pixels} pixels")

    # Convert geographic zone bounds to pixel coordinates if provided.
    if zone_bbox is not None:
        minx, miny, maxx, maxy = zone_bbox

        # Validate that bounds are in correct order.
        if minx >= maxx or miny >= maxy:
            raise ValueError(
                f"Invalid zone bounds: minx must be < maxx and miny must "
                f"be < maxy. Got minx={minx}, miny={miny}, maxx={maxx}, "
                f"maxy={maxy}."
                )

        # Convert geographic bounds to pixel window.
        window = from_bounds(minx, miny, maxx, maxy, transform=transform)

        # Extract pixel coordinates.
        zone_x_min = int(round(window.col_off))
        zone_y_min = int(round(window.row_off))
        zone_x_max = int(round(window.col_off + window.width))
        zone_y_max = int(round(window.row_off + window.height))

        # Clamp to raster bounds.
        zone_x_min = max(0, zone_x_min)
        zone_y_min = max(0, zone_y_min)
        zone_x_max = min(raster_width, zone_x_max)
        zone_y_max = min(raster_height, zone_y_max)

        # Validate that zone intersects with raster.
        if zone_x_min >= raster_width or zone_y_min >= raster_height:
            raise ValueError(
                f"Zone bbox does not intersect with raster. "
                f"Zone in pixels: ({zone_x_min}, {zone_y_min}, "
                f"{zone_x_max}, {zone_y_max}), "
                f"Raster size: ({raster_width}, {raster_height})"
                )
        if zone_x_max <= 0 or zone_y_max <= 0:
            raise ValueError(
                f"Zone bbox does not intersect with raster. "
                f"Zone in pixels: ({zone_x_min}, {zone_y_min}, "
                f"{zone_x_max}, {zone_y_max})"
            )
        print(f"Tiling zone: geographic bounds "
              f"({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")
        print(f"            pixel bounds "
              f"({zone_x_min}, {zone_y_min}, {zone_x_max}, {zone_y_max})")
    else:
        # Tile the entire raster.
        zone_x_min, zone_y_min = 0, 0
        zone_x_max, zone_y_max = raster_width, raster_height
        print(f"Tiling entire raster: {raster_width}x{raster_height} pixels")

    # Calculate zone dimensions.
    zone_width = zone_x_max - zone_x_min
    zone_height = zone_y_max - zone_y_min

    # Calculate number of tiles needed for the zone.
    n_tiles_x = math.ceil(zone_width / tile_size)
    n_tiles_y = math.ceil(zone_height / tile_size)

    tiles_bbox_data = {}
    for ty, tx in itertools.product(range(n_tiles_y), range(n_tiles_x)):
        # Calculate the CORE tile extent WITHOUT overlap (for final mosaic)
        # relative to the ZONE.
        x_start_in_zone = tx * tile_size
        y_start_in_zone = ty * tile_size
        x_end_in_zone = min(zone_width, (tx + 1) * tile_size)
        y_end_in_zone = min(zone_height, (ty + 1) * tile_size)

        # Convert to absolute raster coordinates.
        x_start = zone_x_min + x_start_in_zone
        y_start = zone_y_min + y_start_in_zone
        x_end = zone_x_min + x_end_in_zone
        y_end = zone_y_min + y_end_in_zone

        w = x_end - x_start
        h = y_end - y_start

        # Calculate tile extent WITH overlap for processing
        # Constrained by both zone bounds AND raster bounds.
        x_start_ovlp = max(0, zone_x_min + tx * tile_size - overlap_pixels)
        y_start_ovlp = max(0, zone_y_min + ty * tile_size - overlap_pixels)
        x_end_ovlp = min(
            raster_width,
            zone_x_min + (tx + 1) * tile_size + overlap_pixels
            )
        y_end_ovlp = min(
            raster_height,
            zone_y_min + (ty + 1) * tile_size + overlap_pixels
            )

        w_ovlp = x_end_ovlp - x_start_ovlp
        h_ovlp = y_end_ovlp - y_start_ovlp

        # Skip tiny edge tiles
        if w_ovlp < 10 or h_ovlp < 10:
            continue

        tiles_bbox_data[(ty, tx)] = {
            'core': [x_start, y_start, w, h],
            'overlap': [x_start_ovlp, y_start_ovlp, w_ovlp, h_ovlp],
            'crop_x_offset': x_start - x_start_ovlp,
            'crop_y_offset': y_start - y_start_ovlp
            }

    return tiles_bbox_data


def extract_tile(
        input_raster: Path,
        output_tile: Path,
        bbox: list,
        overwrite: bool = False
        ) -> Path:
    """
    Extract a tile from a raster using pixel coordinates.

    Parameters
    ----------
    input_raster : Path
        Path to input raster file.
    output_tile : Path
        Path where the extracted tile will be saved.
    bbox : list
        Bounding box as [x_start, y_start, width, height] in pixel coordinates.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.

    Returns
    -------
    Path
        Path to the extracted tile.
    """
    if not output_tile.exists() or overwrite:
        gdal.Translate(
            str(output_tile),
            str(input_raster),
            srcWin=bbox,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
            )

    return output_tile


def crop_tile(
        input_tile: Path,
        output_tile: Path,
        crop_x_offset: int,
        crop_y_offset: int,
        width: int,
        height: int,
        overwrite: bool = False
        ) -> Path:
    """
    Crop overlap margins from a processed tile.

    Parameters
    ----------
    input_tile : Path
        Path to the tile with overlap.
    output_tile : Path
        Path where the cropped tile will be saved.
    crop_x_offset : int
        Number of pixels to skip from left edge.
    crop_y_offset : int
        Number of pixels to skip from top edge.
    width : int
        Width of the cropped tile in pixels.
    height : int
        Height of the cropped tile in pixels.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.

    Returns
    -------
    Path
        Path to the cropped tile.
    """
    if not output_tile.exists() or overwrite:
        gdal.Translate(
            str(output_tile),
            str(input_tile),
            srcWin=[crop_x_offset, crop_y_offset, width, height],
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES']
            )

    return output_tile


def mosaic_tiles(
        tile_paths: list,
        output_raster: Path,
        overwrite: bool = False,
        cleanup_tiles: bool = False
        ) -> Path:
    """
    Mosaic tiles into a single raster using GDAL VRT and translate.

    Parameters
    ----------
    tile_paths : list
        List of Path objects pointing to tiles to mosaic.
    output_raster : Path
        Path where the mosaiced raster will be saved.
    overwrite : bool, optional
        Whether to overwrite existing output. Default is False.
    cleanup_tiles : bool, optional
        Whether to delete tiles after mosaicing. Default is False.

    Returns
    -------
    Path
        Path to the output mosaiced raster.
    """
    if output_raster.exists() and not overwrite:
        return output_raster

    # Build VRT.
    vrt_path = output_raster.with_suffix('.vrt')
    gdal.BuildVRT(str(vrt_path), tile_paths)

    # Translate to GeoTIFF.
    gdal.Translate(
        str(output_raster),
        str(vrt_path),
        creationOptions=[
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=YES',
            'NUM_THREADS=ALL_CPUS'
        ]
    )

    # Cleanup.
    vrt_path.unlink(missing_ok=True)

    if cleanup_tiles:
        for tile in tile_paths:
            tile.unlink(missing_ok=True)

    return output_raster


if __name__ == '__main__':
    from sahel import __datadir__ as datadir
    # wtd_path = Path(datadir) / 'data' / 'wtd_obs_all.geojson'
    # output_path = Path(datadir) / 'data' / 'wtd_obs_boundary.geojson'
    # bbox_gdf = create_buffered_bounding_box(wtd_path, output_path)
