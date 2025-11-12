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

# ---- Standard imports.
from pathlib import Path

# ---- Third party imports.
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

# ---- Local imports.


def distance_to_stream(output: Path, dem: Path, streams: Path):
    # Read DEM data and metadata.
    with rasterio.open(dem) as src:
        dem_data = src.read(1)
        meta = src.meta.copy()
        pixel_width = src.transform[0]
        pixel_height = abs(src.transform[4])
        nodata = src.nodata

    # Read streams (0 = no stream, >0 = stream).
    with rasterio.open(streams) as src:
        streams_data = src.read(1)

    # Create binary stream mask (1 = stream, 0 = non-stream)
    stream_mask = streams_data > 0

    # Calculate Euclidean distance to nearest stream (in pixels)
    distance_pixels = distance_transform_edt(~stream_mask)

    # Convert to meters (assuming square pixels, use average if not)
    pixel_size = (pixel_width + pixel_height) / 2
    distance_meters = distance_pixels * pixel_size

    # Set nodata value where dem is null.
    distance_meters[dem_data == nodata] = nodata

    # Save distance to stream
    meta.update(dtype='float32')
    with rasterio.open(output, 'w', **meta) as dst:
        dst.write(distance_meters.astype('float32'), 1)


def height_above_stream(output: Path, dem: Path, streams: Path):
    # Read DEM data and metadata.
    with rasterio.open(dem) as src:
        dem_data = src.read(1).astype('float32')
        meta = src.meta.copy()
        nodata = src.nodata

    # Replace nodata by nan.
    dem_data[dem_data == nodata] = np.nan

    # Read streams (0 = no stream, >0 = stream).
    with rasterio.open(streams) as src:
        streams_data = src.read(1)

    # Create binary stream mask (1 = stream, 0 = non-stream).
    stream_mask = streams_data > 0

    # Get elevation of nearest stream for each pixel
    indices = distance_transform_edt(
        ~stream_mask,
        return_distances=False,
        return_indices=True)

    rows = indices[0]
    cols = indices[1]

    nearest_stream_elevation = dem_data[rows, cols]

    # Calculate elevation above nearest stream
    elevation_above_stream = dem_data - nearest_stream_elevation

    # Fill nan value with nodata.
    elevation_above_stream[np.isnan(elevation_above_stream)] = nodata

    # Save elevation above stream
    meta.update(dtype='float32')
    with rasterio.open(output, 'w', **meta) as dst:
        dst.write(elevation_above_stream.astype('float32'), 1)
