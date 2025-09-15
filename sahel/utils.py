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
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime

#commentaire de JP super pertinent

#2e comment

# Commentaire de JS


def read_obs_wl(filename) -> pd.DataFrame:
    """
    Read water level observations from an Excel file and preprocess
    the data.

    Parameters
    ----------
    filename : str
        Path to the Excel file containing water level observations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ID', 'LON', 'LAT', 'DATE', 'NS'] and
        cleaned, parsed data.
    """
    df = pd.read_excel(
        filename,
        dtype={'DATE': str, 'NS': float, 'LON': float, 'LAT': float,
               'ID': str, 'OMES_ID': str, 'No_Ouvrage': str, 'CODE_OUVRA': str}
        )
    df.columns = ['ID', 'LON', 'LAT', 'DATE', 'NS']

    df['DATE'] = df['DATE'].apply(_normalize_date)
    df['DATE'] = pd.to_datetime(df['DATE'])

    df = df.dropna()

    return df


def _normalize_date(val) -> datetime:
    """Normalize various date string formats to a datetime object."""
    if val == '00:00:00' or val.strip() == '':
        return pd.NaT

    # Pattern: Year only, e.g. '1976'.
    m = re.fullmatch(r'(\d{4})', val)
    if m:
        return datetime.strptime(
            f"{m.group(1)}-01-01", "%Y-%m-%d"
            )

    # Pattern: YYYY-MM-DD HH:MM:SS or with milliseconds.
    m = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}:\d{2,3}', val)
    if m:
        # We keep only the date part.
        return datetime.strptime(
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d"
            )

    # Pattern: DD/MM/YYYY.
    m = re.fullmatch(r'(\d{2})/(\d{2})/(\d{4})', val)
    if m:
        # Convert to ISO format.
        return datetime.strptime(
            f"{m.group(3)}-{m.group(2)}-{m.group(1)}", "%Y-%m-%d"
            )

    raise ValueError(f'Cannot parse date with value={val}')


def plot_wl_hist(df: pd.DataFrame, country: str):
    """
    Plot a histogram of water level observation counts by decade.

    This function generates a histogram of the number of water level (WL)
    observations per decade based on the 'DATE' column of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'DATE' column of datetime type.
    country : str
        Country name to display in the title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated histogram figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    bins = list(range(1950, 2030, 10))

    counts, bins, patches = ax.hist(df['DATE'].dt.year, bins, rwidth=0.8)

    # Annotate each bar.
    for count, bin_left, patch in zip(counts, bins, patches):
        ax.text(
            patch.get_x() + (patch.get_width()/2),
            count,
            str(int(count)),
            ha='center',
            va='bottom',
        )

    # Add a little bit of space at the top of the graph for the text.
    ylim = ax.get_ylim()
    height_pixels = ax.get_window_extent().height
    data_per_pixel = (ylim[1] - ylim[0]) / height_pixels
    ypad = 10 * data_per_pixel
    ax.set_ylim(ylim[0], ylim[1] + ypad)

    ax.yaxis.grid(which='major', color='0.85')
    ax.set_axisbelow(True)
    ax.set_xlabel('Decade', labelpad=15, fontsize=12)
    ax.set_ylabel('Frequency', labelpad=15, fontsize=12)
    fig.suptitle(f'Number of WL observations for {country}')

    ax.set_xticks(bins)

    fig.tight_layout()

    return fig


if __name__ == '__main__':
    from sahel import __datadir__
    import os.path as osp
    countries = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo']
    for country in countries:
        filename = osp.join(__datadir__, 'data', f'{country}.xlsx')
        df = read_obs_wl(filename)
        fig = plot_wl_hist(df, country)

        filepath = osp.join(__datadir__, 'data', f'wl_obs_count_{country}.png')
        fig.savefig(filepath, dpi=220)
