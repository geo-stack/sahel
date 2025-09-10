# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:39:42 2025
@author: User
"""
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime


def normalize_date(val):
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


def read_obs_wl(filename):
    df = pd.read_excel(
        filename,
        dtype={'DATE': str, 'NS': float, 'LON': float, 'LAT': float,
               'ID': str, 'OMES_ID': str, 'No_Ouvrage': str, 'CODE_OUVRA': str}
        )
    df.columns = ['ID', 'LON', 'LAT', 'DATE', 'NS']

    df['DATE'] = df['DATE'].apply(normalize_date)
    df['DATE'] = pd.to_datetime(df['DATE'])

    df = df.dropna()

    return df


def plot_wl_hist(df):
    fig, ax = plt.subplots(figsize=(6, 4))

    bins = list(range(1950, 2030, 10))

    counts, bins, patches = ax.hist(df['DATE'].dt.year, bins, rwidth=0.8)

    # Annotate each bar
    for count, bin_left, patch in zip(counts, bins, patches):
        # Place the text at the center of the bar
        ax.text(
            patch.get_x() + (patch.get_width()/2),
            count,
            str(int(count)),
            ha='center',
            va='bottom',
        )

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


countries = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo']

for country in countries:
    filename = f"D:/Projets/sahel_water_table_ml/data/data/{country}.xlsx"
    df = read_obs_wl(filename)
    plot_wl_hist(df)
