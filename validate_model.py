# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# the Geological Survey of Canada and is licensed under the terms of
# the MIT license.
#
# https://github.com/geo-stack/cgc_ctscan_kth2
# =============================================================================

# ---- Standard imports
import os
import os.path as osp
import sys

WORKDIR = 'D:/Projets/cgc_ctscan_kth2'
if WORKDIR not in sys.path:
    sys.path.append(WORKDIR)
os.chdir(WORKDIR)


# ---- Third party imports
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

# ---- Local imports
from func import read_db
from func.tcscan import get_sample_dims

OUTDIR = osp.join(WORKDIR, 'results/predict_1D_tcprofil')
os.makedirs(OUTDIR, exist_ok=True)

DATADBPATH = osp.join(WORKDIR, "TC_HU_ITRAX_db_R70_H10.csv")
datadb = read_db(DATADBPATH)


mask = datadb['Litho'] == 'dolostone'
datadb.loc[mask, 'Litho'] = 'sandstone'

mask = datadb['ITRAX_cluster'] == 0
datadb.loc[mask, 'ITRAX_cluster'] = 4

varlist = [
    'hu_p05',
    'hu_p25',
    'hu_p50',
    'hu_p75',
    'hu_p95',
    'hu_var',
    'poro_mean',
    'ITRAX_cluster',
    ]

# %%

from func.modeling import (
    perform_cross_validation, plot_pred_vs_obs, CLS_COLORS, CLS_ZORDER)


classes_col = ['Litho', 'ITRAX_cluster'][0]

scale_data = False
regressor = 'xgboost'

model_kwargs = {
    'colsample_bytree': 0.6,
    'gamma': 1,
    'learning_rate': 0.04,
    'max_depth': 7,
    'n_estimators': 950,
    'reg_alpha': 0.75,
    'reg_lambda': 2,
    'subsample': 0.6
    }

predicdf_loso, importances_avg_loso = perform_cross_validation(
    datadb, varlist=varlist, how='LOSO',
    regressor=regressor, model_kwargs=model_kwargs,
    scale_data=scale_data)


fig = plot_pred_vs_obs(
    predicdf_loso.loc[:, 'TC (W/m/K)'].values,
    predicdf_loso.loc[:, 'KTH_EVAL'].values,
    classes=predicdf_loso.loc[:, classes_col].values,
    suptitle='Leave One Sample Out',
    importances=importances_avg_loso,
    colors=CLS_COLORS,
    zorders=CLS_ZORDER,
    )

rmse_sample = np.mean((
    predicdf_loso.loc[:, 'TC (W/m/K)'].values -
    predicdf_loso.loc[:, 'KTH_EVAL'].values
    )**2)**0.5

print(f'RMSE = {rmse_sample:0.3f}')

predicdf_lopo, importances_avg_lopo = perform_cross_validation(
    datadb, varlist=varlist, how='LOPO',
    regressor=regressor, model_kwargs=model_kwargs,
    scale_data=scale_data)


fig = plot_pred_vs_obs(
    predicdf_lopo.loc[:, 'TC (W/m/K)'].values,
    predicdf_lopo.loc[:, 'KTH_EVAL'].values,
    classes=predicdf_lopo.loc[:, classes_col].values,
    suptitle='Leave One Profil Out',
    importances=importances_avg_lopo,
    colors=CLS_COLORS,
    zorders=CLS_ZORDER,
    )

rmse_profil = np.mean((
    predicdf_lopo.loc[:, 'TC (W/m/K)'].values -
    predicdf_lopo.loc[:, 'KTH_EVAL'].values
    )**2)**0.5

print(f'RMSE = {rmse_profil:0.3f}')

# %%

# Compare TC measured vs predicted.

plt.close('all')
plt.ion()

predicdf = predicdf_loso.copy()

figures = []
for sample_name in predicdf['Sample'].unique():
    sample_db = predicdf.loc[predicdf['Sample'] == sample_name, :]

    if sample_name != '46A':
        continue

    sample_length = (
        get_sample_dims().loc[sample_name.replace('B', 'A')]['length (mm)'])

    profils = sample_db['Profil'].values.copy()

    tc_obs = sample_db['TC (W/m/K)'].values
    tc_pred = sample_db['KTH_EVAL'].values

    fig, ax = plt.subplots(figsize=(8.5, 11 / 3))
    ax.axvline(sample_length * 1, color='0.6', ls=':', lw=0.5)
    ax.axvline(sample_length * 2, color='0.6', ls=':', lw=0.5)
    ax.axvline(sample_length * 3, color='0.6', ls=':', lw=0.5)

    for i, profil in enumerate(('P1', 'P2', 'P3', 'P4')):
        mask = profils == profil
        z_mm = sample_db['Z (mm)'].values[mask] + sample_length * i
        ax.plot(z_mm, tc_obs[mask], color='blue')
        ax.plot(z_mm, tc_pred[mask], color='red', ls='-')

    ax.plot([], [], color='blue', label='Measured')
    ax.plot([], [], color='red', ls='-', label='Predicted')

    xticks = [sample_length * i + sample_length/2 for i in range(4)]
    xtick_labels = ['P1', 'P2', 'P3', 'P4']

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    litho = sample_db['Litho'].unique()[0]
    ax.set_xlabel('Z (mm)', fontsize=14, labelpad=10)
    ax.set_ylabel('Kth (W/m/K)', fontsize=14, labelpad=10)
    ax.set_title(
        f'Sample {sample_name} '
        f'({litho}, length = {sample_length:0.1f} mm)')

    ax.axis(ymin=0, ymax=5.5)

    # Plot features.
    ax2 = ax.twinx()
    for varname in varlist:
        if varname == 'ITRAX_cluster':
            continue

        y = []
        x = []
        for i, profil in enumerate(('P1', 'P2', 'P3', 'P4')):
            mask = profils == profil
            z_mm = sample_db['Z (mm)'].values[mask] + sample_length * i

            x += list(z_mm)
            x += [np.nan]

            values = sample_db[varname].values[mask]
            values = values - np.mean(values)
            values = values / np.max(np.abs(values))

            y += list(values)
            y += [np.nan]

        ax2.plot(x, y, label=varname)
    ax2.axis(ymin=-5, ymax=5)
    ax2.legend(frameon=False, ncol=len(varlist)/2)

    # ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    figures.append(fig)


figname = 'TC_obs_vs_pred.pdf'
with PdfPages(osp.join(OUTDIR, figname)) as pdf:
    for fig in figures:
        pdf.savefig(fig)
