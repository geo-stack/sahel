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

# ---- Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from sklearn.preprocessing import StandardScaler, RobustScaler

# ---- Local imports

SCALER = None


def calc_avg_importance(importances):
    impavg = None
    for i in range(len(importances)):
        if i == 0:
            impavg = importances[i].copy()
        else:
            for key, value in importances[i].items():
                impavg[key] += value
    for key, value in impavg.items():
        impavg[key] = round(value / (i + 1) * 100, 1)
    return impavg


def perform_cross_validation(
        datadb: pd.DataFrame, varlist: list,
        regressor: str = 'xgboost', model_kwargs: dict = None,
        scale_data: bool = False, verbose=False
        ):
    """
    Evaluate model with Leave-One-Sample-Out cross-validation method, where
    sample is simply the contry where the measurement was made.
    """
    importances = []
    importances_avg = []

    model_kwargs = {} if model_kwargs is None else model_kwargs
    if regressor == 'xgboost':
        from xgboost import XGBRegressor as Regressor
    elif regressor == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor as Regressor

    labels = ['ID', 'NS', 'country', 'geometry']
    labels += varlist

    predicdf = datadb[labels].copy()
    predicdf = predicdf.dropna()

    X = predicdf.loc[:, varlist].values
    y = predicdf.loc[:, 'NS']

    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    countries = np.unique(predicdf['country'])
    for i, country in enumerate(countries):
        if verbose:
            print(f'Prediction country #{country} '
                  f'({i + 1} de {len(countries)})...')

        # Define a mask to partion the training and test dataset.
        train_mask = (predicdf['country'] != country)
        test_mask = (predicdf['country'] == country)

        Xtrain = X[train_mask, :]
        ytrain = y[train_mask]

        Xtest = X[test_mask, :]
        ytest = y[test_mask]

        Cl = Regressor(**model_kwargs)
        Cl.fit(Xtrain, ytrain)
        yeval = Cl.predict(Xtest)
        predicdf.loc[ytest.index, 'NS'] = yeval

        importances.append({
            varlist[f]: Cl.feature_importances_[f] for
            f in range(X.shape[1])
            })

    importances_avg = calc_avg_importance(importances)

    return predicdf, importances_avg


MARKERSIZE = 2


def plot_pred_vs_obs(
        xobs, xpred, classes, axis: dict,
        suptitle: str = None,
        axtitle: str = None, varlist: list = None,
        importances: list = None, colors: dict = None,
        plot_stats: bool = True, highlight_label: str = None,
        highlight_mask: list = None, zorders: dict = None):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    if suptitle is not None:
        fig.suptitle(suptitle)
    if axtitle is not None:
        ax.set_title(axtitle, fontsize=9, ha='center')
    elif varlist is not None:
        axtitle = ''
        for i in range(len(varlist)):
            if i == 0:
                axtitle += varlist[i]
            elif i % 5 == 0:
                axtitle += '\n' + varlist[i]
            else:
                axtitle += ', ' + varlist[i]
        ax.set_title(axtitle, fontsize=9, ha='center')

    ax.plot([axis['xmin'], axis['xmax']], [axis['ymin'], axis['ymax']],
            ls='--', color='0.5', lw=1)
    ax.grid(color='0.85', lw=0.5)

    colors = {} if colors is None else colors
    zorders = {} if zorders is None else zorders
    for i, cl in enumerate(np.unique(classes)):
        mask = classes == cl

        kwargs = {'marker': 'o', 'ls': 'None', 'ms': MARKERSIZE,
                  'label': cl, 'zorder': zorders.get(cl, 100)}
        color = colors.get(cl, None)
        if color is not None:
            kwargs['color'] = color

        ax.plot(xobs[mask], xpred[mask], **kwargs)

    if highlight_label:
        ax.plot(xobs[highlight_mask], xpred[highlight_mask],
                marker='o', ls='None', ms=MARKERSIZE,
                color='red', label=highlight_label, zorder=120)

    lg = ax.legend(loc='upper left', bbox_to_anchor=[-0.03, 1],
                   ncol=1, frameon=False, handletextpad=0.5)

    hshift = 5/72
    vshift = -(lg.get_window_extent().height + 20) / fig.dpi

    if plot_stats:
        transform = ax.transAxes + ScaledTranslation(
            hshift, vshift, fig.dpi_scale_trans)
        ax.text(0, 1, 'ME [W/m/K] :',
                transform=transform, va='top', ha='left', fontweight='bold',
                fontsize=9, zorder=1000)

        vshift -= 14/72

        for i, cl in enumerate(np.unique(classes)):
            mask = classes == cl
            me = np.mean(xobs[mask] - xpred[mask])
            transform = ax.transAxes + ScaledTranslation(
                hshift, vshift, fig.dpi_scale_trans)
            ax.text(0, 1, f'{str(cl)[:4]} = {me:0.1e}',
                    va='top', ha='left', fontsize=9, transform=transform,
                    zorder=1000)

            vshift -= 12/72

        vshift -= 6/72

        transform = ax.transAxes + ScaledTranslation(
            hshift, vshift, fig.dpi_scale_trans)
        ax.text(0, 1, 'RMSE [W/m/K] :',
                transform=transform, va='top', ha='left', fontweight='bold',
                fontsize=9, zorder=1000)

        vshift -= 14/72

        for i, cl in enumerate(np.unique(classes)):
            mask = classes == cl
            rmse = (np.mean((xobs[mask] - xpred[mask])**2))**0.5
            transform = ax.transAxes + ScaledTranslation(
                hshift, vshift, fig.dpi_scale_trans)
            ax.text(0, 1, f'{str(cl)[:4]} = {rmse:0.3f}',
                    va='top', ha='left', fontsize=9, transform=transform,
                    zorder=1000)

            vshift -= 12/72

    imp_max_width = 0
    text_handles = []
    if importances is not None:
        hshift = -3/72
        vshift = 5/72

        grpsize = 2
        varlist = list(importances.keys())
        vargroups = [varlist[i:i+grpsize] for i in
                     range(0, len(varlist), grpsize)]
        lines = []
        for vargroup in vargroups:
            line = ''
            for varname in vargroup:
                line += f'{varname} ({importances[varname]:0.1f}%); '
            lines.append(line)

        for line in reversed(lines):

            transform = ax.transAxes + ScaledTranslation(
                hshift, vshift, fig.dpi_scale_trans)
            text = ax.text(
                0.5, 0, line,
                va='bottom', ha='left', fontsize=8,
                transform=transform, zorder=1000)
            text_handles.append(text)
            imp_max_width = max(text.get_window_extent().width, imp_max_width)

            vshift += 12/72

        vshift += 2/72

        transform = ax.transAxes + ScaledTranslation(
            hshift, vshift, fig.dpi_scale_trans)
        text = ax.text(0.5, 0, 'Features :',
                       transform=transform, va='bottom', ha='left',
                       fontweight='bold', fontsize=9, zorder=1000)
        text_handles.append(text)
        imp_max_width = max(text.get_window_extent().width, imp_max_width)

        for text in text_handles:
            text.set_position((1 - imp_max_width / ax.bbox.width, 0))

    ax.set_ylabel('WTD predicted [m bgl]', fontsize=12, labelpad=10)
    ax.set_xlabel('WTD observed [m bgl]', fontsize=12, labelpad=10)

    fig.tight_layout()

    ax.axis(**axis)

    return fig
