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
import os
import os.path as osp
import sys
from pathlib import Path

# ---- Third party imports
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import (
    RandomizedSearchCV, LeaveOneGroupOut, GridSearchCV)

import xgboost as xgb

# ---- Local imports
from hdml import __datadir__ as datadir


gwl_df = pd.read_csv(datadir / "wtd_obs_training_dataset.csv")


varlist = [
    'ratio_dist',
    'ratio_stream',
    'dist_stream',
    'alt_stream',
    'dist_top',
    'alt_top',
    'long_hessian_max',
    'long_hessian_mean',
    'long_hessian_var',
    'long_hessian_skew',
    'long_hessian_kurt',
    'long_grad_mean',
    'long_grad_var',
    'short_grad_max',
    'short_grad_var',
    'short_grad_mean',
    'stream_grad_max',
    'stream_grad_var',
    'stream_grad_mean',
    'stream_hessian_max',
    'ndvi',
    'precipitation'
    ]

labels = ['ID', 'NS', 'country', 'geometry']
labels += varlist

# %%

# Perform model cross-validation using the countries to split the dataset.
import numpy as np
from hdml.modeling import perform_cross_validation, plot_pred_vs_obs


df = gwl_df.copy()
df = gwl_df.dropna()
df = df[df.country == 'Mali']


# %%

# Hyperparameter optimization (RandomizedSearchCV).

xgb_model = xgb.XGBRegressor(random_state=42)

params_grid = {
    'n_estimators': [100, 300, 500, 700, 900],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [0, 0.1, 1, 1.5, 2],
    'reg_lambda': [1, 1.5, 2, 2.5, 3]
    }

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params_grid,
    n_iter=100,
    scoring='neg_mean_squared_error',
    # cv=LeaveOneGroupOut(),
    verbose=2,
    random_state=42,
    n_jobs=-1
    )

X = df[varlist].values
y = df['NS'].values
random_search.fit(X, y)

best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)

# %%

# Hyperparameter optimization (GridSearchCV).

params_grid = {
    'n_estimators': [75, 100, 125],
    'max_depth': [8, 9, 10],
    'learning_rate': [0.05, 0.01, 0.02],
    'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0.25, 0.5, 0.75],
    'reg_alpha': [1.75, 2, 2.25],
    'reg_lambda': [1.25, 1.5, 1.75]
    }


grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=params_grid,
    scoring='neg_mean_squared_error',
    # cv=LeaveOneGroupOut(),
    verbose=2,
    n_jobs=-1
    )

grid_search.fit(X, y)

best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)


# %%

train_index = df[df.LON < -6.2].index
test_index = df[df.LON >= -6.2].index

X_train = df.loc[train_index, varlist].values
y_train = df.loc[train_index, 'NS'].values

X_test = df.loc[test_index, varlist].values
y_test = df.loc[test_index, 'NS'].values

Cl = xgb.XGBRegressor(random_state=42, **best_params)
Cl.fit(X_train, y_train)
y_eval = Cl.predict(X_test)

importances = {
    varlist[f]: Cl.feature_importances_[f] for
    f in range(X.shape[1])
    }

classes = ['Mali'] * len(y_test)
axis = {'xmin': 0, 'xmax': 30, 'ymin': 0, 'ymax': 30}
fig = plot_pred_vs_obs(
    y_test, y_train, classes, axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )

# predicdf, importances_avg = perform_cross_validation(
#     gwl_df, varlist=varlist,
#     regressor='xgboost',
#     model_kwargs=model_kwargs,
#     scale_data=False)

# xobs = traindf.NS.values
# xpred = predicdf.NS.values
# classes = traindf.country.values

# max_val = max(np.max(xobs), np.max(xpred))

# axis = {'xmin': 0, 'xmax': 90, 'ymin': 0, 'ymax': 90}

# fig = plot_pred_vs_obs(
#     xobs, xpred, classes, axis,
#     suptitle=None,
#     axtitle=None, varlist=None,
#     importances=None, colors=None,
#     plot_stats=True, highlight_label=None,
#     highlight_mask=None, zorders=None
#     )
