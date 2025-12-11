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
import geopandas as gpd
from sklearn.model_selection import (
    RandomizedSearchCV, LeaveOneGroupOut, GridSearchCV)

import xgboost as xgb

# ---- Local imports
from hdml import __datadir__ as datadir


gwl_gdf = gpd.read_file(datadir / "wtd_obs_training_dataset.csv")


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
    # 'ndvi',
    # 'precipitation'
    ]


xgb_model = xgb.XGBRegressor(random_state=42)
logo = LeaveOneGroupOut()

labels = ['ID', 'NS', 'country', 'geometry']
labels += varlist

predicdf = gwl_gdf[labels].copy()
predicdf = gwl_gdf.dropna()

X = predicdf.loc[:, varlist].values
y = predicdf.loc[:, 'NS']
groups = predicdf['country'].values

# %%

# Hyperparameter optimization (RandomizedSearchCV).

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
    cv=logo,
    verbose=2,
    random_state=42,
    n_jobs=-1
    )

random_search.fit(X, y, groups=groups)

print()
print("Meilleurs hyperparamètres trouvés :")
print(random_search.best_params_)


# %%

params_grid = {
    'n_estimators': [850, 900, 950],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.04, 0.05, 0.06],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'gamma': [1],
    'reg_alpha': [0.75, 1, 1.25],
    'reg_lambda': [1.75, 2, 2.25]
    }


grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=params_grid,
    scoring='neg_mean_squared_error',
    cv=logo,
    verbose=2,
    n_jobs=-1
    )

grid_search.fit(X, y, groups=groups)

print()
print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

# Best params:
# model_kwargs = {
#     'colsample_bytree': 0.6,
#     'gamma': 1,
#     'learning_rate': 0.04,
#     'max_depth': 7,
#     'n_estimators': 950,
#     'reg_alpha': 0.75,
#     'reg_lambda': 2,
#     'subsample': 0.6
#     }
