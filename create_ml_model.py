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
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import (
    RandomizedSearchCV, LeaveOneGroupOut, GridSearchCV)

import xgboost as xgb

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.modeling import perform_cross_validation, plot_pred_vs_obs


gwl_df = pd.read_csv(datadir / "wtd_obs_training_dataset.csv")

df = gwl_df.copy()
df = gwl_df.dropna()
df = df[df.country == 'Mali']

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
    'precipitation',
    # 'smoothed_dem',
    ]




# %%

best_params = {
    'subsample': 0.6,
    'reg_lambda': 1.5,
    'reg_alpha': 2,
    'n_estimators': 100,
    'max_depth': 9,
    'learning_rate': 0.01,
    'gamma': 0.5,
    'colsample_bytree': 0.7
    }

df = gwl_df.copy()
df = gwl_df.dropna()
df = df[df.country == 'Mali']

# %%
import matplotlib.pyplot as plt
plt.close('all')

df.dist_top
df.dist_stream

pixel_size = 30


ratio = df.dist_stream / (np.maximum(df.dist_top, pixel_size))

ratio = ratio.astype('float32')

fig, ax = plt.subplots()
ax.plot(ratio, df.ratio_dist, '.')
ax.plot([0, 500], [0, 500], '-', color='red')

ax.set_xlabel('calc')
ax.set_ylabel('from raster')

ax.set_aspect('equal')

# %%

fig, ax = plt.subplots()
ax.plot(df.dist_top/1000, df.dist_stream/1000, 'o')

ax.set_xlabel('dist to top [km]')
ax.set_ylabel('dist to stream [km]')



# %%
import matplotlib.pyplot as plt

train_index = df[df.LON < -6.2].index
test_index = df[df.LON >= -6.2].index

df_train = df.loc[train_index]
df_test = df.loc[test_index]

plt.plot(df_train.LON, df_train.LAT,'o', color='orange')
plt.plot(df_test.LON, df_test.LAT,'o', color='blue')


# %%

plt.close('all')

X_train = df.loc[train_index, varlist].values
y_train = df.loc[train_index, 'NS'].values

X_test = df.loc[test_index, varlist].values
y_test = df.loc[test_index, 'NS'].values

from sklearn.ensemble import RandomForestRegressor

params = {
    'n_estimators': 50,
    'min_samples_split': 4,
    'min_samples_leaf': 4,
    'max_features': 1.0,
    'max_depth': 12,
    "bootstrap": False
    }

Cl = RandomForestRegressor(random_state=42, **params)
Cl.fit(X_train, y_train)
y_eval = Cl.predict(X_test)

importances = pd.DataFrame(columns=['importance'], index=varlist)
for f in range(len(varlist)):
    importances.loc[
        varlist[f], 'importance'
        ] = Cl.feature_importances_[f]
importances = importances.sort_values(by='importance', ascending=False)

classes = np.full(len(y_test), 'Mali')
axis = {'xmin': 0, 'xmax': 30, 'ymin': 0, 'ymax': 30}
fig = plot_pred_vs_obs(
    y_test, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )

y_eval = Cl.predict(X_train)
classes = np.full(len(y_eval), 'Mali')
fig2 = plot_pred_vs_obs(
    y_train, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )

print(importances)

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

# labels = ['ID', 'NS', 'country', 'geometry']
# labels += varlist

# %%

# Hyperparameter optimization (RandomizedSearchCV).

Cl = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200, 300],         # Number of trees in the forest
    "max_depth": [3, 5, 8, 12, 20],              # Maximum depth of each tree
    "min_samples_split": [2, 4, 8, 12],          # Min samples required to split an internal node
    "min_samples_leaf": [1, 2, 4, 8],            # Min samples needed at a leaf node
    "max_features": ["sqrt", "log2", 0.5, 0.8],  # Number of features to consider at each split
    "max_samples": [0.5, 0.7, 0.9],              #  Fraction of samples for each tree
    }

random_search = RandomizedSearchCV(
    estimator=Cl,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
    )
random_search.fit(X_train, y_train)


best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)


# %%

# X = df[varlist].values
# y = df['NS'].values
# random_search.fit(X, y)

# best_params = random_search.best_params_

# print()
# print("Best hyperparamter found :")
# print(best_params)

# # %%

# # Hyperparameter optimization (GridSearchCV).

# params_grid = {
#     'n_estimators': [75, 100, 125],
#     'max_depth': [8, 9, 10],
#     'learning_rate': [0.05, 0.01, 0.02],
#     'subsample': [0.5, 0.6, 0.7],
#     'colsample_bytree': [0.6, 0.7, 0.8],
#     'gamma': [0.25, 0.5, 0.75],
#     'reg_alpha': [1.75, 2, 2.25],
#     'reg_lambda': [1.25, 1.5, 1.75]
#     }

# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=params_grid,
#     scoring='neg_mean_squared_error',
#     # cv=LeaveOneGroupOut(),
#     verbose=2,
#     n_jobs=-1
#     )

# grid_search.fit(X, y)

# best_params = random_search.best_params_

# print()
# print("Best hyperparamter found :")
# print(best_params)
