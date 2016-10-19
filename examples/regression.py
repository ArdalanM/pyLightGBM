# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMRegressor
np.random.seed(1337) # for reproducibility


X, y = datasets.load_diabetes(return_X_y=True)

# Regressor: 'exec_path' is the absolute path lightgbm
clf = GBMRegressor(exec_path="/home/ardalan/Documents/apps/LightGBM/lightgbm",
                   num_iterations=50, learning_rate=0.01, num_leaves=10, min_data_in_leaf=10)
skf = model_selection.KFold(n_splits=5)


scores = []
for train_idx, val_idx in skf.split(X, y):
    x_train = X[train_idx, :]
    y_train = y[train_idx]

    x_val = X[val_idx, :]
    y_val = y[val_idx]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    score = metrics.mean_absolute_error(y_val, y_pred)
    scores.append(score)
print("CV score: {}".format(scores))
