# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pylightgbm.models import GBMRegressor
from sklearn import datasets, metrics, model_selection

# Parameters
seed = 1337
nfolds = 5
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"

np.random.seed(seed)  # for reproducibility

X, Y = datasets.load_diabetes(return_X_y=True)

# 'exec_path' is the path to lightgbm executable
gbm = GBMRegressor(exec_path=path_to_exec,
                   num_iterations=100,
                   learning_rate=0.1,
                   min_data_in_leaf=1,
                   bagging_freq=10,
                   metric='binary_error',
                   early_stopping_round=10,
                   verbose=False)

param_grid = {
    'learning_rate': [0.1, 0.04],
    'min_data_in_leaf': [1, 10],
    'bagging_fraction': [0.5, 0.9]
}

scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2)

clf.fit(X, Y)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)
