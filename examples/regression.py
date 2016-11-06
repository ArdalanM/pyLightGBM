# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMRegressor

# Parameters
seed = 1337
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"

np.random.seed(seed) # for reproducibility
X, y = datasets.load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

# 'exec_path' is the path to lightgbm executable
clf = GBMRegressor(exec_path=path_to_exec,
                   num_iterations=1000, learning_rate=0.01,
                   num_leaves=10, is_training_metric=True,
                   min_data_in_leaf=10, is_unbalance=True,
                   early_stopping_round=10, verbose=True)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
y_pred = clf.predict(x_test)
print("Mean Square Error: ", metrics.mean_squared_error(y_test, y_pred))
print("Best round: ", clf.best_round)
