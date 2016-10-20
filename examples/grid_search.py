# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""


from sklearn import model_selection, datasets
from pylightgbm.models import GBMRegressor



X, y = datasets.load_diabetes(return_X_y=True)

# Classifier: 'exec_path' is the absolute path lightgbm
clf = GBMRegressor(exec_path="/home/ardalan/Documents/apps/LightGBM/lightgbm",
                    num_iterations=100, learning_rate=0.01, num_leaves=4,
                    min_data_in_leaf=1)

model_selection.cross_val_score(clf, X, y)