# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier


# Parameters
seed = 1337
nfolds = 5
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"

np.random.seed(seed)  # for reproducibility

X, Y = datasets.make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=seed)

# 'exec_path' is the path to lightgbm executable
gbm = GBMClassifier(exec_path=path_to_exec,
                    num_iterations=1000, learning_rate=0.05,
                    min_data_in_leaf=1, num_leaves=5,
                    metric='binary_logloss', verbose=True)

param_grid = {
    'learning_rate': [0.1, 0.04],
    'min_data_in_leaf': [1, 10],
    'bagging_fraction': [0.5, 0.9]
}

scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2)

clf.fit(X, Y)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)
