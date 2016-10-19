# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
np.random.seed(1337) # for reproducibility


# Loading datasets and convert it to binary classification
X, y = datasets.load_iris(return_X_y=True)
y[y == 2] = 0


# Classifier: 'exec_path' is the absolute path lightgbm
clf = GBMClassifier(exec_path="/home/ardalan/Documents/apps/LightGBM/lightgbm",
                    num_iterations=100, learning_rate=0.01, num_leaves=4,
                    min_data_in_leaf=1)
skf = model_selection.StratifiedKFold(n_splits=5)

scores = []
for train_idx, val_idx in skf.split(X, y):
    x_train = X[train_idx, :]
    y_train = y[train_idx]

    x_val = X[val_idx, :]
    y_val = y[val_idx]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    score = metrics.accuracy_score(y_val, np.round(y_pred))
    scores.append(score)
print("CV score: {}".format(scores))
