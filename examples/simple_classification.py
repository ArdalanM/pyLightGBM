# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
np.random.seed(1337)  # for reproducibility

X, y = datasets.load_iris(return_X_y=True)
y[y == 2] = 0
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

clf = GBMClassifier(exec_path="~/Documents/apps/LightGBM/lightgbm", min_data_in_leaf=1)
clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
