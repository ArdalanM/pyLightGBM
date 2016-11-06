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
n_classes = 10
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"
np.random.seed(seed)  # for reproducibility

X, Y = datasets.load_digits(return_X_y=True, n_class=n_classes)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed)

# 'exec_path' is the path to lightgbm executable
clf = GBMClassifier(exec_path=path_to_exec, num_class=n_classes, metric='multi_logloss',
                    application='multiclass', num_iterations=1000,
                    min_data_in_leaf=1, num_leaves=5, early_stopping_round=20)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])

y_prob = clf.predict_proba(x_test)
y_pred = y_prob.argmax(-1)

print("Log loss: ", metrics.log_loss(y_test, y_prob))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Best round: ", clf.best_round)
