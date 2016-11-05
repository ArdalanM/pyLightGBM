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
n_classes = 2
test_size = 0.2
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"
np.random.seed(seed)  # for reproducibility

X, Y = datasets.make_classification(n_samples=1000, n_features=100, class_sep=0.6,
                                    n_classes=n_classes, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

params = {'exec_path': path_to_exec, 'num_iterations': 1000, 'learning_rate': 0.01, 'early_stopping_round': 20,
          'min_data_in_leaf': 1, 'num_leaves': 5, 'verbose': False}

clf_binary = GBMClassifier(application='binary', metric='binary_error', **params)
clf_multiclass = GBMClassifier(application='multiclass', num_class=n_classes, metric='multi_error', **params)

for clf in [clf_binary, clf_multiclass]:

    clf.fit(x_train, y_train, test_data=[(x_test, y_test)])

    y_prob = clf.predict_proba(x_test)
    y_pred = y_prob.argmax(-1)

    print("Log loss: ", metrics.log_loss(y_test, y_prob))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Best round: ", clf.best_round)
