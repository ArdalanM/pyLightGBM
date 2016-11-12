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
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"
np.random.seed(seed)  # for reproducibility

X, Y = datasets.make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed)

params = {'exec_path': path_to_exec,
          'num_iterations': 1000, 'learning_rate': 0.01,
          'min_data_in_leaf': 1, 'num_leaves': 5,
          'metric': 'binary_error', 'verbose': False,
          'early_stopping_round': 20}

clfs = [
    ['gbdt', GBMClassifier(boosting_type='gbdt', **params)],
    ['dart', GBMClassifier(boosting_type='dart', drop_rate=0.02, drop_seed=4, **params)],
]

for boosting_type, clf in clfs:

    clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
    y_prob = clf.predict_proba(x_test)
    y_pred = y_prob.argmax(-1)

    print("booster {} loss: {}, accuracy: {}, best round: {}".format(
        boosting_type,
        metrics.log_loss(y_test, y_prob),
        metrics.accuracy_score(y_test, y_pred),
        clf.best_round
    ))
