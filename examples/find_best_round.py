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

X, Y = datasets.make_classification(n_samples=10000, n_features=100, random_state=seed)

skf = model_selection.StratifiedKFold(n_splits=nfolds, random_state=seed)

clf = GBMClassifier(exec_path=path_to_exec, num_iterations=10000, min_data_in_leaf=10,
                    metric='binary_error', early_stopping_round=10, verbose=False)

best_rounds = []
scores = []
for i, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
    x_train = X[train_idx, :]
    y_train = Y[train_idx]

    x_valid = X[valid_idx, :]
    y_valid = Y[valid_idx]

    clf.fit(x_train, y_train, test_data=[(x_valid, y_valid)])
    best_round = clf.best_round
    best_rounds.append(best_round)

    y_pred = clf.predict(x_valid)

    score = metrics.accuracy_score(y_valid, y_pred)
    scores.append(score)

    print("Fold: [{}/{}]: Accuracy: {:.3f}, best round: {}".format(i+1, skf.n_splits, score, best_round))
print("Average: accuracy: {:.3f}, best round: {}".format(np.mean(scores), int(np.mean(best_rounds))))