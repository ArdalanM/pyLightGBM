# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import pytest
import pickle
import numpy as np
import scipy.sparse as sps
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier, GBMRegressor

# Params
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"
seed = 1337
test_size = 0.2
n_classes = 3
np.random.seed(seed)

X, Y = datasets.make_classification(n_samples=100, n_features=10, n_classes=2, random_state=seed)
Xmulti, Ymulti = datasets.make_multilabel_classification(n_samples=100, n_features=10,
                                                         n_classes=n_classes, random_state=seed)
Xreg, Yreg = datasets.make_regression(n_samples=100, n_features=10, random_state=seed)


class TestGBMClassifier(object):
    def test_simple_fit(self):
        clf = GBMClassifier(exec_path=path_to_exec, min_data_in_leaf=1, learning_rate=0.001, num_leaves=5)
        clf.fit(X, Y)
        score = metrics.accuracy_score(Y, clf.predict(X))
        assert score > 0.9

    def test_early_stopping(self):
        num_iterations = 10000
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size,
                                                                            random_state=seed)
        clf = GBMClassifier(exec_path=path_to_exec, num_iterations=num_iterations,
                            min_data_in_leaf=3, learning_rate=0.01, num_leaves=2, early_stopping_round=2)
        clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
        score = metrics.accuracy_score(y_test, clf.predict(x_test))
        assert (score > 0.7 and clf.best_round < num_iterations)

    def test_grid_search(self):
        gbm = GBMClassifier(exec_path=path_to_exec, num_iterations=100, learning_rate=0.01,
                            min_data_in_leaf=1, num_leaves=5, bagging_freq=2,
                            metric='binary_logloss', verbose=False)

        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'num_leaves': [2, 5, 50],
            'min_data_in_leaf': [1, 10, 100],
            'bagging_fraction': [0.1, 0.5]
        }
        scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
        clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2, refit=True)
        clf.fit(X, Y)
        score = metrics.accuracy_score(Y, clf.predict(X))
        assert score > 0.75

    def test_sparse(self):

        clf = GBMClassifier(exec_path=path_to_exec, min_data_in_leaf=1, learning_rate=0.001, num_leaves=5)
        clf.fit(sps.csr_matrix(X), Y)
        score = metrics.accuracy_score(Y, clf.predict(sps.csr_matrix(X)))
        assert score > 0.9

    def test_pickle(self):

        clf = GBMClassifier(exec_path=path_to_exec, verbose=False)
        clf.fit(X, Y)
        pickle.dump(clf, open("clf_gbm.pkl", "wb"))
        clf2 = pickle.load(open("clf_gbm.pkl", "rb"))
        assert np.allclose(clf.predict(X), clf2.predict(X))

    def test_multiclass(self):

        clf = GBMClassifier(exec_path=path_to_exec, min_data_in_leaf=1, learning_rate=0.1,
                            num_leaves=5, num_class=n_classes, metric='multi_logloss',
                            application='multiclass', num_iterations=100)
        clf.fit(Xmulti, Ymulti.argmax(-1))
        clf.fit(Xmulti, Ymulti.argmax(-1), test_data=[(Xmulti, Ymulti.argmax(-1))])
        score = metrics.accuracy_score(Ymulti.argmax(-1), clf.predict(Xmulti))
        assert score > 0.8


class TestGBMRegressor(object):
    def test_simple_fit(self):
        clf = GBMRegressor(exec_path=path_to_exec, num_iterations=1000,
                           min_data_in_leaf=5, learning_rate=0.1, num_leaves=10, verbose=False)
        clf.fit(Xreg, Yreg)
        score = metrics.mean_squared_error(Yreg, clf.predict(Xreg))
        assert score < 1.

    def test_early_stopping(self):
        num_iterations = 10000
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size,
                                                                            random_state=seed)
        clf = GBMRegressor(exec_path=path_to_exec, num_iterations=num_iterations,
                           min_data_in_leaf=3, learning_rate=0.01, num_leaves=5, early_stopping_round=2)
        clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
        score = metrics.mean_squared_error(y_test, clf.predict(x_test))
        print(score)
        assert (score < 1. and clf.best_round < num_iterations)

    def test_grid_search(self):
        gbm = GBMRegressor(exec_path=path_to_exec, num_iterations=200, learning_rate=0.01,
                           min_data_in_leaf=2, num_leaves=5, bagging_freq=2,
                           metric='l2', verbose=False)

        param_grid = {
            'learning_rate': [0.01, 0.1, 1, 2],
            'num_leaves': [2, 5, 50],
            'min_data_in_leaf': [1, 10, 100],
            'bagging_fraction': [0.1, 0.5]
        }
        scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
        clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2, refit=True)
        clf.fit(Xreg, Yreg)
        score = metrics.mean_squared_error(Yreg, clf.predict(Xreg))
        print(score)
        assert score < 2000

    def test_sparse(self):
        clf = GBMRegressor(exec_path=path_to_exec, num_iterations=1000,
                           min_data_in_leaf=5, learning_rate=0.1, num_leaves=10, verbose=False)
        clf.fit(sps.csr_matrix(Xreg), Yreg)
        score = metrics.mean_squared_error(Yreg, clf.predict(sps.csr_matrix(Xreg)))
        assert score < 1.

    def test_pickle(self):
        clf = GBMRegressor(exec_path=path_to_exec, verbose=False)
        clf.fit(Xreg, Yreg)
        pickle.dump(clf, open("clf_gbm.pkl", "wb"))
        clf2 = pickle.load(open("clf_gbm.pkl", "rb"))
        assert np.allclose(clf.predict(Xreg), clf2.predict(Xreg))


if __name__ == '__main__':
    pytest.main([__file__])
