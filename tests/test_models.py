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
dataset_params = {'n_samples': 100, 'n_features': 10, 'random_state': seed}

X, Y = datasets.make_classification(n_classes=2, **dataset_params)
Xmulti, Ymulti = datasets.make_multilabel_classification(n_classes=n_classes, **dataset_params)
Xreg, Yreg = datasets.make_regression(**dataset_params)


class TestGBM(object):
    def test_simple_fit(self):

        params = dict(exec_path=path_to_exec, num_iterations=100, min_data_in_leaf=1,
                      learning_rate=0.1, num_leaves=5)
        clfs = [
            [Xreg, Yreg, 'regression', GBMRegressor(boosting_type='gbdt', **params)],
            [Xreg, Yreg, 'regression', GBMRegressor(boosting_type='dart', **params)],
            [X, Y, 'classification', GBMClassifier(boosting_type='gbdt', **params)],
            [X, Y, 'classification', GBMClassifier(boosting_type='dart', **params)],
        ]

        for x, y, name, clf in clfs:
            clf.fit(x, y, init_scores=np.zeros(x.shape[0]))

            if name == 'regression':
                score = metrics.mean_squared_error(y, clf.predict(x))
                score < 1.
            else:
                score = metrics.accuracy_score(Y, clf.predict(X))
                assert score > 0.9

    def test_early_stopping(self):

        cv_params = dict(test_size=test_size, random_state=seed)
        xtr, xte, ytr, yte = model_selection.train_test_split(X, Y, **cv_params)
        xtr_reg, xte_reg, ytr_reg, yte_reg = model_selection.train_test_split(X, Y, **cv_params)

        params = dict(exec_path=path_to_exec, num_iterations=10000, min_data_in_leaf=3,
                      learning_rate=0.01, num_leaves=2, early_stopping_round=2)
        clfs = [
            [xtr_reg, ytr_reg, xte_reg, yte_reg, 'regression', GBMRegressor(boosting_type='gbdt', **params)],
            [xtr_reg, ytr_reg, xte_reg, yte_reg, 'regression', GBMRegressor(boosting_type='dart', **params)],
            [xtr, ytr, xte, yte, 'classification', GBMClassifier(boosting_type='gbdt', **params)],
            [xtr, ytr, xte, yte, 'classification', GBMClassifier(boosting_type='dart', **params)],
        ]

        for xtr, ytr, xte, yte, name, clf in clfs:
            clf.fit(xtr, ytr, test_data=[(xte, yte)])

            if name == 'regression':
                score = metrics.mean_squared_error(yte, clf.predict(xte))
                assert (score < 1. and clf.best_round < clf.param['num_iterations'])
            else:
                score = metrics.accuracy_score(yte, clf.predict(xte))
                assert (score > 0.7 and clf.best_round < clf.param['num_iterations'])

    def test_grid_search(self):

        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'num_leaves': [2, 5, 50],
            'min_data_in_leaf': [1, 10, 100],
            'bagging_fraction': [0.1, 1]
        }

        params = {
            'exec_path': path_to_exec, 'num_threads': 2,
            'num_iterations': 100, 'learning_rate': 0.1,
            'min_data_in_leaf': 1, 'num_leaves': 10,
            'bagging_freq': 2, 'verbose': False
        }

        clfs = [
            [Xreg, Yreg, 'regression', GBMRegressor(boosting_type='gbdt', metric='l2', **params)],
            [Xreg, Yreg, 'regression', GBMRegressor(boosting_type='dart', metric='l2', **params)],
            [X, Y, 'classification', GBMClassifier(boosting_type='gbdt', metric='binary_logloss', **params)],
            [X, Y, 'classification', GBMClassifier(boosting_type='dart', metric='binary_logloss', **params)],
        ]

        for x, y, name, clf in clfs:

            if name == 'regression':
                scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
                grid = model_selection.GridSearchCV(clf, param_grid, scoring=scorer, cv=2, refit=True)
                grid.fit(x, y)

                score = metrics.mean_squared_error(y, grid.predict(x))
                print(score)
                assert score < 2000
            else:
                scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
                grid = model_selection.GridSearchCV(clf, param_grid, scoring=scorer, cv=2, refit=True)
                grid.fit(x, y)

                score = metrics.accuracy_score(y, grid.predict(x))
                print(score)
                assert score > .9

    def test_sparse(self):

        params = {'exec_path': path_to_exec, 'num_iterations': 1000, 'verbose': False,
                  'min_data_in_leaf': 1, 'learning_rate': 0.1, 'num_leaves': 5}

        clfs = [
            [sps.csr_matrix(X), Y, 'classification', GBMClassifier(**params)],
            [sps.csr_matrix(Xreg), Yreg, 'regression', GBMRegressor(**params)],
        ]

        for x, y, name, clf in clfs:
            clf.fit(x, y)

            if name == 'classification':
                score = metrics.accuracy_score(y, clf.predict(x))
                assert score > 0.9
            else:
                score = metrics.mean_squared_error(y, clf.predict(x))
                assert score < 1.

    def test_pickle(self):

        params = {'exec_path': path_to_exec, 'verbose': False}

        clfs = [
            [X, Y, GBMClassifier(**params)],
            [Xreg, Yreg, GBMRegressor(**params)],
        ]

        for x, y, clf in clfs:
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

if __name__ == '__main__':
    pytest.main([__file__])
