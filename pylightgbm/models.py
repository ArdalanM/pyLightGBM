# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import sys
import os
import numpy as np
import tempfile
import shutil
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class genericGMB(BaseEstimator):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2, bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, ):

        self.exec_path = exec_path
        self.config = config
        self.param = {
            'application': application,
            'num_iterations': num_iterations,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'tree_learner': tree_learner,
            'num_threads': num_threads,
            'min_data_in_leaf': min_data_in_leaf,
            'metric': metric,
            'feature_fraction': feature_fraction,
            'feature_fraction_seed': feature_fraction_seed,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'bagging_seed': bagging_seed,
            'metric_freq': metric_freq,
            'early_stopping_round': early_stopping_round
        }

        # create tmp dir to hold data and model (especially the latter)
        self.tmp_dir = tempfile.mkdtemp()
        self.output_model = os.path.join(self.tmp_dir, "LightGBM_model.txt")
        print("CREATE: {}".format(self.tmp_dir), file=sys.stderr)
        self.param['output_model'] = self.output_model

    def fit(self, X, y):
        train_filepath = os.path.abspath("{}/X.svm".format(self.tmp_dir))
        datasets.dump_svmlight_file(X, y, train_filepath)


        self.param['task'] = 'train'
        self.param['data'] = train_filepath

        calls = ["{}={}\n".format(k, self.param[k]) for k in self.param]

        if self.config == "":
            conf_filepath = os.path.join(self.tmp_dir, "train.conf")
            open(conf_filepath, 'w').writelines(calls)
            os.system("{} config={}".format(self.exec_path, conf_filepath))
        else:
            os.system("{} config={}".format(self.exec_path, self.config))

    def predict(self, X):
        predict_filepath = os.path.abspath(os.path.join(self.tmp_dir, "X_to_pred.svm"))
        conf_filepath = os.path.join(self.tmp_dir, "predict.conf")
        output_results = os.path.abspath(os.path.join(self.tmp_dir, "LightGBM_predict_result.txt"))
        datasets.dump_svmlight_file(X, np.zeros(len(X)), f=predict_filepath)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(self.output_model),
            "output_result={}\n".format(output_results)
        ]

        open(conf_filepath, 'w').writelines(calls)
        os.system("{} config={}".format(self.exec_path, conf_filepath))
        y_pred = np.loadtxt(output_results, dtype=float)
        return y_pred

    def get_params(self, deep):
        params = dict(self.param)
        params['exec_path'] = self.exec_path
        params['config'] = self.config
        del params['output_model']
        return params

    def set_params(self, *args, **kwargs):
        print(args)
        print(kwargs)

    def __del__(self):
        if self.tmp_dir:
            print("REMOVE: {}".format(self.tmp_dir), file=sys.stderr)
            shutil.rmtree(self.tmp_dir)
            self.tmp_dir=None

class GBMRegressor(genericGMB, RegressorMixin):
     def __init__(self, *args, **kwargs):
         kwargs['application'] = 'regression'
         super(GBMRegressor, self).__init__(*args, **kwargs)


class GBMClassifier(genericGMB, ClassifierMixin):
     def __init__(self, *args, **kwargs):
         kwargs['application'] = 'binary'
         super(GBMClassifier, self).__init__(*args, **kwargs)
