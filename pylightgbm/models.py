# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import numpy as np
from pylightgbm.utils import os_utils
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class genericGMB(BaseEstimator):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., bagging_fraction=1., bagging_freq=0,
                 metric_freq=1):

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
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'metric_freq': metric_freq
        }

        # create tmp dir to hold data and model (especially the latter)
        self.tmp_dir = "lightgbm_models/{}".format(os_utils._gen_signature())
        os_utils._create_dirs([self.tmp_dir])
        self.output_model = os.path.join(self.tmp_dir, "LightGBM_model_{}.txt".format(os_utils._gen_signature()))
        self.param['output_model'] = self.output_model

    def __del__(self):
        os_utils._remove_dirs([self.tmp_dir])

    def fit(self, X, y=None, test_data=()):

        train_filepath = os.path.abspath("{}/X_{}.svm".format(self.tmp_dir, os_utils._gen_signature()))
        datasets.dump_svmlight_file(X, y, train_filepath)

        valid = []
        if len(test_data) > 0:
            for i, (X, y) in enumerate(test_data):
                test_filepath = os.path.abspath(os.path.join(self.tmp_dir,
                                                             "Xval{}_{}.svm".format(i, os_utils._gen_signature())))
                valid.append(test_filepath)
                datasets.dump_svmlight_file(X, y, test_filepath)

        self.param['task'] = 'train'
        self.param['data'] = train_filepath
        self.param['valid'] = ",".join(valid)

        calls = ["{}={}\n".format(k, self.param[k]) for k in self.param]

        if self.config == "":
            conf_filepath = os.path.join(self.tmp_dir, "train.conf")
            open(conf_filepath, 'w').writelines(calls)
            os.system("{} config={}".format(self.exec_path, conf_filepath))
        else:
            os.system("{} config={}".format(self.exec_path, self.config))

    def predict(self, X):

        predict_filepath = os.path.abspath(
            os.path.join(self.tmp_dir, "X_to_pred_{}.svm".format(os_utils._gen_signature())))
        conf_filepath = os.path.join(self.tmp_dir, "predict_{}.conf".format(os_utils._gen_signature()))
        output_results = os.path.abspath(
            os.path.join(self.tmp_dir, "LightGBM_predict_result_{}.txt".format(os_utils._gen_signature())))

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


class GBMRegressor(genericGMB, RegressorMixin):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., bagging_fraction=1., bagging_freq=0,
                 metric_freq=1):
        super(GBMRegressor, self).__init__(exec_path=exec_path,
                                           config=config,
                                           application=application,
                                           num_iterations=num_iterations,
                                           learning_rate=learning_rate,
                                           num_leaves=num_leaves,
                                           tree_learner=tree_learner,
                                           num_threads=num_threads,
                                           min_data_in_leaf=min_data_in_leaf,
                                           metric=metric,
                                           feature_fraction=feature_fraction,
                                           bagging_fraction=bagging_fraction,
                                           bagging_freq=bagging_freq,
                                           metric_freq=metric_freq),


class GBMClassifier(genericGMB, ClassifierMixin):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="binary",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='binary_logloss',
                 feature_fraction=1., bagging_fraction=1., bagging_freq=0,
                 metric_freq=1):
        super(GBMClassifier, self).__init__(exec_path=exec_path,
                                            config=config,
                                            application=application,
                                            num_iterations=num_iterations,
                                            learning_rate=learning_rate,
                                            num_leaves=num_leaves,
                                            tree_learner=tree_learner,
                                            num_threads=num_threads,
                                            min_data_in_leaf=min_data_in_leaf,
                                            metric=metric,
                                            feature_fraction=feature_fraction,
                                            bagging_fraction=bagging_fraction,
                                            bagging_freq=bagging_freq,
                                            metric_freq=metric_freq),
