# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import numpy as np
import tempfile
import shutil
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class GenericGMB(BaseEstimator):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial",
                 num_threads=1, min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2,
                 bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, max_bin=255, model=None):

        self.exec_path = exec_path
        self.config = config
        self.model = model
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
            'early_stopping_round': early_stopping_round,
            'max_bin': max_bin
        }

        # create tmp dir to hold data and model (especially the latter)

    def fit(self, X, y, test_data=None):

        tmp_dir = tempfile.mkdtemp()
        train_filepath = os.path.abspath("{}/X.svm".format(tmp_dir))
        datasets.dump_svmlight_file(X, y, train_filepath)

        valid = []
        if test_data:
            for i, (x_test, y_test) in enumerate(test_data):
                test_filepath = os.path.abspath("{}/X{}_test.svm".format(tmp_dir, i))
                print(test_filepath)
                valid.append(test_filepath)
                datasets.dump_svmlight_file(x_test, y_test, test_filepath)

        self.param['task'] = 'train'
        self.param['data'] = train_filepath
        self.param['valid'] = ",".join(valid)
        self.param['output_model'] = os.path.join(tmp_dir, "LightGBM_model.txt")

        calls = ["{}={}\n".format(k, self.param[k]) for k in self.param]
        if self.config == "":
            conf_filepath = os.path.join(tmp_dir, "train.conf")
            open(conf_filepath, 'w').writelines(calls)
            os.system("{} config={}".format(self.exec_path, conf_filepath))
        else:
            os.system("{} config={}".format(self.exec_path, self.config))

        with open(self.param['output_model'], mode='rb') as file:
            self.model = file.read()

        shutil.rmtree(tmp_dir)

    def predict(self, X):
        tmp_dir = tempfile.mkdtemp()
        predict_filepath = os.path.abspath(os.path.join(tmp_dir, "X_to_pred.svm"))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))
        with open(output_model, mode="wb") as file:
            file.write(self.model)

        datasets.dump_svmlight_file(X, np.zeros(len(X)), f=predict_filepath)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(output_model),
            "output_result={}\n".format(output_results)
        ]

        open(conf_filepath, 'w').writelines(calls)
        os.system("{} config={}".format(self.exec_path, conf_filepath))
        y_pred = np.loadtxt(output_results, dtype=float)

        shutil.rmtree(tmp_dir)

        return y_pred

    def get_params(self, deep=True):
        params = dict(self.param)
        params['exec_path'] = self.exec_path
        params['config'] = self.config
        params['model'] = self.model
        if 'output_model' in params:
            del params['output_model']
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self

    def __del__(self):
        pass


class GBMClassifier(GenericGMB, ClassifierMixin):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="binary",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2, bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, max_bin=255, model=None):
        super(GBMClassifier, self).__init__(exec_path, config, application, num_iterations, learning_rate, num_leaves,
                                            tree_learner, num_threads, min_data_in_leaf, metric, feature_fraction,
                                            feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                                            metric_freq, early_stopping_round, max_bin, model)

    def predict_proba(self, X):
        tmp_dir = tempfile.mkdtemp()
        predict_filepath = os.path.abspath(os.path.join(tmp_dir, "X_to_pred.svm"))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))
        with open(output_model, mode="wb") as file:
            file.write(self.model)

        datasets.dump_svmlight_file(X, np.zeros(len(X)), f=predict_filepath)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(output_model),
            "output_result={}\n".format(output_results)
        ]

        open(conf_filepath, 'w').writelines(calls)
        os.system("{} config={}".format(self.exec_path, conf_filepath))

        probability_of_one = np.loadtxt(output_results, dtype=float)
        probability_of_zero = 1 - probability_of_one

        shutil.rmtree(tmp_dir)

        return np.transpose(np.vstack((probability_of_zero, probability_of_one)))

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return y_prob.argmax(-1)


class GBMRegressor(GenericGMB, RegressorMixin):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2, bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, max_bin=255, model=None):
        super(GBMRegressor, self).__init__(exec_path, config, application, num_iterations, learning_rate, num_leaves,
                                           tree_learner, num_threads, min_data_in_leaf, metric, feature_fraction,
                                           feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                                           metric_freq, early_stopping_round, max_bin, model)
