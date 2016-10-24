# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import re
import shutil
import tempfile
import subprocess
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class GenericGMB(BaseEstimator):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="regression",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial",
                 num_threads=1, min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2,
                 bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, max_bin=255,
                 verbose=True, model=None):

        # '~/path/to/lightgbm' becomes 'absolute/path/to/lightgbm'
        self.exec_path = os.path.expanduser(exec_path)

        self.config = config
        self.model = model
        self.verbose = verbose

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
                valid.append(test_filepath)
                datasets.dump_svmlight_file(x_test, y_test, test_filepath)

        self.param['task'] = 'train'
        self.param['data'] = train_filepath
        self.param['valid'] = ",".join(valid)
        self.param['output_model'] = os.path.join(tmp_dir, "LightGBM_model.txt")

        calls = ["{}={}\n".format(k, self.param[k]) for k in self.param]

        if self.config == "":
            conf_filepath = os.path.join(tmp_dir, "train.conf")
            with open(conf_filepath, 'w') as f:
                f.writelines(calls)

            process = subprocess.check_output([self.exec_path, "config={}".format(conf_filepath)],
                                              universal_newlines=True)

        else:
            process = subprocess.check_output([self.exec_path, "config={}".format(self.config)],
                                              universal_newlines=True)

        if self.verbose:
            print(process)

        if test_data and self.param['early_stopping_round'] > 0:
            # Extracting best round from raw logs: 'best iteration round is'
            pattern = re.compile("best iteration round is ((\d+))")
            match = re.search(pattern, process)
            if match:
                self.best_round = int(match.group(1))
            else:
                self.best_round = self.param['num_iterations']

        with open(self.param['output_model'], mode='r') as file:
            self.model = file.read()

        shutil.rmtree(tmp_dir)

    def predict(self, X):
        tmp_dir = tempfile.mkdtemp()
        predict_filepath = os.path.abspath(os.path.join(tmp_dir, "X_to_pred.svm"))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")

        with open(output_model, mode="w") as file:
            file.write(self.model)

        datasets.dump_svmlight_file(X, np.zeros(len(X)), f=predict_filepath)

        calls = ["task = predict\n",
                 "data = {}\n".format(predict_filepath),
                 "input_model = {}\n".format(output_model),
                 "output_result={}\n".format(output_results)]

        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        process = subprocess.check_output([self.exec_path, "config={}".format(conf_filepath)],
                                          universal_newlines=True)

        if self.verbose:
            print(process)

        y_pred = np.loadtxt(output_results, dtype=float)

        shutil.rmtree(tmp_dir)

        return y_pred

    def get_params(self, deep=True):
        params = dict(self.param)
        params['exec_path'] = self.exec_path
        params['config'] = self.config
        params['model'] = self.model
        params['verbose'] = self.verbose
        if 'output_model' in params:
            del params['output_model']
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self

    def feature_importance(self, feature_names=[], importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:
            'weight' - the number of times a feature is used to split the data across all trees.
            'gain' - the average gain of the feature when it is used in trees
            'cover' - the average coverage of the feature when it is used in trees

        Parameters
        ----------
        feature_names: list (optional)
           List of feature names.
        importance_type: string
            The type of feature importance
        """
        assert(importance_type == 'weight', 'For now, only weighted feature importance is implemented')

        pattern_nfeat = re.compile("max_feature_idx=(\d+)")
        pattern_split_feat = re.compile("split_feature=([\d+\s\d+]+)\n")

        match_nfeat = re.match(pattern_nfeat, self.model)
        match_split = re.findall(pattern_split_feat, self.model)

        if match_nfeat:
            # total number of features
            nfeatures = int(match_nfeat.group(1)) + 1
        else:
            raise ValueError

        if importance_type == 'weight':
            if match_split:
                # list of feature index used for splitting a node
                string_of_idx = " ".join(match_split)
                list_of_int = [int(val) for val in string_of_idx.split(" ")]

            # Sorted list of [(feature_index, feature_importance)]
            feat_importance = Counter(list_of_int).most_common()
            feat_importance = [(idx, importance/float(nfeatures))
                               for idx, importance in feat_importance]

        if len(feature_names) > 0:
            feat_importance = [(feature_names[idx], importance)
                               for idx, importance in feat_importance]

        return feat_importance


class GBMClassifier(GenericGMB, ClassifierMixin):
    def __init__(self, exec_path="LighGBM/lightgbm", config="", application="binary",
                 num_iterations=10, learning_rate=0.1,
                 num_leaves=127, tree_learner="serial", num_threads=1,
                 min_data_in_leaf=100, metric='l2',
                 feature_fraction=1., feature_fraction_seed=2, bagging_fraction=1., bagging_freq=0, bagging_seed=3,
                 metric_freq=1, early_stopping_round=0, max_bin=255, verbose=True, model=None):

        super(GBMClassifier, self).__init__(exec_path, config, application, num_iterations, learning_rate, num_leaves,
                                            tree_learner, num_threads, min_data_in_leaf, metric, feature_fraction,
                                            feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                                            metric_freq, early_stopping_round, max_bin, verbose, model)

    def predict_proba(self, X):

        tmp_dir = tempfile.mkdtemp()
        predict_filepath = os.path.abspath(os.path.join(tmp_dir, "X_to_pred.svm"))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))

        with open(output_model, mode="w") as file:
            file.write(self.model)

        datasets.dump_svmlight_file(X, np.zeros(len(X)), f=predict_filepath)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(output_model),
            "output_result={}\n".format(output_results)
        ]

        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        process = subprocess.check_output([self.exec_path, "config={}".format(conf_filepath)],
                                          universal_newlines=True)

        if self.verbose:
            print(process)

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
                 metric_freq=1, early_stopping_round=0, max_bin=255, verbose=True, model=None):

        super(GBMRegressor, self).__init__(exec_path, config, application, num_iterations, learning_rate, num_leaves,
                                           tree_learner, num_threads, min_data_in_leaf, metric, feature_fraction,
                                           feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                                           metric_freq, early_stopping_round, max_bin, verbose, model)
