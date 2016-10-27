# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from xgboost import XGBRegressor
from pylightgbm.models import GBMRegressor

# Parameters
seed = 1337
nfolds = 5
test_size = 0.2

# 'path_to_exec' is the path to lightgbm executable (lightgbm.exe on Windows)
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"
# for reproducibility
np.random.seed(seed)

datasets = datasets.load_boston(return_X_y=False)
X = datasets['data']
Y = datasets['target']
feature_names = datasets['feature_names']

clf_xgb = XGBRegressor(max_depth=3, n_estimators=1000)
clf_gbm = GBMRegressor(exec_path=path_to_exec, num_iterations=1000, learning_rate=0.01,
                       num_leaves=255, min_data_in_leaf=1, early_stopping_round=20, verbose=False)


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Training the two models
clf_gbm.fit(x_train, y_train, test_data=[(x_test, y_test)])
clf_xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='rmse', early_stopping_rounds=20, verbose=False)

print("xgboost: feature importance")
dic_fi = clf_xgb.booster().get_fscore()
xgb_fi = [(feature_names[int(k[1:])], dic_fi[k]) for k in dic_fi]
xgb_fi = sorted(xgb_fi, key=lambda x: x[1], reverse=True)
print(xgb_fi)

print("lightgbm: feature importance")
gbm_fi = clf_gbm.feature_importance(feature_names)
print(gbm_fi)
