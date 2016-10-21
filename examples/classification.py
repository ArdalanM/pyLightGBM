# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier


# params
seed = 1337
np.random.seed(seed)  # for reproducibility

X, Y = datasets.make_classification(n_samples=1000, n_features=100, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed)

# 'exec_path' is the path to lightgbm executable
clf = GBMClassifier(exec_path="~/Documents/apps/LightGBM/lightgbm",
                    num_iterations=100, learning_rate=0.1,
                    min_data_in_leaf=1,
                    metric='binary_error',
                    early_stopping_round=10)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])

y_prob = clf.predict_proba(x_test)
y_pred = y_prob.argmax(-1)

print("Log loss: ", metrics.log_loss(y_test, y_prob))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
