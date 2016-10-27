# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import pickle
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier

# Parameters
path_to_exec = "~/Documents/apps/LightGBM/lightgbm"

X, Y = datasets.make_classification(n_samples=1000, n_features=100, random_state=1337)

# 'exec_path' is the path to lightgbm executable
clf = GBMClassifier(exec_path=path_to_exec, verbose=False)

clf.fit(X, Y)

y_pred = clf.predict(X)

print("Accuracy: ", metrics.accuracy_score(Y, y_pred))

# The sklearn API models are picklable
print("Pickling sklearn API models")
pickle.dump(clf, open("clf_gbm.pkl", "wb"))
clf2 = pickle.load(open("clf_gbm.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))