pyLightGBM
==========
Python binding for Microsoft LightGBM: https://github.com/Microsoft/LightGBM

**pyLightGBM** have several advantages:

 - Early stopping (```clf.best_round```)
 - Works with scikit-learn: ```GridSearchCV```, ```cross_val_score```, etc...
 - Silent mode (```verbose=False```)

Installation
------------

Install lastest verion of [Microsoft LightGBM](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide) then install the wrapper:
```
 pip install git+https://github.com/ArdalanM/pyLightGBM.git
 ```

Examples
------------

* Regression:

```python
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMRegressor

# full path to lightgbm executable (on Windows include .exe)
exec = "~/Documents/apps/LightGBM/lightgbm"

X, y = datasets.load_diabetes(return_X_y=True)
clf = GBMRegressor(exec_path=exec,
                   num_iterations=100, early_stopping_round=10,
                   num_leaves=10, min_data_in_leaf=10)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
print("Mean Square Error: ", metrics.mean_squared_error(y_test, clf.predict(x_test)))
```

* Binary Classification:

```python
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier

# full path to lightgbm executable (on Windows include .exe)
exec = "~/Documents/apps/LightGBM/lightgbm"

X, Y = datasets.make_classification(n_samples=200, n_features=10)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = GBMClassifier(exec_path=exec, min_data_in_leaf=1)
clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
```

* Grid Search:

```python
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier

# full path to lightgbm executable (on Windows include .exe)
exec = "~/Documents/apps/LightGBM/lightgbm"

X, Y = datasets.make_classification(n_samples=1000, n_features=10)

gbm = GBMClassifier(exec_path=exec,
                    metric='binary_error', early_stopping_round=10, bagging_freq=10)

param_grid = {'learning_rate': [0.1, 0.04], 'bagging_fraction': [0.5, 0.9]}

scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2)

clf.fit(X, Y)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)
```

Notebooks
------------
* [**Using pyLightGBM for Kaggle competition (Allstate Claims Severity)**](https://github.com/ArdalanM/pyLightGBM/blob/master/notebooks/regression_example_kaggle_allstate.ipynb)
