# pyLightGBM
Python wrapper for Microsoft LightGBM
https://github.com/Microsoft/LightGBM

## Regression example:
```python
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMRegressor

X, y = datasets.load_diabetes(return_X_y=True)
clf = GBMRegressor(exec_path="~/Documents/apps/LightGBM/lightgbm",
                   num_iterations=100, learning_rate=0.01, num_leaves=10, min_data_in_leaf=10)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
print("Mean Square Error: ", metrics.mean_squared_error(y_test, clf.predict(x_test)))
```

## Binary Classification example:
```python
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
np.random.seed(1337)  # for reproducibility

X, y = datasets.load_iris(return_X_y=True)
y[y == 2] = 0 # to have binary labels
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

clf = GBMClassifier(exec_path="~/Documents/apps/LightGBM/lightgbm", min_data_in_leaf=1)
clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
```


