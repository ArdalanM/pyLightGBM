# -*- coding: utf-8 -*-
"""
@author: Evgeny Bazarov <baz.evgenii@gmail.com>
@brief:
"""
import os
import numpy as np
from sklearn import datasets


def dump_data(X, y, f, issparse=False):
    """
    store data into CSV file and sparse data into SVM
    """
    if issparse:
        datasets.dump_svmlight_file(X, y, f)
    else:
        # From LightGBM docs https://github.com/Microsoft/LightGBM/wiki/Quick-Start
        # Label is the data of first column, and there is no header in the file.
        np.savetxt(f, X=np.column_stack((y, X)), delimiter=',', newline=os.linesep)
