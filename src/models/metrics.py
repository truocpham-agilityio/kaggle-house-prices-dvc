# -*- coding: utf-8 -*-
import numpy as np

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Compute the mean squared error of predicted values"""
    return np.sqrt(mean_squared_error(y_true, y_pred))
