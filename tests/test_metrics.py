import numpy as np

from src.evaluate import mse
from src.evaluate import rmse
from src.evaluate import r2_score

def test_rmse_zero():

    y_true = np.array([1,2,3])

    y_pred = np.array([1,2,3])

    assert rmse(y_true,y_pred) == 0

def test_r2_perfect():

    y_true = np.array([1,2,3])

    y_pred = np.array([1,2,3])

    assert r2_score(y_true,y_pred) == 1

def test_mse_zero():

    y_true = np.array([1,2,3])

    y_pred = np.array([1,2,3])

    assert mse(y_true,y_pred) == 0