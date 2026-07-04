import numpy as np
import pandas as pd

def train_val_split(x: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2,):

    n = len(x)
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_val = x.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    return x_train, x_val, y_train, y_val

def get_numeric_columns(x: pd.DataFrame) -> list[str]:
    return x.select_dtypes(include=[np.number]).columns.tolist()

def fit_numeric_imputer(x_train_num: pd.DataFrame) -> pd.Series:
    medians = x_train_num.median()
    return medians

def transform_numeric_imputer(x_num: pd.DataFrame,medians: pd.Series) -> pd.DataFrame:
    return x_num.fillna(medians)

def fit_standardizer(x_train_num: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = x_train_num.mean()
    std = x_train_num.std(ddof=0)
    std = std.replace(0, 1.0)
    return mean, std

def transform_standardizer(x_num: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (x_num - mean) / std

def add_bias_column(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0],1))
    return np.hstack((ones, x))

def prepare_numeric_features(x_train: pd.DataFrame,x_val: pd.DataFrame) -> tuple[
    np.ndarray, np.ndarray, list[str], pd.Series, pd.Series, pd.Series]:

    numeric_cols = get_numeric_columns(x_train)

    x_train_num = x_train[numeric_cols].copy()

    x_val_num = x_val[numeric_cols].copy()

    medians = fit_numeric_imputer(x_train_num)

    x_train_num = transform_numeric_imputer(x_train_num, medians)

    x_val_num = transform_numeric_imputer(x_val_num, medians)

    mean, std = fit_standardizer(x_train_num)

    x_train_num = transform_standardizer(x_train_num, mean, std)

    x_val_num = transform_standardizer(x_val_num, mean, std)

    x_train_np = x_train_num.to_numpy(dtype=np.float64)

    x_val_np = x_val_num.to_numpy(dtype=np.float64)

    x_train_np = add_bias_column(x_train_np)

    x_val_np = add_bias_column(x_val_np)

    return x_train_np, x_val_np, numeric_cols, medians, mean, std