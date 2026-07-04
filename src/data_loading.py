import pandas as pd
from pathlib import Path

def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"File {csv_path} not found")
    return pd.read_csv(csv_path)

def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in dataframe")
    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return x, y