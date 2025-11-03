from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

def load_feature_master(path: str, target_col: str, drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    drop_cols = drop_cols or []
    assert target_col in df.columns, f"Target {target_col} not found"
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    return X, y

def train_valid_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
