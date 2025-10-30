"""
io_utils: thin wrappers for parquet IO + basic dataframe checks.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Iterable



def load_tsv(path: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path / name, sep="\t", na_values="\\N", low_memory=False)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise SystemExit(f"{name} missing required columns: {missing}")
