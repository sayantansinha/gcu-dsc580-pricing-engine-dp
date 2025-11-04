"""
io_utils: thin wrappers for parquet IO + basic dataframe checks.
"""
from pathlib import Path

import pandas as pd
import yaml


def load_tsv(path: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path / name, sep="\t", na_values="\\N", low_memory=False)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
