import _io
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import requests
from typing import Optional, Tuple
from src.config import SETTINGS
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("io_utils")


def _new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_uploaded_file(df: pd.DataFrame, base_name: str) -> Tuple[str, str]:
    run_id = _new_run_id("file")
    raw_path = os.path.join(SETTINGS.RAW_DIR, f"{base_name}_{run_id}.parquet")
    df.to_parquet(raw_path, index=False)
    LOGGER.info(f"Saved uploaded dataset to {raw_path}")
    return run_id, raw_path


def load_local_file(path: str) -> pd.DataFrame:
    LOGGER.info(f"Loading local file {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def read_from_url(url: str, base_name: Optional[str] = None) -> Tuple[pd.DataFrame, str, str]:
    lower = url.lower()
    LOGGER.info(f"Reading data from URL: {url}")
    if base_name is None:
        base_name = Path(url).stem.replace(".tsv", "").replace(".gz", "") or "remote"

    if lower.endswith((".parquet", ".pq")):
        df = pd.read_parquet(url)
    elif lower.endswith((".tsv.gz", ".tsv")):
        df = pd.read_csv(url, sep="\t", compression="infer", na_values="\\N", low_memory=False)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(_io.StringIO(r.text))

    run_id, raw_path = save_uploaded_file(df, base_name)
    LOGGER.info(f"File read and saved from URL: {url}")
    return df, run_id, raw_path
