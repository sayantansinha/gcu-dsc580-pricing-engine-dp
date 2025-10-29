import os
import time
import uuid
import pandas as pd
import requests
from typing import Optional, Tuple
from src.config import SETTINGS
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger()

def _new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_uploaded_file(df: pd.DataFrame, base_name: str) -> Tuple[str, str]:
    run_id = _new_run_id("file")
    raw_path = os.path.join(SETTINGS.RAW_DIR, f"{base_name}_{run_id}.parquet")
    df.to_parquet(raw_path, index=False)
    LOGGER.info(f"Saved uploaded dataset to {raw_path}")
    return run_id, raw_path

def load_local_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_from_url(url: str, base_name: Optional[str] = None) -> Tuple[pd.DataFrame, str, str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()
    if base_name is None:
        base_name = "remote"

    if "parquet" in content_type or url.lower().endswith((".parquet", ".pq")):
        tmp = os.path.join(SETTINGS.RAW_DIR, f"{base_name}_tmp.parquet")
        with open(tmp, "wb") as f:
            f.write(r.content)
        df = pd.read_parquet(tmp)
        os.remove(tmp)
    else:
        # assume CSV as default
        df = pd.read_csv(pd.compat.StringIO(r.text))

    run_id, raw_path = save_uploaded_file(df, base_name)
    return df, run_id, raw_path
