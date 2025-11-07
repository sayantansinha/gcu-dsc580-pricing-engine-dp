import _io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from src.config.env_loader import SETTINGS
from src.utils.log_utils import get_logger

LOGGER = get_logger("data_io")


# -----------------------------
# Session cache
# -----------------------------
# def set_final_df(df: pd.DataFrame):
#     """Cache the active DataFrame in Streamlit session."""
#     st.session_state["final_df"] = df
#     LOGGER.info("final_df stored in Streamlit session")
#
#
# def get_final_df() -> pd.DataFrame | None:
#     """Retrieve cached DataFrame."""
#     return st.session_state.get("final_df")

# -----------------------------
# Raw data
# -----------------------------
def save_raw(df: pd.DataFrame, base_dir: str, name: str) -> str:
    """Save processed dataset under PROCESSED_DIR."""
    path = os.path.join(Path(SETTINGS.RAW_DIR) / base_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    LOGGER.info(f"Saved processed data file → {path}")
    return path


def load_raw(path: str) -> pd.DataFrame:
    """Load processed dataset from PROCESSED_DIR."""
    try:
        LOGGER.info(f"Loading processed data file ← {path}")
        return pd.read_parquet(path)
    except Exception as ex:
        error_text = f"Error loading file from {path}"
        LOGGER.exception(error_text)
        raise ValueError(error_text) from ex


def save_from_url(url: str, base_dir: str) -> str:
    try:
        lower = url.lower()
        LOGGER.info(f"Reading data from URL: {url}")
        base_name = Path(url).stem.replace(".tsv", "").replace(".gz", "") or "remote"

        if lower.endswith((".parquet", ".pq")):
            df = pd.read_parquet(url)
        elif lower.endswith((".tsv.gz", ".tsv")):
            df = pd.read_csv(url, sep="\t", compression="infer", na_values="\\N", low_memory=False)
        else:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(_io.StringIO(r.text))

        raw_path = save_raw(df, base_dir, base_name)
        LOGGER.info(f"File read and saved from URL: {url}")
        return raw_path
    except Exception as ex:
        error_text = f"Error reading file from {url}"
        LOGGER.exception(error_text)
        raise ValueError(error_text) from ex


def list_raw_files(base_dir: str) -> list[str]:
    try:
        raw = Path(SETTINGS.RAW_DIR) / base_dir
        if not os.path.isdir(raw):
            return []
        files = [f for f in os.listdir(raw) if f.endswith(".parquet")]
        files.sort()
        LOGGER.info(f"Listing raw files in {raw} directory => {files}")
        return files
    except Exception as ex:
        error_text = f"Error listing raw files from {raw} directory"
        LOGGER.exception(error_text)
        raise ValueError(error_text) from ex


# -----------------------------
# Processed data
# -----------------------------
def save_processed(df: pd.DataFrame, base_dir: str, name: str) -> str:
    """Save processed dataset under PROCESSED_DIR."""
    path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    LOGGER.info(f"Saved processed data file → {path}")
    return path


def load_processed(name: str, base_dir: str) -> pd.DataFrame:
    """Load processed dataset from PROCESSED_DIR."""
    try:
        name = name if name.endswith(".parquet") else f"{name}.parquet"
        path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, f"{name}")
        LOGGER.info(f"Loading processed data file ← {path}")
        return pd.read_parquet(path)
    except Exception as ex:
        LOGGER.error("Error loading processed data file", exc_info=ex)
        raise ValueError("Error loading processed data file") from ex


# -----------------------------
# Figures
# -----------------------------
def save_figure(fig, base_dir: str, name: str) -> str:
    """Save matplotlib/plotly figure to FIGURES_DIR."""
    path = os.path.join(Path(SETTINGS.FIGURES_DIR) / base_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    LOGGER.info(f"Saved figure → {path}")
    return path


# -----------------------------
# Reports (HTML)
# -----------------------------
def save_reports(html: str, base_dir: str, report_name: str) -> str:
    """Save HTML report to /data/public (derived from DATA_DIR)."""
    path = os.path.join(Path(SETTINGS.DATA_DIR) / base_dir, f"{report_name}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    LOGGER.info(f"Saved HTML report → {path}")
    return path


# -----------------------------
# Profiles / validation summaries
# -----------------------------
def save_profile(profile_obj: dict, base_dir: str, name: str) -> str:
    """Save data validation or profiling result to PROFILES_DIR."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(Path(SETTINGS.PROFILES_DIR) / base_dir, f"{name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_obj, f, indent=2)
    LOGGER.info(f"Saved profile summary → {path}")
    return path
