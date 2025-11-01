import json
import os
from datetime import datetime

import pandas as pd

from src.config import SETTINGS
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
# Processed data
# -----------------------------
def save_processed(df: pd.DataFrame, name: str = "final_df") -> str:
    """Save processed dataset under PROCESSED_DIR."""
    path = os.path.join(SETTINGS.PROCESSED_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    LOGGER.info(f"Saved processed data file → {path}")
    return path


def load_processed(name: str = "final_df") -> pd.DataFrame:
    """Load processed dataset from PROCESSED_DIR."""
    path = os.path.join(SETTINGS.PROCESSED_DIR, f"{name}.parquet")
    LOGGER.info(f"Loading processed data file ← {path}")
    return pd.read_parquet(path)


# -----------------------------
# Figures
# -----------------------------
def save_figure(fig, name: str) -> str:
    """Save matplotlib/plotly figure to FIGURES_DIR."""
    path = os.path.join(SETTINGS.FIGURES_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    LOGGER.info(f"Saved figure → {path}")
    return path


# -----------------------------
# Reports (HTML)
# -----------------------------
def save_report_local(html: str, report_name: str) -> str:
    """Save HTML report to /data/public (derived from DATA_DIR)."""
    public_dir = os.path.join(SETTINGS.DATA_DIR, "public")
    os.makedirs(public_dir, exist_ok=True)
    path = os.path.join(public_dir, f"{report_name}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    LOGGER.info(f"Saved HTML report → {path}")
    return path


# -----------------------------
# Profiles / validation summaries
# -----------------------------
def save_profile(profile_obj: dict, name: str) -> str:
    """Save data validation or profiling result to PROFILES_DIR."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(SETTINGS.PROFILES_DIR, f"{name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_obj, f, indent=2)
    LOGGER.info(f"Saved profile summary → {path}")
    return path
