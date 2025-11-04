from __future__ import annotations

import os
from datetime import datetime

import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.source_data.preprocessing.feature_builder import build_features
from src.utils.log_utils import get_logger, streamlit_safe

LOGGER = get_logger("ui_features")


def _list_processed_parquet_files() -> list[str]:
    processed = SETTINGS.PROCESSED_DIR
    if not os.path.isdir(processed):
        return []
    files = [f for f in os.listdir(processed) if f.endswith(".parquet")]
    files.sort()
    return files


def _validate_selections_ui(base: str, basics: str, ratings: str, akas: str) -> tuple[bool, str]:
    """Pure UI validation: uniqueness + 'tconst' presence in each chosen file."""
    # 1) All selected and unique
    chosen = [base, basics, ratings, akas]
    if any(x is None or x == "" for x in chosen):
        return False, "Please choose all four files."
    if len(set(chosen)) != 4:
        return False, "Each category must reference a different file."

    # 2) Ensure 'tconst' exists in each file
    #    Read just the 'tconst' column to avoid heavy loads
    # processed = SETTINGS.PROCESSED_DIR
    # for label, filename in [("Base", base), ("IMDB Basics", basics), ("IMDB Ratings", ratings), ("IMDB Akas", akas)]:
    #     path = os.path.join(processed, filename)
    #     try:
    #         # If parquet doesn't have tconst, this will raise
    #         _ = pd.read_parquet(path, columns=["tconst"])
    #     except Exception:
    #         return False, f"'{label}' file does not contain required 'tconst' column: {filename}"

    return True, ""


def _make_run_dir() -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SETTINGS.PROCESSED_DIR, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

@streamlit_safe
def render():
    st.header("Build Feature Master")

    files = _list_processed_parquet_files()
    if not files:
        st.warning("No processed parquet files found. Put your cleaned files in PROCESSED_DIR and refresh.")
        return

    col1, col2 = st.columns(2)
    with col1:
        base_file = st.selectbox("Base (synthetic generated)", files, index=None, placeholder="Select base file")
        basics_file = st.selectbox("IMDB Basics", files, index=None, placeholder="Select IMDB basics")
    with col2:
        ratings_file = st.selectbox("IMDB Ratings", files, index=None, placeholder="Select IMDB ratings")
        akas_file = st.selectbox("IMDB Akas", files, index=None, placeholder="Select IMDB akas")

    ok, msg = _validate_selections_ui(base_file, basics_file, ratings_file, akas_file)
    if not ok and any(x is not None for x in [base_file, basics_file, ratings_file, akas_file]):
        st.error(msg)

    build = st.button("Build Feature Master", type="primary", disabled=not ok)

    if build and ok:
        run_dir = _make_run_dir()
        out_path = os.path.join(run_dir, "feature_master.parquet")

        with st.spinner("Creating feature master…"):
            try:
                build_features(
                    base_file,
                    basics_file,
                    ratings_file,
                    akas_file
                )
                LOGGER.info(f"Feature master created → {out_path}")
            except Exception as e:
                st.error(f"Feature build failed: {e}")
                return

        st.session_state["last_run_dir"] = run_dir
        st.success(f"Feature master saved to: {out_path}")
        st.info("Proceed to **Analytical Tools – Model** to train on this run’s feature master.")
