from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.source_data.feature.feature_builder import label_staged_raw_files, build_features
from src.ui.common import get_run_id_from_session_state
from src.utils.data_io_utils import latest_file_under_directory, save_processed
from src.utils.log_utils import get_logger, streamlit_safe

LOGGER = get_logger("ui_features")

BASICS_SIG = {"primaryTitle", "titleType", "genres", "startYear"}
RATINGS_SIG = {"averageRating", "numVotes"}
AKAS_SIG = {"title", "region", "language", "types"}


def _guess_role(name: str, df: pd.DataFrame) -> Optional[str]:
    LOGGER.info(f"Guessing role: {name}")
    lname = name.lower()
    cols = set(map(str, df.columns))
    if "basics" in lname or BASICS_SIG & cols:
        return "basics"
    if "ratings" in lname or RATINGS_SIG & cols:
        return "ratings"
    if "akas" in lname or AKAS_SIG & cols:
        return "akas"
    return None  # base


def _auto_defaults(
        staged_labels: list[str],
        label_to_df: dict[str,
        pd.DataFrame]
) -> dict[str, Optional[str]]:
    LOGGER.info(f"Setting default using Staged labels: {staged_labels}")
    defaults = {"base": None, "basics": None, "ratings": None, "akas": None}
    for lbl in staged_labels:
        role = _guess_role(lbl, label_to_df[lbl])
        if role and defaults.get(role) is None:
            defaults[role] = lbl
    if defaults["base"] is None and staged_labels:
        for lbl in staged_labels:
            if lbl not in (defaults["basics"], defaults["ratings"], defaults["akas"]):
                defaults["base"] = lbl
                break
    for k in defaults:
        if defaults[k] is None and staged_labels:
            defaults[k] = staged_labels[0]
    return defaults


def _save_feature_master(df: pd.DataFrame, run_id: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = save_processed(df, run_id, f"feature_master_{ts}")
    st.session_state["last_feature_master_path"] = out_path
    st.session_state["df"] = df
    return out_path


def _latest_fm_for_run(run_id: str) -> Optional[Path]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    return latest_file_under_directory("feature_master_", run_proc, exclusion="cleaned")


@streamlit_safe
def render():
    run_id = get_run_id_from_session_state()
    st.caption("File Mapping for Feature Master")

    if not st.session_state["staged_raw"]:
        LOGGER.info("No staged raw data")
        st.session_state["last_feature_master_path"] = None
        st.info("No staged sources yet. Load data files first.")
    else:
        staged_labels, label_to_df = label_staged_raw_files()
        defaults = _auto_defaults(staged_labels, label_to_df)
        col1, col2 = st.columns(2)
        with col1:
            base_label = st.selectbox(
                "Base file",
                staged_labels,
                index=staged_labels.index(defaults["base"]) if defaults["base"] in staged_labels else 0)

            basics_label = st.selectbox(
                "Basics file",
                staged_labels,
                index=staged_labels.index(defaults["basics"]) if defaults["basics"] in staged_labels else 0)

        with col2:
            ratings_label = st.selectbox(
                "Ratings file",
                staged_labels,
                index=staged_labels.index(defaults["ratings"]) if defaults["ratings"] in staged_labels else 0)

            akas_label = st.selectbox(
                "Akas file",
                staged_labels,
                index=staged_labels.index(defaults["akas"]) if defaults["akas"] in staged_labels else 0)

        # st.markdown("---")
        if st.button("Build Feature Master", type="primary"):
            try:
                base_raw = st.session_state["staged_raw"][base_label]
                basics_raw = st.session_state["staged_raw"][basics_label]
                ratings_raw = st.session_state["staged_raw"][ratings_label]
                akas_raw = st.session_state["staged_raw"][akas_label]
                out = build_features(base_raw, basics_raw, ratings_raw, akas_raw)
                if isinstance(out, str) and os.path.exists(out):
                    st.session_state["last_feature_master_path"] = out
                    st.session_state["df"] = pd.read_parquet(out)
                    st.success(f"Feature master created: {os.path.basename(out)}")
                elif isinstance(out, pd.DataFrame):
                    saved = _save_feature_master(out, run_id)
                    st.success(f"Feature master created and set for subsequent use: {os.path.basename(saved)}")
                else:
                    st.warning("Builder did not return a valid parquet/DataFrame output.")
            except Exception as e:
                st.error(f"Feature build failed: {e}")

        # Display info for current feature master
        latest_fm = _latest_fm_for_run(run_id)
        st.markdown("---")
        if latest_fm:
            st.session_state["last_feature_master_path"] = latest_fm
            st.success(f"Current Feature Master being used for subsequent steps: **{latest_fm.name}**")
            st.caption("Rebuild from staged RAW above if you want to replace it.")
        else:
            st.session_state["last_feature_master_path"] = None
            st.info("No Feature Master found, create one using the button above")
