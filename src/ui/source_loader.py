from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.source_data.preprocessing.feature_builder import build_features
from src.ui.common import get_run_id_from_session_state
from src.utils.data_io_utils import save_raw, list_raw_files, load_raw, save_processed, save_from_url, \
    latest_file_under_directory
from src.utils.log_utils import streamlit_safe, get_logger

LOGGER = get_logger("source_data_stager")

# -------------------------------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------------------------------
REQUIRED_KEYS = ["title_id"]

BASICS_SIG = {"primaryTitle", "titleType", "genres", "startYear"}
RATINGS_SIG = {"averageRating", "numVotes"}
AKAS_SIG = {"title", "region", "language", "types"}


# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------
def _ensure_staging() -> None:
    """Keep a session-local 'staged' map of RAW paths the user intends to use for the FM build."""
    st.session_state.setdefault("staged_raw", {})  # label -> full raw_path


def _quick_checks(df: pd.DataFrame) -> dict:
    """Return basic profiling info for a staged DataFrame, safely handling missing keys."""
    nan_pct = df.isna().mean().sort_values(ascending=False).head(10)
    dtypes = df.dtypes.astype(str)
    existing_keys = [k for k in REQUIRED_KEYS if k in df.columns]
    if existing_keys:
        dup_keys = int(df.duplicated(subset=existing_keys).sum())
        has_keys = True
    else:
        dup_keys = 0
        has_keys = False
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "nan_pct": nan_pct,
        "dtypes": dtypes,
        "has_keys": has_keys,
        "dup_keys": dup_keys,
    }


def _guess_role(name: str, df: pd.DataFrame) -> Optional[str]:
    lname = name.lower()
    cols = set(map(str, df.columns))
    if "basics" in lname or BASICS_SIG & cols:
        return "basics"
    if "ratings" in lname or RATINGS_SIG & cols:
        return "ratings"
    if "akas" in lname or AKAS_SIG & cols:
        return "akas"
    return None  # base


def _add_staged(label: str, raw_path: str) -> None:
    st.session_state["staged_raw"][label] = raw_path


def _auto_defaults(staged_labels: list[str], label_to_df: dict[str, pd.DataFrame]) -> dict[str, Optional[str]]:
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


def _load_df_from_cache(raw_path: str) -> pd.DataFrame:
    cache_key = f"_raw_preview::{raw_path}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    df = load_raw(raw_path)
    st.session_state[cache_key] = df
    return df


def _save_feature_master(df: pd.DataFrame, run_id: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = save_processed(df, run_id, f"feature_master_{ts}")
    st.session_state["last_feature_master_path"] = out_path
    st.session_state["df"] = df
    return out_path


def _latest_fm_for_run(run_id: str) -> Optional[Path]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    return latest_file_under_directory("feature_master_", run_proc, "cleaned")


# -------------------------------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------------------------------
@streamlit_safe
def render():
    run_id = get_run_id_from_session_state()
    st.header("Data Staging")
    _ensure_staging()

    raw_dir = Path(SETTINGS.RAW_DIR) / run_id

    # =============================================================================================
    # (0) AUTO-STAGE existing RAW files on resume using full paths
    # =============================================================================================
    if not st.session_state["staged_raw"]:
        try:
            files = list_raw_files(run_id)
            staged_now = 0
            if files:
                for f in files:
                    full_path = str(raw_dir / os.path.basename(str(f)))
                    if not os.path.exists(full_path):
                        continue
                    label = os.path.basename(full_path)
                    if label not in st.session_state["staged_raw"]:
                        _add_staged(label, full_path)
                        staged_now += 1
            if staged_now:
                st.info(f"Auto-staged {staged_now} RAW file(s) for run {run_id}.")
        except Exception as e:
            LOGGER.exception("Auto-stage failed")
            st.warning(f"Auto-stage skipped: {e}")

    # ---------------------------------------------------------------------------------------------
    # A) Add sources into RAW (upload / url)
    # ---------------------------------------------------------------------------------------------
    st.subheader("Add Raw Data Sources")
    mode = st.radio("Add", ["Upload Files", "Load from URL"], horizontal=True)

    if mode == "Upload Files":
        files = st.file_uploader(
            "Upload CSV/Parquet (multi-select)",
            type=["csv", "parquet", "pq"],
            accept_multiple_files=True,
        )
        if files:
            added = 0
            for f in files:
                try:
                    raw_path = save_raw(
                        pd.read_parquet(io.BytesIO(f.read())) if f.name.endswith((".parquet", ".pq")) else pd.read_csv(
                            f),
                        run_id,
                        os.path.splitext(f.name)[0],
                    )
                    _add_staged(f.name, raw_path)
                    added += 1
                except Exception as e:
                    st.error(f"Failed to save {f.name} to RAW: {e}")
            if added:
                st.success(f"Saved {added} file(s) to RAW and staged them.")
    else:
        url = st.text_input("Enter CSV/Parquet URL")
        if st.button("Load Data"):
            if not url:
                st.warning("Please enter a URL.")
            else:
                try:
                    raw_path = save_from_url(url, run_id)
                    label = os.path.basename(url)
                    _add_staged(label, raw_path)
                    st.success(f"Saved and staged: {label}")
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")

    # ---------------------------------------------------------------------------------------------
    # B) Add From Existing RAW Directory
    # ---------------------------------------------------------------------------------------------
    with st.expander("Add From Existing RAW Directory", expanded=False):
        try:
            files = list_raw_files(run_id)
            if not files:
                st.info("No files found in RAW.")
            else:
                labels = [os.path.basename(str(f)) for f in files]
                full_paths = [str(raw_dir / l) for l in labels]
                choices = dict(zip(labels, full_paths))

                pick = st.multiselect("Select RAW files to stage", labels)
                if st.button("Stage Selected RAW Files"):
                    for disp in pick:
                        if disp not in st.session_state["staged_raw"]:
                            _add_staged(disp, choices[disp])
                    st.success(f"Staged {len(pick)} RAW file(s).")
        except Exception:
            LOGGER.exception("Could not list raw data files for staging")
            st.error("Could not list raw data files for staging")

    if not st.session_state["staged_raw"]:
        st.info("No staged sources yet. Upload/add URLs or pick from RAW.")
    else:
        # -----------------------------------------------------------------------------------------
        # C) Light EDA for each staged RAW file
        # -----------------------------------------------------------------------------------------
        st.subheader("Data Insights (staged files)")
        staged_labels = list(st.session_state["staged_raw"].keys())
        label_to_df: Dict[str, pd.DataFrame] = {}
        for lbl in staged_labels:
            raw_path = st.session_state["staged_raw"][lbl]
            try:
                df = _load_df_from_cache(raw_path)
                label_to_df[lbl] = df
            except Exception as e:
                st.error(f"Failed reading RAW file for {lbl}: {e}")

        for lbl, df in label_to_df.items():
            checks = _quick_checks(df)
            with st.expander(f"{lbl} — {checks['rows']} rows, {checks['cols']} cols", expanded=False):
                tab_nan, tab_dtypes, tab_preview = st.tabs(["Percentage NaNs (10)", "Dtypes", "Preview"])
                with tab_nan:
                    st.dataframe(checks["nan_pct"], use_container_width=True)
                with tab_dtypes:
                    st.dataframe(checks["dtypes"].to_frame("dtype"), use_container_width=True)
                with tab_preview:
                    st.dataframe(df.head(10), use_container_width=True)
        st.markdown("---")

        # -----------------------------------------------------------------------------------------
        # D) Role mapping and build feature master
        # -----------------------------------------------------------------------------------------
        st.subheader("File Mapping for Feature Master")
        defaults = _auto_defaults(staged_labels, label_to_df)
        col1, col2 = st.columns(2)
        with col1:
            base_label = st.selectbox("Base file", staged_labels,
                                      index=staged_labels.index(defaults["base"]) if defaults[
                                                                                         "base"] in staged_labels else 0)
            basics_label = st.selectbox("Basics file", staged_labels,
                                        index=staged_labels.index(defaults["basics"]) if defaults[
                                                                                             "basics"] in staged_labels else 0)
        with col2:
            ratings_label = st.selectbox("Ratings file", staged_labels,
                                         index=staged_labels.index(defaults["ratings"]) if defaults[
                                                                                               "ratings"] in staged_labels else 0)
            akas_label = st.selectbox("Akas file", staged_labels,
                                      index=staged_labels.index(defaults["akas"]) if defaults[
                                                                                         "akas"] in staged_labels else 0)

        st.markdown("---")
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
                    st.success(f"Feature master created: {os.path.basename(saved)}")
                else:
                    st.warning("Builder did not return a valid parquet/DataFrame output.")
            except Exception as e:
                st.error(f"Feature build failed: {e}")

    # ---------------------------------------------------------------------------------------------
    # F) Maintenance + Existing FM banner
    # ---------------------------------------------------------------------------------------------
    with st.expander("Staging Maintenance", expanded=False):
        if st.button("Clear Staged Items", type="secondary"):
            st.session_state["staged_raw"].clear()
            st.info("Cleared all staged RAW references.")

    latest_fm = _latest_fm_for_run(run_id)
    if latest_fm:
        st.markdown("---")
        with st.container():
            st.success(f"Detected existing Feature Master: **{latest_fm.name}**")
            c1, c2 = st.columns([0.3, 0.7])
            with c1:
                if st.button("Use this Feature Master", key="use_existing_fm_bottom"):
                    try:
                        df = pd.read_parquet(latest_fm)
                        st.session_state["last_feature_master_path"] = str(latest_fm)
                        st.session_state["df"] = df
                        st.session_state["ingested"] = True
                        st.toast(f"Loaded {latest_fm.name} into session.", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to load {latest_fm.name}: {e}")
            with c2:
                st.caption("Rebuild from staged RAW above if you want to replace it.")
