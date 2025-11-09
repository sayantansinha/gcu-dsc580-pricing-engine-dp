from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.ui.common import store_last_model_info_in_session
from src.utils.data_io_utils import latest_file_under_directory
from src.utils.log_utils import get_logger

# Orchestrated pages
from src.ui import source_loader
from src.ui.display_data import render_display_section
from src.ui.exploration import render_exploration_section
from src.ui.cleaning import render_cleaning_section
from src.ui.analytical_tools import render as render_models
from src.ui.visual_tools import render as render_visuals
from src.ui.report_generator import render as render_reports

LOGGER = get_logger("pipeline_hub")


def _artifacts(run_id: str) -> Dict[str, Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_clean = latest_file_under_directory("feature_master_cleaned_", run_proc)
    fm_raw = latest_file_under_directory("feature_master_", run_proc, "cleaned")
    model = None
    models_dir = Path(SETTINGS.MODELS_DIR) / run_id
    if models_dir.exists():
        any_files = [p for p in models_dir.iterdir() if p.is_file()]
        if any_files:
            model = any_files[0]
    if model is None and "last_model" in st.session_state:
        model = Path("__in_session__")
    return {"fm_raw": fm_raw, "fm_clean": fm_clean, "model": model}


def probe_feature_master_artifacts(run_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_raw = latest_file_under_directory("feature_master_", run_proc, "cleaned")
    fm_clean = latest_file_under_directory("feature_master_cleaned_", run_proc)
    return fm_raw, fm_clean


def probe_model_artifacts(run_id: str) -> Optional[Path]:
    models_dir = Path(SETTINGS.MODELS_DIR) / run_id
    return models_dir if models_dir.exists() and any(models_dir.iterdir()) else None


def _activate_feature_master(fm_clean: Optional[Path], fm_raw: Optional[Path]):
    # Raw Feature Master
    if fm_raw:
        try:
            df = pd.read_parquet(fm_raw)
            st.session_state["last_feature_master_path"] = str(fm_raw)
            st.session_state["df"] = df
        except Exception as e:
            st.warning(f"Could not load feature master {fm_raw.name}: {e}")
    else:
        LOGGER.info("No raw feature master found, use Build Feature Master first.")

    # Cleaned Feature Master
    if fm_clean:
        try:
            df = pd.read_parquet(fm_clean)
            st.session_state["last_cleaned_feature_master_path"] = str(fm_clean)
            st.session_state["cleaned_df"] = df
        except Exception as e:
            st.warning(f"Could not load clean feature master {fm_clean.name}: {e}")
    else:
        LOGGER.info("No clean feature master found, save a cleaned feature master first.")


def _activate_model(model_dir: Optional[Path]):
    preds_path = model_dir / "predictions.csv"
    avg_path = model_dir / "ensemble_avg.json"
    wgt_path = model_dir / "ensemble_weighted.json"
    pm_path = model_dir / "per_model_metrics.csv"
    params_path = model_dir / "params_map.json"

    if not preds_path.exists():
        LOGGER.info(f"No predictions.csv found under {model_dir}")
        return False

    try:
        # Load predictions
        pred_df = pd.read_csv(preds_path)
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        pred_src = pred_df["pred_source"].iloc[0] if "pred_source" in pred_df.columns and len(pred_df) else "unknown"

        # Load ensemble summaries (metrics already computed at train time)
        with open(avg_path, "r") as f:
            ensemble_avg = json.load(f) if avg_path.exists() else {}
        with open(wgt_path, "r") as f:
            ensemble_wgt = json.load(f) if wgt_path.exists() else {}

        # Load per-model metrics
        pm_metrics_df = pd.read_csv(pm_path)
        trained_models = pm_metrics_df["model"].dropna().astype(str).tolist()
        per_model_metrics = pm_metrics_df.to_dict(orient="records") if pm_path.exists() else []

        # Load hyperparams used
        params_map = {}
        if params_path.exists():
            with open(params_path, "r") as f:
                params_map = json.load(f)

        # Store in session
        store_last_model_info_in_session(
            {"per_model_metrics": per_model_metrics},
            ensemble_avg,
            ensemble_wgt,
            y_true,
            y_pred,
            pred_src,
            params_map,
            trained_models
        )
        st.session_state["last_model_run_dir"] = str(model_dir)
        return True
    except Exception as e:
        st.warning(f"Could not activate model run from disk: {e}")
        return False


@contextlib.contextmanager
def _suppress_child_section_panels():
    prev = st.session_state.get("_suppress_section_panel", False)
    st.session_state["_suppress_section_panel"] = True
    try:
        yield
    finally:
        st.session_state["_suppress_section_panel"] = prev


def render():
    LOGGER.info("Rendering pipeline_hub..")
    run_id = st.session_state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        st.error("run_id is not initialized. Ensure app.py sets it before loading the hub.")
        return

    st.header(f"Pipeline – {run_id}")

    # Probe and reload feature master artifacts
    fm_raw_path, fm_clean_path = probe_feature_master_artifacts(run_id)
    _activate_feature_master(fm_clean_path, fm_raw_path)
    LOGGER.info("Feature master artifacts loaded and session states activated")

    # Probe and reload model artifacts
    model_path = probe_model_artifacts(run_id)
    if model_path:
        _activate_model(model_path)

    # Status strip for clarity when resuming
    with st.container():
        fm_raw_label = fm_raw_path.name if fm_raw_path else "N/A"
        fm_clean_label = fm_clean_path.name if fm_clean_path else "N/A"
        model_flag = "available" if model_path else "N/A"
        st.markdown(
            f"> **Current artifacts** — Feature Master (raw): **{fm_raw_label}** "
            f"| Feature Master (cleaned): **{fm_clean_label}** |  Model: **{model_flag}**"
        )

    # Stage Sources & Build Feature (plain section -> child can use expanders)
    st.subheader("Stage Sources & Build Feature")
    source_loader.render()
    st.divider()

    # Display Data
    user_msg: str = "Build a Feature Master first."
    with st.expander("Display Data", expanded=False):
        if fm_raw_path:
            with _suppress_child_section_panels():
                render_display_section()
        else:
            st.info(user_msg)

    # Exploration (EDA)
    with st.expander("Exploration (EDA)", expanded=False):
        if fm_raw_path:
            with _suppress_child_section_panels():
                render_exploration_section()
        else:
            st.info(user_msg)

    # Cleaning & Preprocessing
    with st.expander("Preprocessing (and Cleaning)", expanded=False):
        if fm_raw_path:
            with _suppress_child_section_panels():
                render_cleaning_section()
        else:
            st.info(user_msg)

    # Re-probe after cleaning, only reading cleaned feature master
    _, fm_clean_path = probe_feature_master_artifacts(run_id)

    # Analytical Tools – Model
    with st.expander("Analytical Tools – Model", expanded=False):
        if fm_clean_path:
            with _suppress_child_section_panels():
                render_models()
        else:
            st.info("Save a cleaned Feature Master to enable modeling.")

    # Re-probe after modeling
    model_path = probe_model_artifacts(run_id)
    if st.session_state["last_model"] is None:
        _activate_model(model_path)

    # Visual Tools
    with st.expander("Visual Tools", expanded=False):
        if model_path:
            with _suppress_child_section_panels():
                render_visuals()
        else:
            st.info("Train a model to enable visual tools.")

    # Report Generator
    with st.expander("Report Generator", expanded=False):
        if model_path:
            with _suppress_child_section_panels():
                render_reports()
        else:
            st.info("Train a model to enable the report generator.")
