from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.ui.common import store_last_model_info_in_session
from src.ui.pipeline_flow import render_pipeline_flow
from src.utils.data_io_utils import latest_file_under_directory
from src.utils.log_utils import get_logger

# Orchestrated pages
from src.ui.pipeline_steps import source_loader, features
from src.ui.pipeline_steps.display_data import render_display_section
from src.ui.pipeline_steps.exploration import render_exploration_section
from src.ui.pipeline_steps.cleaning import render_cleaning_section
from src.ui.pipeline_steps.analytical_tools import render as render_models
from src.ui.pipeline_steps.visual_tools import render as render_visuals
from src.ui.pipeline_steps.report_generator import render as render_reports

LOGGER = get_logger("pipeline_hub")


def _artifacts(run_id: str) -> Dict[str, Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_clean = latest_file_under_directory("feature_master_cleaned_", run_proc)
    fm_raw = latest_file_under_directory("feature_master_", run_proc, exclusion="cleaned")
    model = None
    models_dir = Path(SETTINGS.MODELS_DIR) / run_id
    if models_dir.exists():
        any_files = [p for p in models_dir.iterdir() if p.is_file()]
        if any_files:
            model = any_files[0]
    if model is None and "last_model" in st.session_state:
        model = Path("__in_session__")
    return {"fm_raw": fm_raw, "fm_clean": fm_clean, "model": model}


def _probe_feature_master_artifacts(run_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_raw = latest_file_under_directory("feature_master_", run_proc, exclusion="cleaned")
    fm_clean = latest_file_under_directory("feature_master_cleaned_", run_proc)
    return fm_raw, fm_clean


def _probe_model_artifacts(run_id: str) -> Optional[Path]:
    models_dir = Path(SETTINGS.MODELS_DIR) / run_id
    return models_dir if models_dir.exists() and any(models_dir.iterdir()) else None


def _activate_feature_master(fm_raw: Optional[Path], fm_clean: Optional[Path]):
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
            st.session_state["preprocessing_performed"] = True
        except Exception as e:
            st.warning(f"Could not load clean feature master {fm_clean.name}: {e}")
    else:
        st.session_state["preprocessing_performed"] = False
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
        st.session_state["model_trained"] = True
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
    run_id = st.session_state.get("run_id")
    LOGGER.info(f"Rendering pipeline_hub for run_id [{run_id}]")
    if not isinstance(run_id, str) or not run_id.strip():
        st.error("run_id is not initialized. Ensure app.py sets it before loading the hub.")
        return

    st.header(f"Pipeline – {run_id}")

    # Probe and reload feature master artifacts
    fm_raw_path, fm_clean_path = _probe_feature_master_artifacts(run_id)
    _activate_feature_master(fm_raw_path, fm_clean_path)
    LOGGER.info("Feature master artifacts loaded and session states activated")

    # Probe and reload model artifacts
    model_path = _probe_model_artifacts(run_id)
    if model_path:
        _activate_model(model_path)

    # Status strip for clarity when resuming
    with st.container(border=True):
        fm_raw_label = fm_raw_path.name if fm_raw_path else "N/A"
        fm_clean_label = fm_clean_path.name if fm_clean_path else "N/A"
        model_flag = "available" if model_path else "N/A"
        st.markdown(
            f"> **Current artifacts** — Feature Master (raw): **{fm_raw_label}** "
            f"| Feature Master (cleaned): **{fm_clean_label}** |  Model: **{model_flag}**"
        )

        # Reserve a fixed slot to render
        pipeline_flow_slot = st.empty()

        def _render_flow_diagram():
            ctx = {
                "files_staged": st.session_state.get("staged_files_count", 0) > 0,
                "feature_master_exists": st.session_state.get("last_feature_master_path") is not None,
                "data_displayed": st.session_state.get("data_displayed"),
                "eda_performed": st.session_state.get("eda_performed"),
                "preprocessing_performed": st.session_state.get("preprocessing_performed"),
                "model_trained": st.session_state.get("model_trained"),
                "report_generated": st.session_state.get("report_generated"),
            }

            with pipeline_flow_slot.container():
                render_pipeline_flow(ctx)

        _render_flow_diagram()

    # Stage Sources
    with st.expander("Data Staging", expanded=False):
        source_loader.render()

    _render_flow_diagram()

    # Build Feature
    with st.expander("Feature Master", expanded=False):
        features.render()

    _render_flow_diagram()

    # Display Data
    user_msg: str = "Build a Feature Master first."
    with st.expander("Display Data", expanded=False):
        if fm_raw_path:
            st.session_state["data_displayed"] = True
            with _suppress_child_section_panels():
                render_display_section()
        else:
            st.session_state["data_displayed"] = False
            st.info(user_msg)

    _render_flow_diagram()

    # Exploration (EDA)
    with st.expander("Exploration (EDA)", expanded=False):
        if fm_raw_path:
            st.session_state["eda_performed"] = True
            with _suppress_child_section_panels():
                render_exploration_section()
        else:
            st.session_state["eda_performed"] = False
            st.info(user_msg)

    _render_flow_diagram()

    # Cleaning & Preprocessing
    with st.expander("Preprocessing (and Cleaning)", expanded=False):
        if fm_raw_path:
            with _suppress_child_section_panels():
                render_cleaning_section()
        else:
            st.session_state["preprocessing_performed"] = False
            st.info(user_msg)

    _render_flow_diagram()

    # Re-probe after cleaning, only reading cleaned feature master
    _, fm_clean_path = _probe_feature_master_artifacts(run_id)

    # Analytical Tools – Model
    with st.expander("Analytical Tools – Model", expanded=False):
        if fm_clean_path:
            with _suppress_child_section_panels():
                render_models()
        else:
            st.session_state["model_trained"] = False
            st.info("Save a cleaned Feature Master to enable modeling.")

    # Re-probe after modeling
    model_path = _probe_model_artifacts(run_id)
    if st.session_state["last_model"] is None:
        _activate_model(model_path)

    _render_flow_diagram()

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
            st.session_state["report_generated"] = False
            st.info("Train a model to enable the report generator.")

    _render_flow_diagram()
