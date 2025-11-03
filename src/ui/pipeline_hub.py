from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
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


def _latest_under(prefix: str, under: Path) -> Optional[Path]:
    if not under.exists():
        return None
    files = [p for p in under.iterdir() if p.is_file() and p.name.startswith(prefix) and p.suffix == ".parquet"]
    if not files:
        return None
    files.sort(key=lambda p: p.name, reverse=True)
    return files[0]


def _artifacts(run_id: str) -> Dict[str, Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_clean = _latest_under("feature_master_clean_", run_proc)
    fm_raw = _latest_under("feature_master_", run_proc)
    model = None
    models_dir = run_proc / "models"
    if models_dir.exists():
        any_files = [p for p in models_dir.iterdir() if p.is_file()]
        if any_files:
            model = any_files[0]
    if model is None and "last_model" in st.session_state:
        model = Path("__in_session__")
    return {"fm_raw": fm_raw, "fm_clean": fm_clean, "model": model}


def _activate_df(best_fm: Optional[Path]):
    if not best_fm:
        return
    try:
        df = pd.read_parquet(best_fm)
        st.session_state["last_feature_master_path"] = str(best_fm)
        st.session_state["df"] = df
    except Exception as e:
        st.warning(f"Could not load feature master {best_fm.name}: {e}")


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
    if not isinstance(run_id, str) or not run_id.strip():
        st.error("run_id is not initialized. Ensure app.py sets it before loading the hub.")
        return

    st.header(f"Pipeline – {run_id}")

    arts = _artifacts(run_id)
    best_fm = arts["fm_clean"] or arts["fm_raw"]
    _activate_df(best_fm)

    # Status strip for clarity when resuming
    with st.container():
        fm_label = best_fm.name if best_fm else "—"
        model_flag = "available" if arts["model"] else "—"
        st.markdown(
            f"> **Current artifacts** — Feature Master: **{fm_label}**  |  Model: **{model_flag}**"
        )

    # 1) Stage Sources & Build Feature (plain section -> child can use expanders)
    st.subheader("Stage Sources & Build Feature")
    source_loader.render()
    st.divider()

    # 2) Display Data
    with st.expander("Display Data", expanded=False):
        if arts["fm_raw"] or arts["fm_clean"]:
            with _suppress_child_section_panels():
                render_display_section()
        else:
            st.info("Build a Feature Master first.")

    # 3) Exploration (EDA)
    with st.expander("Exploration (EDA)", expanded=False):
        if arts["fm_raw"] or arts["fm_clean"]:
            with _suppress_child_section_panels():
                render_exploration_section()
        else:
            st.info("Build a Feature Master first.")

    # 4) Cleaning & Preprocessing
    with st.expander("Cleaning & Preprocessing", expanded=False):
        if arts["fm_raw"] or arts["fm_clean"]:
            with _suppress_child_section_panels():
                render_cleaning_section()
        else:
            st.info("Build a Feature Master first.")

    # Re-probe after cleaning
    arts = _artifacts(run_id)

    # 5) Analytical Tools – Model
    with st.expander("Analytical Tools – Model", expanded=False):
        if arts["fm_clean"]:
            with _suppress_child_section_panels():
                render_models()
        else:
            st.info("Save a cleaned Feature Master to enable modeling.")

    # Re-probe after modeling
    arts = _artifacts(run_id)

    # 6) Visual Tools
    with st.expander("Visual Tools", expanded=False):
        if arts["model"]:
            with _suppress_child_section_panels():
                render_visuals()
        else:
            st.info("Train a model to enable visual tools.")

    # 7) Report Generator
    with st.expander("Report Generator", expanded=False):
        if arts["model"]:
            with _suppress_child_section_panels():
                render_reports()
        else:
            st.info("Train a model to enable the report generator.")
